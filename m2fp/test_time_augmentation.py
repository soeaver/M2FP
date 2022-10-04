# Copyright (c) Facebook, Inc. and its affiliates.
import copy, cv2
import logging
import os
import sys
from itertools import count

import numpy as np
import torch
from fvcore.transforms import HFlipTransform
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from detectron2.data.detection_utils import read_image
from detectron2.data import MetadataCatalog
from detectron2.modeling import DatasetMapperTTA

from detectron2.data.transforms import (
    Resize,
    RandomFlip,
    apply_augmentations,
)
from pycocotools import mask as maskUtils
from detectron2.config import configurable
from detectron2.data import MetadataCatalog

from .utils.utils import predictions_merge


HUMAN_PARSING_DATASETS = ["cihp", "mhp", "lip", "ATR"]


class SingleHumanDatasetMapperTTA:
    """
    Implement test-time augmentation for detection data.
    It is a callable which takes a dataset dict from a detection dataset,
    and returns a list of dataset dicts where the images
    are augmented from the input image by the transformations defined in the config.
    This is used for test-time augmentation.
    """

    @configurable
    def __init__(self, flip: bool, scales):
        """
        Args:
            flip: whether to apply flipping augmentation
        """

        self.flip = flip
        self.scales = scales

    @classmethod
    def from_config(cls, cfg):
        return {
            "flip": cfg.TEST.AUG.FLIP,
            "scales": cfg.INPUT.SINGLE_HUMAN.TEST_SCALES
        }

    def __call__(self, dataset_dict):
        """
        Args:
            dict: a dict in standard model input format. See tutorials for details.

        Returns:
            list[dict]:
                a list of dicts, which contain augmented version of the input image.
                The total number of dicts is ``len(min_sizes) * (2 if flip else 1)``.
                Each dict has field "transforms" which is a TransformList,
                containing the transforms that are used to generate this image.
        """
        numpy_image = dataset_dict["image"].permute(1, 2, 0).numpy()

        h, w = numpy_image.shape[0], numpy_image.shape[1]
        aug_candidates = []
        for scale in self.scales:
            new_h, new_w = int(h * scale), int(w * scale)

            new_box = []
            for coord in dataset_dict["crop_box"]:
                new_box.append(int(coord * scale))

            aug_candidates.append(
                {
                    "augs": [Resize((new_h, new_w))],
                    "crop_box": new_box
                }
            )
            if self.flip:
                aug_candidates.append(
                    {
                        "augs": [Resize((new_h, new_w)), RandomFlip(prob=1.0)],
                        "crop_box": new_box
                    }
                )

        # Apply all the augmentations
        ret = []
        for aug in aug_candidates:
            new_image, tfms = apply_augmentations(aug["augs"], np.copy(numpy_image))
            torch_image = torch.from_numpy(np.ascontiguousarray(new_image.transpose(2, 0, 1)))

            dic = copy.deepcopy(dataset_dict)
            dic["transforms"] = tfms
            dic["image"] = torch_image
            dic["crop_box"] = aug["crop_box"]
            ret.append(dic)
        return ret


class SemanticSegmentorWithTTA(nn.Module):
    """
    A SemanticSegmentor with test-time augmentation enabled.
    Its :meth:`__call__` method has the same interface as :meth:`SemanticSegmentor.forward`.
    """

    def __init__(self, cfg, model, tta_mapper=None, batch_size=1):
        """
        Args:
            cfg (CfgNode):
            model (SemanticSegmentor): a SemanticSegmentor to apply TTA on.
            tta_mapper (callable): takes a dataset dict and returns a list of
                augmented versions of the dataset dict. Defaults to
                `DatasetMapperTTA(cfg)`.
            batch_size (int): batch the augmented images into this batch size for inference.
        """
        super().__init__()
        if isinstance(model, DistributedDataParallel):
            model = model.module
        self.cfg = cfg.clone()
        self.model = model

        if cfg.DATASETS.TEST[0].split('_')[0] in HUMAN_PARSING_DATASETS:
            self.human_semseg = True
            self.flip_map = MetadataCatalog.get(cfg.DATASETS.TEST[0]).flip_map
        else:
            self.human_semseg = False

        if tta_mapper is None:
            if cfg.INPUT.SINGLE_HUMAN.ENABLED:
                tta_mapper = SingleHumanDatasetMapperTTA(cfg)
            else:
                tta_mapper = DatasetMapperTTA(cfg)

        self.tta_mapper = tta_mapper
        self.batch_size = batch_size

    def __call__(self, batched_inputs):
        """
        Same input/output format as :meth:`SemanticSegmentor.forward`
        """

        def _maybe_read_image(dataset_dict):
            ret = copy.copy(dataset_dict)
            if "image" not in ret:
                image = read_image(ret.pop("file_name"), self.model.input_format)
                image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))  # CHW
                ret["image"] = image
            if "height" not in ret and "width" not in ret:
                ret["height"] = image.shape[1]
                ret["width"] = image.shape[2]
            return ret

        processed_results = []
        for x in batched_inputs:
            result = self._inference_one_image(_maybe_read_image(x))
            processed_results.append(result)
        return processed_results

    def _inference_one_image(self, input):
        """
        Args:
            input (dict): one dataset dict with "image" field being a CHW tensor
        Returns:
            dict: one output dict
        """
        orig_shape = (input["height"], input["width"])
        augmented_inputs, tfms = self._get_augmented_inputs(input)

        final_predictions = None
        count_predictions = 0
        for input, tfm in zip(augmented_inputs, tfms):
            count_predictions += 1

            with torch.no_grad():
                if final_predictions is None:
                    if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
                        predictions = self.model([input])[0].pop("sem_seg")

                        if self.human_semseg:
                            final_predictions = self.flip_human_semantic_back(predictions)
                        else:
                            final_predictions = predictions.flip(dims=[2])
                    else:
                        final_predictions = self.model([input])[0].pop("sem_seg")
                else:
                    if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
                        predictions = self.model([input])[0].pop("sem_seg")

                        if self.human_semseg:
                            final_predictions += self.flip_human_semantic_back(predictions)
                        else:
                            final_predictions += predictions.flip(dims=[2])
                    else:
                        final_predictions += self.model([input])[0].pop("sem_seg")

        final_predictions = final_predictions / count_predictions
        return {"sem_seg": final_predictions}

    def _get_augmented_inputs(self, input):
        augmented_inputs = self.tta_mapper(input)
        tfms = [x.pop("transforms") for x in augmented_inputs]
        return augmented_inputs, tfms

    def flip_human_semantic_back(self, predictions):
        spatial_flipback_predictions = predictions.flip(dims=[2])
        channel_flipback_predictions = copy.deepcopy(spatial_flipback_predictions)

        # channel transaction to flip human part label
        for ori_label, new_label in self.flip_map:
            org_channel = spatial_flipback_predictions[ori_label, :, :]
            new_channel = spatial_flipback_predictions[new_label, :, :]

            channel_flipback_predictions[new_label, :, :] = org_channel
            channel_flipback_predictions[ori_label, :, :] = new_channel

        return channel_flipback_predictions


class ParsingWithTTA(nn.Module):
    """
    A SemanticSegmentor with test-time augmentation enabled.
    Its :meth:`__call__` method has the same interface as :meth:`SemanticSegmentor.forward`.
    """

    def __init__(self, cfg, model, tta_mapper=None, batch_size=1):
        """
        Args:
            cfg (CfgNode):
            model (SemanticSegmentor): a SemanticSegmentor to apply TTA on.
            tta_mapper (callable): takes a dataset dict and returns a list of
                augmented versions of the dataset dict. Defaults to
                `DatasetMapperTTA(cfg)`.
            batch_size (int): batch the augmented images into this batch size for inference.
        """
        super().__init__()
        if isinstance(model, DistributedDataParallel):
            model = model.module
        self.cfg = cfg.clone()
        self.flip_map = MetadataCatalog.get(self.cfg.DATASETS.TEST[0]).flip_map
        self.model = model

        if tta_mapper is None:
            if cfg.INPUT.SINGLE_HUMAN.ENABLED:
                tta_mapper = SingleHumanDatasetMapperTTA(cfg)
            else:
                tta_mapper = DatasetMapperTTA(cfg)

        self.tta_mapper = tta_mapper
        self.batch_size = batch_size

        self.num_parsing = MetadataCatalog.get(self.cfg.DATASETS.TEST[0]).num_parsing

    def __call__(self, batched_inputs):
        """
        Same input/output format as :meth:`SemanticSegmentor.forward`
        """

        def _maybe_read_image(dataset_dict):
            ret = copy.copy(dataset_dict)
            if "image" not in ret:
                image = read_image(ret.pop("file_name"), self.model.input_format)
                image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))  # CHW
                ret["image"] = image
            if "height" not in ret and "width" not in ret:
                ret["height"] = image.shape[1]
                ret["width"] = image.shape[2]
            return ret

        processed_results = []
        for x in batched_inputs:
            result = self._inference_one_image(_maybe_read_image(x))
            processed_results.append(result)
        return processed_results

    def _inference_one_image(self, input):
        """
        Args:
            input (dict): one dataset dict with "image" field being a CHW tensor
        Returns:
            dict: one output dict
        """
        orig_shape = (input["height"], input["width"])
        augmented_inputs, tfms = self._get_augmented_inputs(input)

        all_semantic_predictions = []
        all_part_predictions = {}
        all_human_predictions = {}

        for aug_input, tfm in zip(augmented_inputs, tfms):
            with torch.no_grad():
                model_out = self.model([aug_input])[0].pop("parsing")

                if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
                    flipped_semantic_predictions = model_out["semantic_outputs"]
                    flipped_part_predictions = model_out["part_outputs"]
                    flipped_human_predictions = model_out["human_outputs"]

                    semantic_predictions = self.flip_semantic_back(flipped_semantic_predictions)
                    part_predictions = self.flip_instance_back(flipped_part_predictions)
                    human_predictions = self.flip_instance_back(flipped_human_predictions, "human") \
                        if len(flipped_human_predictions) > 0 else []

                else:
                    semantic_predictions = model_out["semantic_outputs"]
                    part_predictions = model_out["part_outputs"]
                    human_predictions = model_out["human_outputs"]

                # collect aug prediction
                # semantic prediction
                all_semantic_predictions.append(semantic_predictions)

                # store the part and human prediction by category
                for part_prediction in part_predictions:
                    if part_prediction["category_id"] not in all_part_predictions:
                        all_part_predictions[part_prediction["category_id"]] = {"masks": [], "scores": []}

                    all_part_predictions[part_prediction["category_id"]]["masks"].append(
                        part_prediction["mask"].cpu().numpy())
                    all_part_predictions[part_prediction["category_id"]]["scores"].append(part_prediction["score"])

                for human_prediction in human_predictions:
                    if human_prediction["category_id"] not in all_human_predictions:
                        all_human_predictions[human_prediction["category_id"]] = {"masks": [], "scores": []}

                    all_human_predictions[human_prediction["category_id"]]["masks"].append(
                        human_prediction["mask"].cpu().numpy())
                    all_human_predictions[human_prediction["category_id"]]["scores"].append(human_prediction["score"])

        # merge predictions from different augmentations
        # semantic prediction
        all_semantic_predictions = torch.stack(all_semantic_predictions).transpose(1, 0)
        final_semantic_predictions = torch.mean(all_semantic_predictions, dim=1)

        # part and human instance predictions
        final_part_predictions = predictions_merge(all_part_predictions, device=final_semantic_predictions.device)
        final_human_predictions = predictions_merge(all_human_predictions, device=final_semantic_predictions.device)

        return {
            "parsing": {
                "semantic_outputs": final_semantic_predictions,
                "part_outputs": final_part_predictions,
                "human_outputs": final_human_predictions
            }
        }

    def _get_augmented_inputs(self, input):
        augmented_inputs = self.tta_mapper(input)
        tfms = [x.pop("transforms") for x in augmented_inputs]
        return augmented_inputs, tfms

    def flip_semantic_back(self, predictions):
        spatial_flipback_predictions = predictions.flip(dims=[2])
        channel_flipback_predictions = copy.deepcopy(spatial_flipback_predictions)

        # channel transaction to flip human part label
        for ori_label, new_label in self.flip_map:
            org_channel = spatial_flipback_predictions[ori_label, :, :]
            new_channel = spatial_flipback_predictions[new_label, :, :]

            channel_flipback_predictions[new_label, :, :] = org_channel
            channel_flipback_predictions[ori_label, :, :] = new_channel

        return channel_flipback_predictions

    def flip_instance_back(self, predictions, instance_type="part"):
        for prediction in predictions:
            prediction["mask"] = prediction["mask"].flip(dims=[1])
            if instance_type in ["part"]:
                flip_map_dict = {}
                for (k, v) in self.flip_map:
                    flip_map_dict.update({k: v, v: k})
                if prediction["category_id"] in flip_map_dict:
                    prediction["category_id"] = flip_map_dict[prediction["category_id"]]
        return predictions
