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
    RandomFlip,
    apply_augmentations,
)
from pycocotools import mask as maskUtils
from detectron2.config import configurable
from detectron2.data import MetadataCatalog

from .utils.utils import predictions_supress, predictions_merge, parsing_nms

__all__ = [
    "ParsingSemanticSegmentorWithTTA",
    "SingleHumanDatasetMapperTTA",
    "SemanticSegmentorWithTTA",
]

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
    def __init__(self, flip: bool):
        """
        Args:
            flip: whether to apply flipping augmentation
        """

        self.flip = flip

    @classmethod
    def from_config(cls, cfg):
        return {
            "flip": cfg.TEST.AUG.FLIP,
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

        # Create all combinations of augmentations to use
        aug_candidates = [[]]  # each element is a list[Augmentation]
        if self.flip:
            flip = RandomFlip(prob=1.0)
            aug_candidates.append([flip])  # resize + flip

        # Apply all the augmentations
        ret = []
        for aug in aug_candidates:
            new_image, tfms = apply_augmentations(aug, np.copy(numpy_image))
            torch_image = torch.from_numpy(np.ascontiguousarray(new_image.transpose(2, 0, 1)))

            dic = copy.deepcopy(dataset_dict)
            dic["transforms"] = tfms
            dic["image"] = torch_image
            ret.append(dic)
        return ret


class ParsingWithTTA(nn.Module):
    """
    A SemanticSegmentor with test-time augmentation enabled.
    Its :meth:`__call__` method has the same interface as :meth:`SemanticSegmentor.forward`.
    """

    def __init__(self, cfg, model, tta_mapper=None, batch_size=1, merge_mode='supress'):
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
        self.merge_mode = merge_mode

        if tta_mapper is None:
            if cfg.INPUT.SINGLE_HUMAN.ENABLED:
                tta_mapper = SingleHumanDatasetMapperTTA(cfg)
            else:
                tta_mapper = DatasetMapperTTA(cfg)

        self.tta_mapper = tta_mapper
        self.batch_size = batch_size

        self.num_parsing = MetadataCatalog.get(self.cfg.DATASETS.TEST[0]).num_parisng

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

        final_semantic_predictions = None
        all_part_predictions = {}
        all_human_predictions = {}
        parsings = []
        parsing_instance_scores = []

        count_predictions = 0
        for aug_input, tfm in zip(augmented_inputs, tfms):
            count_predictions += 1
            with torch.no_grad():
                model_out = self.model([aug_input])[0].pop("parsing")

                if final_semantic_predictions is None:
                    if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
                        flipped_semantic_predictions = model_out['semseg_outputs']
                        flipped_part_predictions = model_out['part_outputs']
                        flipped_human_predictions = model_out['human_outputs']
                        flipped_parsing_predictions = model_out['parsing_outputs']

                        semantic_predictions = self.flip_semantic_back(flipped_semantic_predictions)
                        part_predictions = self.flip_instance_back(flipped_part_predictions)
                        human_predictions = self.flip_instance_back(flipped_human_predictions, 'human') \
                            if len(flipped_human_predictions) > 0 else []
                        parsing_predictions = self.flip_parsing_back(flipped_parsing_predictions) \
                            if len(flipped_parsing_predictions) > 0 else []
                    else:
                        semantic_predictions = model_out['semseg_outputs']
                        part_predictions = model_out['part_outputs']
                        human_predictions = model_out['human_outputs']
                        parsing_predictions = model_out['parsing_outputs']
                else:
                    if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
                        flipped_semantic_predictions = model_out['semseg_outputs']
                        flipped_part_predictions = model_out['part_outputs']
                        flipped_human_predictions = model_out['human_outputs']
                        flipped_parsing_predictions = model_out['parsing_outputs']

                        semantic_predictions += self.flip_semantic_back(flipped_semantic_predictions)
                        part_predictions = self.flip_instance_back(flipped_part_predictions)
                        human_predictions = self.flip_instance_back(flipped_human_predictions, 'human') \
                            if len(flipped_human_predictions) > 0 else []
                        parsing_predictions = self.flip_parsing_back(flipped_parsing_predictions) \
                            if len(flipped_parsing_predictions) > 0 else []
                    else:
                        semantic_predictions += model_out['semseg_outputs']
                        part_predictions = model_out['part_outputs']
                        human_predictions = model_out['human_outputs']
                        parsing_predictions = model_out['parsing_outputs']

                # store the part and human prediction by category
                for part_prediction in part_predictions:
                    if part_prediction['category_id'] not in all_part_predictions:
                        all_part_predictions[part_prediction['category_id']] = {'masks': [], 'scores': []}

                    all_part_predictions[part_prediction['category_id']]['masks'].append(part_prediction['mask'])
                    all_part_predictions[part_prediction['category_id']]['scores'].append(part_prediction['score'])

                for human_prediction in human_predictions:
                    if human_prediction['category_id'] not in all_human_predictions:
                        all_human_predictions[human_prediction['category_id']] = {'masks': [], 'scores': []}

                    all_human_predictions[human_prediction['category_id']]['masks'].append(human_prediction['mask'])
                    all_human_predictions[human_prediction['category_id']]['scores'].append(human_prediction['score'])

                # store parsings
                for parsing_prediction in parsing_predictions:
                    parsings.append(parsing_prediction['parsing'])
                    parsing_instance_scores.append(parsing_prediction['instance_score'])

        # merge predictions from different augmentations
        final_semantic_predictions = semantic_predictions.cpu() / count_predictions
        if self.merge_mode == "supress":
            final_part_predictions = predictions_supress(all_part_predictions)
            final_human_predictions = predictions_supress(all_human_predictions) \
                if len(all_human_predictions) > 0 else []
        elif self.merge_mode == "merge":
            final_part_predictions = predictions_merge(all_part_predictions)
            final_human_predictions = predictions_merge(all_human_predictions) \
                if len(all_human_predictions) > 0 else []
        else:
            raise NotImplementedError(
                "Have not implement other merge method, e.g. simply collect al results."
            )

        final_parsing_predictions = []
        final_parsings, final_parsing_instance_scores = parsing_nms(
            np.array(parsings), np.array(parsing_instance_scores), num_parsing=self.num_parsing
        ) if len(parsing_instance_scores) > 0 else ([], [])
        for final_parsing, final_parsing_instance_score in zip(final_parsings, final_parsing_instance_scores):
            final_parsing_predictions.append(
                {
                    "parsing": final_parsing,
                    "instance_score": final_parsing_instance_score,
                }
            )

        return {
            "parsing": {
                "semseg_outputs": final_semantic_predictions,
                "parsing_outputs": final_parsing_predictions,
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

    def flip_instance_back(self, predictions, instance_type='part'):
        for prediction in predictions:
            prediction['mask'] = prediction['mask'].flip(dims=[1])
            if instance_type in ['part']:
                flip_map_dict = {}
                for (k, v) in self.flip_map:
                    flip_map_dict.update({k: v, v: k})
                if prediction['category_id'] in flip_map_dict:
                    prediction['category_id'] = flip_map_dict[prediction['category_id']]
        return predictions

    def flip_parsing_back(self, predictions):
        for prediction in predictions:
            prediction['parsing'] = prediction['parsing'].flip(dims=[1])

            for r, l in self.flip_map:
                parsing_tmp = copy.deepcopy(prediction['parsing'])
                prediction['parsing'][parsing_tmp == r] = l
                prediction['parsing'][parsing_tmp == l] = r
        return predictions


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
