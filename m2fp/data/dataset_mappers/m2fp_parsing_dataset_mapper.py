# Copyright (c) Facebook, Inc. and its affiliates.
import copy, cv2, os
import copy, cv2, os
import logging

import numpy as np
import pycocotools.mask as mask_util
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances, polygons_to_bitmask

from ..parsing_utils import read_semseg_gt, gen_parsing_instances


__all__ = ["M2FPParsingDatasetMapper"]


class M2FPParsingDatasetMapper:
    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        size_divisibility,
        num_parsing,
        flip_map,
        with_human_instance,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.size_divisibility = size_divisibility
        self.num_parsing = num_parsing
        self.flip_map = flip_map
        self.with_human_instance = with_human_instance

        assert self.is_train, "M2FPParsingDatasetMapper should only be used for training!"
        logger = logging.getLogger(__name__)
        logger.info(f"[{self.__class__.__name__}] Augmentations used in training: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        # for multi person human parsing, e.g. CIHP and MHP
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            ),
            T.RandomFlip()
        ]

        meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "flip_map": meta.flip_map,
            "num_parsing": meta.num_parsing,
            "with_human_instance": cfg.MODEL.M2FP.WITH_HUMAN_INSTANCE,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # read category and human gt files
        # PyTorch transformation not implemented for uint16, so converting it to double first
        human_gt = read_semseg_gt(dataset_dict.pop("human_file_name")).astype("double")
        category_gt = read_semseg_gt(dataset_dict.pop("category_file_name")).astype("double")

        # apply transforms to image
        aug_input = T.AugInput(image)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image

        if isinstance(transforms, (tuple, list)):
            transforms = T.TransformList(transforms)
            
        human_gt = transforms.apply_segmentation(human_gt)
        category_gt = transforms.apply_segmentation(category_gt)

        # flip pixel labels of human parts
        do_hflip = sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1
        if do_hflip:
            for ori_label, new_label in self.flip_map:
                left = category_gt == ori_label
                right = category_gt == new_label
                category_gt[left] = new_label
                category_gt[right] = ori_label

        # generate instance labels and masks
        classes, masks = gen_parsing_instances(
            human_gt, category_gt, self.with_human_instance, self.num_parsing
        )

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        masks = [torch.from_numpy(np.ascontiguousarray(x)) for x in masks]

        classes = torch.tensor(classes, dtype=torch.int64)
        assert bool(torch.all(classes >= 0))

        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            # pad image
            image = F.pad(image, padding_size, value=128).contiguous()
            # pad mask
            masks = [F.pad(x, padding_size, value=0).contiguous() for x in masks]

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image

        # Prepare per-category binary masks
        instances = Instances(image_shape)
        instances.gt_classes = classes
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros((0, image.shape[-2], image.shape[-1]))
        else:
            masks = BitMasks(torch.stack(masks))
            instances.gt_masks = masks.tensor

        dataset_dict["instances"] = instances

        return dataset_dict
