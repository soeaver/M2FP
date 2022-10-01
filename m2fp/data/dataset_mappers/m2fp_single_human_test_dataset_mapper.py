# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch
import random, cv2, os
from PIL import Image

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances

from ..parsing_utils import center_to_target_size_test

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["M2FPSingleHumanTestDatasetMapper"]


class M2FPSingleHumanTestDatasetMapper:
    @configurable
    def __init__(
        self,
        is_train,
        *,
        image_format,
        test_size,
        size_divisibility,
    ):
        # fmt: off
        self.is_train = is_train
        self.image_format = image_format
        self.test_size = test_size
        self.size_divisibility = size_divisibility

        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(
            f"[M2FPSingleHumanTestDatasetMapper] Augmentations used in {mode}: self.implemented center_to_target_size"
        )

    @classmethod
    def from_config(cls, cfg, is_train=False):
        dataset_names = cfg.DATASETS.TEST
        meta = MetadataCatalog.get(dataset_names[0])

        ret = {
            "is_train": is_train,
            "image_format": cfg.INPUT.FORMAT,
            "test_size": cfg.INPUT.SINGLE_HUMAN.SIZES[0],
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,  # -1
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert not self.is_train, "MaskFormerSingleHumanTestDatasetMapper should only be used for testing!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        image, crop_box = center_to_target_size_test(image, self.test_size)

        # image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["crop_box"] = crop_box

        # USER: Modify this if you want to keep them for some reason.
        dataset_dict.pop("annotations", None)
        dataset_dict.pop("sem_seg_file_name", None)
        return dataset_dict
