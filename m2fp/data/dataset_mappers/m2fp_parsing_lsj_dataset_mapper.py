# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging

import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances

from pycocotools import mask as coco_mask

from ..parsing_utils import read_semseg_gt, gen_parsing_instances
from ..transforms.augmentation_impl import RandomCenterRotation


__all__ = ["M2FPParsingLSJDatasetMapper"]


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    if cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )

    if cfg.INPUT.COLOR_AUG_SSD:
        augmentation.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))

    if cfg.INPUT.ROTATION:
        augmentation.append(RandomCenterRotation(cfg.INPUT.ROTATION))

    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])

    return augmentation


# This is specifically designed for the COCO dataset.
class M2FPParsingLSJDatasetMapper:
    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
        num_parsing,
        flip_map,
        with_human_instance,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.is_train = is_train
        self.tfm_gens = tfm_gens
        self.img_format = image_format
        self.num_parsing = num_parsing
        self.flip_map = flip_map
        self.with_human_instance = with_human_instance

        assert self.is_train, "M2FPParsingLSJDatasetMapper should only be used for training!"
        logger = logging.getLogger(__name__)
        logger.info(f"[{self.__class__.__name__}] Augmentations used in training: {tfm_gens}")
    
    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
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

        # generate instance labels and masks
        class_ids, masks = gen_parsing_instances(
            human_gt, category_gt, self.with_human_instance, self.num_parsing
        )

        # apply transforms to image
        aug_input = T.AugInput(image)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image

        if isinstance(transforms, (tuple, list)):
            transforms = T.TransformList(transforms)
        # whether the transformation flip pixel labels of human parts
        do_hflip = sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1

        # apply transforms to instances
        tfmd_masks = []
        tfmd_classes = []
        for class_id, mask in zip(class_ids, masks):
            tfmd_mask = transforms.apply_segmentation(mask.astype("double")).astype(np.int64)

            tfmd_class_id = class_id
            if do_hflip:
                for ori_label, new_label in self.flip_map:
                    if tfmd_class_id == ori_label:
                        tfmd_class_id = new_label
                    elif tfmd_class_id == new_label:
                        tfmd_class_id = ori_label
            # filter empty instances(due to augmentation)
            if int(np.max(tfmd_mask)) != 0:
                tfmd_masks.append(tfmd_mask)
                tfmd_classes.append(tfmd_class_id)

        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        masks = [torch.from_numpy(np.ascontiguousarray(x)) for x in tfmd_masks]
        classes = torch.tensor(tfmd_classes, dtype=torch.int64)
        assert bool(torch.all(classes >= 0))

        # Prepare per-category binary masks
        image_shape = (image.shape[-2], image.shape[-1])  # h, w
        instances = Instances(image_shape)
        instances.gt_classes = classes
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros((0, image.shape[-2], image.shape[-1]))
        else:
            masks = BitMasks(torch.stack(masks))
            instances.gt_masks = masks.tensor

        dataset_dict["image"] = image
        dataset_dict["instances"] = instances

        return dataset_dict
