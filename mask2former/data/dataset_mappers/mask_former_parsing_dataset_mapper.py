# Copyright (c) Facebook, Inc. and its affiliates.
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

from ..parsing_utils import filter_instance_by_attributes, transform_parsing_instance_annotations,\
    affine_to_target_size, center_to_target_size_parsing
from ..transforms.augmentation_impl import ResizeByAspectRatio, ResizeByScale, RandomCenterRotation

__all__ = ["MaskFormerParsingDatasetMapper"]


class MaskFormerParsingDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for instance segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        size_divisibility,
        flip_map,
        with_human_instance,
        with_bkg_instance,
        single_human_aug,
        train_size
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
        self.flip_map = flip_map
        self.with_human_instance = with_human_instance
        self.with_bkg_instance = with_bkg_instance
        self.single_human_aug = single_human_aug

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

        if self.single_human_aug:
            self.train_size = train_size

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # decide whether to parse multi person
        single_human_aug = False
        train_size = None

        # Build augmentation
        if "lip" in cfg.DATASETS.TRAIN[0]:
            # for single person human parsing, e.g. LIP and ATR
            single_human_aug = True

            train_size = cfg.INPUT.SINGLE_PARSING.SCALES[0]
            scale_factor = cfg.INPUT.SINGLE_PARSING.SCALE_FACTOR

            augs = [
                T.RandomFlip(),
                ResizeByScale(scale_factor)
            ]

            if cfg.INPUT.SINGLE_PARSING.ROTATION:
                rot_factor = cfg.INPUT.SINGLE_PARSING.ROT_FACTOR
                augs.append(
                    RandomCenterRotation(rot_factor)
                )
        else:
            # for multi person human parsing, e.g. CIHP and MHP
            augs = [
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                )
            ]
            if cfg.INPUT.COLOR_AUG_SSD:
                augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
            augs.append(T.RandomFlip())

        meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "flip_map": meta.flip_map,
            "with_human_instance": cfg.MODEL.MASK_FORMER.TEST.PARSING.WITH_HUMAN_INSTANCE,
            "with_bkg_instance": cfg.MODEL.MASK_FORMER.TEST.PARSING.WITH_BKG_INSTANCE,
            "single_human_aug": single_human_aug,
            "train_size": train_size
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MaskFormerPanopticDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # filter instance according to attributes and config, maybe change "category_id" of instance
        dataset_dict = filter_instance_by_attributes(dataset_dict, self.with_human_instance, self.with_bkg_instance)

        aug_input = T.AugInput(image)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image

        # transform instance masks
        assert "annotations" in dataset_dict
        for anno in dataset_dict["annotations"]:
            anno.pop("keypoints", None)

        annos = [
            transform_parsing_instance_annotations(
                obj, transforms, image.shape[:2],
                flip_map=self.flip_map,
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]

        if self.single_human_aug:
            image, annos = center_to_target_size_parsing(image, annos, self.train_size)

        if len(annos):
            assert "segmentation" in annos[0]
        segms = [obj["segmentation"] for obj in annos]
        masks = []
        for segm in segms:
            if isinstance(segm, list):
                # polygon
                masks.append(polygons_to_bitmask(segm, *image.shape[:2]))
            elif isinstance(segm, dict):
                # COCO RLE
                masks.append(mask_util.decode(segm))
            elif isinstance(segm, np.ndarray):
                assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                    segm.ndim
                )
                # mask array
                masks.append(segm)
            else:
                raise ValueError(
                    "Cannot convert segmentation of type '{}' to BitMasks!"
                    "Supported types are: polygons as list[list[float] or ndarray],"
                    " COCO-style RLE as a dict, or a binary segmentation mask "
                    " in a 2D numpy array of shape HxW.".format(type(segm))
                )

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        masks = [torch.from_numpy(np.ascontiguousarray(x)) for x in masks]

        classes = [int(obj["category_id"]) for obj in annos]
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
