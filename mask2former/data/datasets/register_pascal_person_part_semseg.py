# Copyright (c) Facebook, Inc. and its affiliates.
import json
import logging
import numpy as np
import os
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_sem_seg

PASCAL_PERSON_PART_SEMSEG_CATEGORIES = [
    "Background",
    "Lower-arm",
    "Head",
    "Upper-leg",
    "Torso",
    "Lower-leg",
    "Upper-arm",
]

_PREDEFINED_SPLITS = {
    "pascal_person_part_semseg_train": ("Training/Images/", "Training/Category_ids/"),
    "pascal_person_part_semseg_test": ("Testing/Images/", "Testing/Category_ids/"),
}

PASCAL_PERSON_PART_FLIP_MAP = ()


def register_pascal_person_part_semseg(root):
    root = os.path.join(root, "pascal-person-part")
    for name, (image_dir, gt_dir) in _PREDEFINED_SPLITS.items():
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=PASCAL_PERSON_PART_SEMSEG_CATEGORIES[:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            flip_map=PASCAL_PERSON_PART_FLIP_MAP
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_pascal_person_part_semseg(_root)
