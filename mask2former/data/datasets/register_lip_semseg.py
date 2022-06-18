# Copyright (c) Facebook, Inc. and its affiliates.
import json
import logging
import numpy as np
import os
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_sem_seg

LIP_SEMSEG_CATEGORIES = [
    'Background',
    'Hat',
    'Hair',
    'Glove',
    'Sunglasses',
    'UpperClothes',
    'Dress',
    'Coat',
    'Socks',
    'Pants',
    'Jumpsuits',
    'Scarf',
    'Skirt',
    'Face',
    'Left-arm',
    'Right-arm',
    'Left-leg',
    'Right-leg',
    'Left-shoe',
    'Right-shoe'
]

# ==== Predefined splits for raw LIP images ===========
_PREDEFINED_SPLITS = {
    "lip_semseg_train": ("Training/Images/", "Training/Category_ids/"),
    "lip_semseg_val": ("Validation/Images/", "Validation/Category_ids/"),
}

LIP_FLIP_MAP = ((14, 15), (16, 17), (18, 19))


def register_lip_semseg(root):
    root = os.path.join(root, "lip")
    for name, (image_dir, gt_dir) in _PREDEFINED_SPLITS.items():
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=LIP_SEMSEG_CATEGORIES[:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            flip_map=LIP_FLIP_MAP
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_lip_semseg(_root)
