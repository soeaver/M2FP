# Copyright (c) Facebook, Inc. and its affiliates.
import json
import logging
import numpy as np
import os
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_sem_seg

ATR_SEMSEG_CATEGORIES = [
    'Background',
    'Hat',
    'Hair',
    'Glass',
    'Up',
    'Skirt',
    'Pants',
    'Dress',
    'Belt',
    'L-Shoe',
    'R-Shoe',
    'Face',
    'L-Leg',
    'R-Leg',
    'L-Arm',
    'R-Arm',
    'Bag',
    'Scarf'
]

# ==== Predefined splits for raw ATR images ===========
_PREDEFINED_SPLITS = {
    "atr_semseg_train": ("Training/Images/", "Training/Category_ids/"),
    "atr_semseg_val": ("Validation/Images/", "Validation/Category_ids/"),
}

ATR_FLIP_MAP = ((9, 10), (12, 13), (14, 15))


def register_atr_semseg(root):
    root = os.path.join(root, "atr")
    for name, (image_dir, gt_dir) in _PREDEFINED_SPLITS.items():
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=ATR_SEMSEG_CATEGORIES[:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            flip_map=ATR_FLIP_MAP
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_atr_semseg(_root)
