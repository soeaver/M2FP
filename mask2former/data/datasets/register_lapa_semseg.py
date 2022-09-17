# Copyright (c) Facebook, Inc. and its affiliates.
import json
import logging
import numpy as np
import os
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_sem_seg

LAPA_SEMSEG_CATEGORIES = [
    'Background',
    'Skin',
    'Left-eyebrow',
    'Right-eyebrow',
    'Left-eye',
    'Right-eye',
    'Nose',
    'Upper-lip',
    'Inner-mouth',
    'Lower-lip',
    'Hair'
]

# ==== Predefined splits for raw LaPa images ===========
_PREDEFINED_SPLITS = {
    "lapa_semseg_train": ("Training/Images/", "Training/Category_ids/"),
    "lapa_semseg_val": ("Validation/Images/", "Validation/Category_ids/"),
}

LAPA_FLIP_MAP = ((2, 3), (4, 5))


def register_lapa_semseg(root):
    root = os.path.join(root, "lapa")
    for name, (image_dir, gt_dir) in _PREDEFINED_SPLITS.items():
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=LAPA_SEMSEG_CATEGORIES[:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            flip_map=LAPA_FLIP_MAP
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_lapa_semseg(_root)
