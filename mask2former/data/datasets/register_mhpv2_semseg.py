# Copyright (c) Facebook, Inc. and its affiliates.
import json
import logging
import numpy as np
import os
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_sem_seg

MHPv2_SEMSEG_CATEGORIES = [
    'Background',
    'Cap/Hat',
    'Helmet',
    'Face',
    'Hair',
    'Left-arm',
    'Right-arm',
    'Left-hand',
    'Right-hand',
    'Protector',
    'Bikini/bra',
    'Jacket/Windbreaker/Hoodie',
    'T-shirt',
    'Polo-shirt',
    'Sweater',
    'Singlet',
    'Torso-skin',
    'Pants',
    'Shorts/Swim-shorts',
    'Skirt', 'Stockings',
    'Socks', 'Left-boot',
    'Right-boot',
    'Left-shoe',
    'Right-shoe',
    'Left-highheel',
    'Right-highheel',
    'Left-sandal',
    'Right-sandal',
    'Left-leg',
    'Right-leg',
    'Left-foot',
    'Right-foot',
    'Coat',
    'Dress',
    'Robe',
    'Jumpsuits',
    'Other-full-body-clothes',
    'Headwear',
    'Backpack',
    'Ball',
    'Bats',
    'Belt',
    'Bottle',
    'Carrybag',
    'Cases',
    'Sunglasses',
    'Eyewear',
    'Gloves',
    'Scarf',
    'Umbrella',
    'Wallet/Purse',
    'Watch',
    'Wristband',
    'Tie',
    'Other-accessaries',
    'Other-upper-body-clothes',
    'Other-lower-body-clothes'
]

_PREDEFINED_SPLITS = {
    "mhpv2_semseg_train": ("Training/Images/", "Training/Category_ids/"),
    "mhpv2_semseg_val": ("Validation/Images/", "Validation/Category_ids/"),
}

MHPv2_FLIP_MAP = ((5, 6), (7, 8), (22, 23), (24, 25), (26, 27), (28, 29), (30, 31), (32, 33))


def register_mhpv2_semseg(root):
    root = os.path.join(root, "mhpv2")
    for name, (image_dir, gt_dir) in _PREDEFINED_SPLITS.items():
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=MHPv2_SEMSEG_CATEGORIES[:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            flip_map=MHPv2_FLIP_MAP
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_mhpv2_semseg(_root)
