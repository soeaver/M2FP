# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog

from .parsing import load_parsing


CIHP_PARSING_CATEGORIES = [
    {"id": 0, "name": "Background"},
    {"id": 1, "name": "Hat"},
    {"id": 2, "name": "Hair"},
    {"id": 3, "name": "Gloves"},
    {"id": 4, "name": "Sunglasses"},
    {"id": 5, "name": "UpperClothes"},
    {"id": 6, "name": "Dress"},
    {"id": 7, "name": "Coat"},
    {"id": 8, "name": "Socks"},
    {"id": 9, "name": "Pants"},
    {"id": 10, "name": "Torso-skin"},
    {"id": 11, "name": "Scarf"},
    {"id": 12, "name": "Skirt"},
    {"id": 13, "name": "Face"},
    {"id": 14, "name": "Left-arm"},
    {"id": 15, "name": "Right-arm"},
    {"id": 16, "name": "Left-leg"},
    {"id": 17, "name": "Right-leg"},
    {"id": 18, "name": "Left-shoe"},
    {"id": 19, "name": "Right-shoe"},
    {"id": 20, "name": "Human"}
]


CIHP_FLIP_MAP = ((14, 15), (16, 17), (18, 19))


_PREDEFINED_SPLITS = {
    "cihp_parsing_train": (
        "Training/Images/",
        "Training/Category_ids/",
        "Training/Instance_ids/",
        "Training/Human_ids/",
    ),
    "cihp_parsing_val": (
        "Validation/Images/",
        "Validation/Category_ids/",
        "Validation/Instance_ids/",
        "Validation/Human_ids/",
    ),
}


def _get_cihp_parsing_meta():
    thing_ids = [k["id"] for k in CIHP_PARSING_CATEGORIES]
    assert len(thing_ids) == 21, len(thing_ids)
    # Mapping from the incontiguous CIHP category id to an id in [0, 99]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in CIHP_PARSING_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "flip_map": CIHP_FLIP_MAP,
        "num_parsing": 20,
        "semseg": {
            "semseg_format": "mask",
            "ignore_label": 255,
            'label_shift': 0,
            "name_trans": ('jpg', 'png'),
        },
    }
    return ret


def register_cihp_parsing(root):
    root = os.path.join(root, "cihp")
    meta = _get_cihp_parsing_meta()
    for name, (image_root, category_gt_root, instance_gt_root, human_gt_root) in _PREDEFINED_SPLITS.items():
        image_root = os.path.join(root, image_root)
        category_gt_root = os.path.join(root, category_gt_root)
        instance_gt_root = os.path.join(root, instance_gt_root)
        human_gt_root = os.path.join(root, human_gt_root)

        DatasetCatalog.register(
            name, lambda k=image_root, l=category_gt_root, m=instance_gt_root, n=human_gt_root: load_parsing(
                l, m, n, k, gt_ext="png", image_ext="jpg"
            )
        )
        MetadataCatalog.get(name).set(
            image_root=image_root,
            category_gt_root=category_gt_root,
            instance_gt_root=instance_gt_root,
            human_gt_root=human_gt_root,
            evaluator_type="parsing",
            **meta
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_cihp_parsing(_root)
