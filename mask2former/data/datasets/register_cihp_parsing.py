# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json


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
        "cihp/annotations/CIHP_parsing_train.json",
        "cihp/Training/Images/",
        "cihp/Training/Category_ids/",
        "cihp/Training/Instance_ids/",
        "cihp/Training/Human_ids/",
    ),
    "cihp_parsing_val": (
        "cihp/annotations/CIHP_parsing_val.json",
        "cihp/Validation/Images/",
        "cihp/Validation/Category_ids/",
        "cihp/Validation/Instance_ids/",
        "cihp/Validation/Human_ids/",
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
    meta = _get_cihp_parsing_meta()
    extra_keys = ["parsing_id", "area", "ispart", "isfg"]
    for name, (json_file, image_root, semantic_gt_root, part_gt_root, human_gt_root) in _PREDEFINED_SPLITS.items():
        json_file = os.path.join(root, json_file)
        image_root = os.path.join(root, image_root)
        semantic_gt_root = os.path.join(root, semantic_gt_root)
        part_gt_root = os.path.join(root, part_gt_root)
        human_gt_root = os.path.join(root, human_gt_root)

        DatasetCatalog.register(
            name,
            lambda json_file=json_file, image_root=image_root, name=name, extra_keys=extra_keys: load_coco_json(
                json_file, image_root,
                dataset_name=name,
                extra_annotation_keys=extra_keys
            )
        )
        MetadataCatalog.get(name).set(
            json_file=json_file,
            image_root=image_root,
            semantic_gt_root=semantic_gt_root,
            part_gt_root=part_gt_root,
            human_gt_root=human_gt_root,
            evaluator_type="parsing",
            **meta
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_cihp_parsing(_root)
