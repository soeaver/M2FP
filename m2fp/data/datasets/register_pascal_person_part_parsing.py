# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog

from .parsing import load_parsing


PASCAL_PERSON_PART_PARSING_CATEGORIES = [
    {"id": 0, "name": "Background"},
    {"id": 1, "name": "Lower-arm"},
    {"id": 2, "name": "Head"},
    {"id": 3, "name": "Upper-leg"},
    {"id": 4, "name": "Torso"},
    {"id": 5, "name": "Lower-leg"},
    {"id": 6, "name": "Upper-arm"},
    {"id": 7, "name": "Human"}
]

PASCAL_PERSON_PART_FLIP_MAP = ()

_PREDEFINED_SPLITS = {
    "pascal-person-part_parsing_train": (
        "Training/Images/",
        "Training/Category_ids/",
        "Training/Instance_ids/",
        "Training/Human_ids/",
    ),
    "pascal-person-part_parsing_test": (
        "Testing/Images/",
        "Testing/Category_ids/",
        "Testing/Instance_ids/",
        "Testing/Human_ids/",
    ),
}


def _get_pascal_person_part_parsing_meta():
    thing_ids = [k["id"] for k in PASCAL_PERSON_PART_PARSING_CATEGORIES]
    assert len(thing_ids) == 8, len(thing_ids)
    # Mapping from the incontiguous PASCAL_PERSON_PART category id to an id in [0, 99]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in PASCAL_PERSON_PART_PARSING_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "flip_map": PASCAL_PERSON_PART_FLIP_MAP,
        "num_parsing": 7,
        "semseg": {
            "semseg_format": "mask",
            "ignore_label": 255,
            'label_shift': 0,
            "name_trans": ('jpg', 'png'),
        },
    }
    return ret


def register_pascal_person_part_parsing(root):
    root = os.path.join(root, "pascal-person-part")
    meta = _get_pascal_person_part_parsing_meta()
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
register_pascal_person_part_parsing(_root)
