# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json


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
    "pascal_person_part_parsing_train": (
        "annotations/PASCAL-Person-Part_parsing_train.json",
        "Training/Images/",
        "Training/Category_ids/",
        "Training/Instance_ids/",
        "Training/Human_ids/",
    ),
    "pascal_person_part_parsing_test": (
        "annotations/PASCAL-Person-Part_parsing_test.json",
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
register_pascal_person_part_parsing(_root)
