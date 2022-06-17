import os
import cv2
from typing import Dict, Sequence
import numpy as np
import itertools
from tabulate import tabulate
from termcolor import colored
import logging

def is_not_crowd(ann: Dict) -> bool:
    return ann.get("iscrowd", 0) == 0


def has_valid_bbox(anns: Sequence[Dict]) -> int:
    return sum(all(s > 1 for s in ann['bbox'][2:4]) for ann in anns) > 0


def has_valid_person(anns: Sequence[Dict]) -> bool:
    return sum(ann['category_id'] == 1 for ann in anns) > 0


def count_visible_keypoints(anns: Sequence[Dict]) -> int:
    return sum(sum(v > 0 for v in ann["keypoints"][2::3]) for ann in anns)


def has_visible_hier(anns: Sequence[Dict]) -> int:
    return sum(sum(v > 0 for v in ann["hier"][4::5]) for ann in anns) > 0

def get_parsing(root_dir, file_name, parsing_ids):
    human_dir = root_dir.replace('Images', 'Human_ids')
    category_dir = root_dir.replace('Images', 'Category_ids')
    file_name = file_name.replace('jpg', 'png')
    human_path = os.path.join(human_dir, file_name)
    category_path = os.path.join(category_dir, file_name)
    human_mask = cv2.imread(human_path, 0)
    category_mask = cv2.imread(category_path, 0)
    parsing = []
    for id in parsing_ids:
        parsing.append(category_mask * (human_mask == id))
    return parsing


def print_instances_class_histogram(dataset_dicts, class_names):
    """
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
    """
    _logger=logging.getLogger(__name__)
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,), dtype=np.int)
    for annos in dataset_dicts:
        classes = [x["category_id"] for x in annos if not x.get("iscrowd", 0)]
        histogram += np.histogram(classes, bins=hist_bins)[0]

    N_COLS = min(6, len(class_names) * 2)

    def short_name(x):
        # make long class names shorter. useful for lvis
        if len(x) > 13:
            return x[:11] + ".."
        return x

    data = list(
        itertools.chain(*[[short_name(class_names[i]), int(v)] for i, v in enumerate(histogram)])
    )
    total_num_instances = sum(data[1::2])
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    if num_classes > 1:
        data.extend(["total", total_num_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instances"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    _logger.info(
        "Distribution of instances among all {} categories:\n".format(
            num_classes) + colored(table, "green")
    )
