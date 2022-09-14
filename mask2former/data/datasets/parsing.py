# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import datetime
import io
import json
import logging
import numpy as np
import os
import shutil
import pycocotools.mask as mask_util
from fvcore.common.timer import Timer
from iopath.common.file_io import file_lock
from PIL import Image

from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from detectron2.utils.file_io import PathManager

logger = logging.getLogger(__name__)

__all__ = ["load_parsing"]


def load_parsing(category_gt_root, instance_gt_root, human_gt_root, image_root, gt_ext="png", image_ext="jpg"):

    # We match input images with ground truth based on their relative filepaths (without file
    # extensions) starting from 'image_root' and 'gt_root' respectively.
    def file2id(folder_path, file_path):
        # extract relative path starting from `folder_path`
        image_id = os.path.normpath(os.path.relpath(file_path, start=folder_path))
        # remove file extension
        image_id = os.path.splitext(image_id)[0]
        return image_id

    input_files = sorted(
        (os.path.join(image_root, f) for f in PathManager.ls(image_root) if f.endswith(image_ext)),
        key=lambda file_path: file2id(image_root, file_path),
    )

    dataset_dicts = []
    for img_path in input_files:
        image_name = img_path.split("/")[-1]

        category_gt_path = os.path.join(category_gt_root, image_name.replace(image_ext, gt_ext))
        human_gt_path = os.path.join(human_gt_root, image_name.replace(image_ext, gt_ext))
        instance_gt_path = os.path.join(instance_gt_root, image_name.replace(image_ext, gt_ext))

        do_load = os.path.exists(category_gt_path) and \
                  os.path.exists(human_gt_path) and \
                  os.path.exists(instance_gt_path)

        # load image into dataset only if all gt png exist
        if do_load:
            record = {}
            record["file_name"] = img_path
            record["category_file_name"] = category_gt_path
            record["human_file_name"] = human_gt_path
            record["instance_file_name"] = instance_gt_path
            dataset_dicts.append(record)

    logger.info(
        "Loaded {} images with parsing from {}".format(len(dataset_dicts), image_root)
    )

    return dataset_dicts
