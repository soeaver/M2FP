import random
import cv2
import copy
import logging
import numpy as np
from typing import List, Union
import pycocotools.mask as mask_util
import torch
from PIL import Image

from fvcore.transforms.transform import CropTransform
from detectron2.structures import BoxMode
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog
from detectron2.utils.file_io import PathManager
from detectron2.data.detection_utils import _apply_exif_orientation

from .transforms.transform import PadTransform

__all__ = [
    "read_semseg_gt",
    "gen_parsing_instances",
    "flip_human_semantic_category",
    "center_to_target_size_semantic",
    "center_to_target_size_test",
]


def read_semseg_gt(file_name):
    with PathManager.open(file_name, "rb") as f:
        gt_pil = Image.open(f)
        # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
        gt_pil = _apply_exif_orientation(gt_pil)

        gt_array = np.asarray(gt_pil)
        if len(gt_array.shape) == 3:
            assert gt_array.shape[2] == 3
            gt_array = gt_array.transpose(2, 0, 1)[0, :, :]

        return gt_array


def gen_parsing_instances(human_png, category_png, with_human_instance, num_parsing):
    classes = []
    masks = []

    # generate parsing png for each human
    parsing_ids = np.unique(human_png)
    bg_id_index = np.where(parsing_ids == 0)[0]
    parsing_ids = list(np.delete(parsing_ids, bg_id_index))
    humans = np.stack([human_png == parsing_id for parsing_id in parsing_ids])
    humans_categories = humans * category_png

    # add bkg instance
    if np.max(np.where(human_png == 0, 1, 0) - np.where(category_png == 0, 1, 0)) == 0:
        bkg_mask = np.where(human_png == 0, 1, 0).copy()
    else:
        bkg_mask = np.where(category_png == 0, 1, 0).copy()
    bkg_class = 0
    classes.append(bkg_class)
    masks.append(bkg_mask)

    for ind, human_id in enumerate(parsing_ids):
        # add human instance
        if with_human_instance:
            human_mask = np.where(human_png == human_id, 1, 0).copy()
            assert np.max(human_mask) == 1, "human {} is missed".format(human_id + 1)
            human_class = num_parsing
            classes.append(human_class)
            masks.append(human_mask)

        # add part instances
        human_categories = humans_categories[ind, :, :]
        part_ids = np.unique(human_categories)
        _bg_id_index = np.where(part_ids == 0)[0]
        part_ids = list(np.delete(part_ids, _bg_id_index))

        part_masks = [(human_categories == part_id).astype(np.uint8) for part_id in part_ids]
        part_classes = part_ids

        classes.extend(part_classes)
        masks.extend(part_masks)

    return classes, masks


def flip_human_semantic_category(img, gt, flip_map, prob):
    do_hflip = random.random() < prob
    if do_hflip:
        img = np.flip(img, axis=1)
        gt = gt[:, ::-1]
        gt = np.ascontiguousarray(gt)
        for ori_label, new_label in flip_map:
            left = gt == ori_label
            right = gt == new_label
            gt[left] = new_label
            gt[right] = ori_label
    return img, gt


def center_to_target_size_semantic(img, gt, target_size):
    assert img.shape[:2] == gt.shape
    tfmd_h, tfmd_w = img.shape[0], img.shape[1]

    new_image = np.ones((target_size[1], target_size[0], 3), dtype=img.dtype) * 128
    new_gt = np.ones((target_size[1], target_size[0]), dtype=gt.dtype) * 255

    if tfmd_h > target_size[1] and tfmd_w > target_size[0]:
        range_ori_h = (int((tfmd_h - target_size[1]) / 2), int((tfmd_h + target_size[1]) / 2))
        range_ori_w = (int((tfmd_w - target_size[0]) / 2), int((tfmd_w + target_size[0]) / 2))

        new_image = img[range_ori_h[0]:range_ori_h[1], range_ori_w[0]:range_ori_w[1], :]
        new_gt = gt[range_ori_h[0]:range_ori_h[1], range_ori_w[0]:range_ori_w[1]]

    elif tfmd_h > target_size[1] and tfmd_w <= target_size[0]:
        range_ori_h = (int((tfmd_h - target_size[1]) / 2), int((tfmd_h + target_size[1]) / 2))
        range_new_w = (int((target_size[0] - tfmd_w) / 2), int((tfmd_w + target_size[0]) / 2))

        new_image[:, range_new_w[0]:range_new_w[1], :] = img[range_ori_h[0]:range_ori_h[1], :, :]
        new_gt[:, range_new_w[0]:range_new_w[1]] = gt[range_ori_h[0]:range_ori_h[1], :]

    elif tfmd_h <= target_size[1] and tfmd_w > target_size[0]:
        range_ori_w = (int((tfmd_w - target_size[0]) / 2), int((tfmd_w + target_size[0]) / 2))
        range_new_h = (int((target_size[1] - tfmd_h) / 2), int((tfmd_h + target_size[1]) / 2))

        new_image[range_new_h[0]:range_new_h[1], :, :] = img[:, range_ori_w[0]:range_ori_w[1], :]
        new_gt[range_new_h[0]:range_new_h[1], :] = gt[:, range_ori_w[0]:range_ori_w[1]]

    else:
        range_new_h = (int((target_size[1] - tfmd_h) / 2), int((tfmd_h + target_size[1]) / 2))
        range_new_w = (int((target_size[0] - tfmd_w) / 2), int((tfmd_w + target_size[0]) / 2))

        new_image[range_new_h[0]:range_new_h[1], range_new_w[0]:range_new_w[1], :] = img
        new_gt[range_new_h[0]:range_new_h[1], range_new_w[0]:range_new_w[1]] = gt

    return new_image, new_gt


def center_to_target_size_test(img, target_size):
    src_h, src_w = img.shape[0], img.shape[1]
    trg_h, trg_w = target_size[1], target_size[0]

    new_h, new_w = 0, 0
    tfm_list = []
    if src_h > trg_h and src_w > trg_w:
        if src_h > src_w:
            new_h = trg_h
            new_w = int(new_h * src_w / src_h)
            if new_w > trg_w:
                new_w = trg_w
                new_h = int(new_w * src_h / src_w)
        elif src_w > src_h:
            new_w = trg_w
            new_h = int(new_w * src_h / src_w)
            if new_h > trg_h:
                new_h = trg_h
                new_w = int(new_h * src_w / src_h)
        tfm_list.append(T.ResizeTransform(src_h, src_w, new_h, new_w))
        tfm_list.append(PadTransform(new_h, new_w, trg_h, trg_w))

    elif src_h > trg_h and src_w <= trg_w:
        new_h = trg_h
        new_w = int(new_h * src_w / src_h)
        tfm_list.append(T.ResizeTransform(src_h, src_w, new_h, new_w))
        tfm_list.append(PadTransform(new_h, new_w, trg_h, trg_w))

    elif src_h <= trg_h and src_w > trg_w:
        new_w = trg_w
        new_h = int(new_w * src_h / src_w)
        tfm_list.append(T.ResizeTransform(src_h, src_w, new_h, new_w))
        tfm_list.append(PadTransform(new_h, new_w, trg_h, trg_w))

    else:
        new_h, new_w = src_h, src_w
        tfm_list.append(PadTransform(new_h, new_w, trg_h, trg_w))

    box = get_box(new_h, new_w, trg_h, trg_w)

    new_img = copy.deepcopy(img)
    for tfm in tfm_list:
        new_img = tfm.apply_image(new_img)

    return new_img, box


def get_box(src_h, src_w, trg_h, trg_w):
    assert src_h <= trg_h, "expect src_h <= trg_h"
    assert src_w <= trg_w, "expect src_w <= trg_w"

    x0 = int((trg_w - src_w) / 2)
    x1 = src_w + x0
    y0 = int((trg_h - src_h) / 2)
    y1 = src_h + y0

    box = [x0, y0, x1, y1]
    return box
