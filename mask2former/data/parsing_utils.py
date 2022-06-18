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
    "filter_instance_by_attributes",
    "flip_human_semantic_category",
    "transform_parsing_instance_annotations",
    "affine_to_target_size",
    "center_to_target_size_semantic",
    "center_to_target_size_parsing",
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


def filter_instance_by_attributes(dataset_dicts, with_human_ins, with_bkg_ins):
    annos = dataset_dicts.pop("annotations")
    """
             ispart  isfg
    part  :    1      1
    human :    0      1
    bkg   :    0      0
    """

    new_annos = []
    if not with_bkg_ins and not with_human_ins:  # discard bkg and human instances
        for obj in annos:
            if obj["ispart"] == 1 and obj["isfg"] == 1:
                obj['category_id'] -= 1
                new_annos.append(obj)
    elif with_bkg_ins and not with_human_ins:  # discard human instances
        for obj in annos:
            if obj["ispart"] == obj["isfg"]:
                new_annos.append(obj)
    elif not with_bkg_ins and with_human_ins:  # discard bkg instances
        for obj in annos:
            if obj["isfg"] == 1:
                if obj["ispart"] == 0:
                    obj['category_id'] = 0
                new_annos.append(obj)
    else: # keep all part, bkg and human instances
        raise NotImplementedError("Parsing with part, human and bkg instances not implemented yet !!!")

    dataset_dicts["annotations"] = new_annos
    return dataset_dicts


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


def transform_parsing_instance_annotations(annotation, transforms, image_size, flip_map):

    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # clip transformed bbox to image size
    bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
    annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if "segmentation" in annotation:
        segm = annotation["segmentation"]

        if isinstance(segm, list):
            # polygons
            polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
            annotation["segmentation"] = [
                p.reshape(-1) for p in transforms.apply_polygons(polygons)
            ]

            # change part label if do h_flip
            annotation["category_id"] = flip_human_instance_category(
                annotation["category_id"], transforms, flip_map
            )
        elif isinstance(segm, dict):
            # RLE
            mask = mask_util.decode(segm)
            mask = transforms.apply_segmentation(mask)
            assert tuple(mask.shape[:2]) == image_size
            annotation["segmentation"] = mask

            # change part label if do h_flip
            annotation["category_id"] = flip_human_instance_category(
                annotation["category_id"], transforms, flip_map
            )
        else:
            raise ValueError(
                "Cannot transform segmentation of type '{}'!"
                "Supported types are: polygons as list[list[float] or ndarray],"
                " COCO-style RLE as a dict.".format(type(segm))
            )

    return annotation


def flip_human_instance_category(category, transforms, flip_map):
    do_hflip = sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1  # bool

    if do_hflip:
        for ori_label, new_label in flip_map:
            if category == ori_label:
                category = new_label
            elif category == new_label:
                category = ori_label
    return category


def affine_to_target_size(img, gt, target_size):
    assert img.shape[:2] == gt.shape
    org_h, org_w = img.shape[:2]
    bbox = np.asarray((0, 0, org_w, org_h))
    x0, y0, w, h = bbox
    xc = x0 + w * 0.5
    yc = y0 + h * 0.5

    aspect_ratio = target_size[0] / target_size[1]
    w, h = change_aspect_ratio(w, h, aspect_ratio)

    bbox = torch.tensor([xc, yc, w, h, 0.])
    trans = get_affine_transform(bbox, target_size)

    new_img = cv2.warpAffine(
        img, trans, (int(target_size[0]), int(target_size[1])), flags=cv2.INTER_LINEAR, borderValue=(128, 128, 128)
    )
    new_gt  = cv2.warpAffine(
        gt, trans, (int(target_size[0]), int(target_size[1])), flags=cv2.INTER_NEAREST, borderValue=(255, 255, 255)
    )
    return new_img, new_gt


def get_affine_transform(box, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    center = np.array([box[0], box[1]], dtype=np.float32)
    scale = np.array([box[2], box[3]], dtype=np.float32)
    rot = box[4]

    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def change_aspect_ratio(w, h, aspect_ratio):
    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    return w, h


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


def center_to_target_size_parsing(img, annos, target_size):
    """
    Use detectron2.data.transforms.ExtentTransform and fvcore.transforms.transform.CropTransform to
    adapt the image and instance annos(bbox and polygon) to target output size.
    """
    src_h, src_w = img.shape[0], img.shape[1]
    trg_h, trg_w = target_size[1], target_size[0]

    tfm_list = []
    if src_h > trg_h and src_w > trg_w:
        crop_w = int((src_w - trg_w) / 2)
        crop_h = int((src_h - trg_h) / 2)
        tfm_list.append(CropTransform(crop_w, crop_h, trg_w, trg_h))

    elif src_h > trg_h and src_w <= trg_w:
        crop_h = int((src_h - trg_h) / 2)
        tfm_list.append(CropTransform(0, crop_h, src_w, trg_h))
        tfm_list.append(PadTransform(trg_h, src_w, trg_h, trg_w))

    elif src_h <= trg_h and src_w > trg_w:
        crop_w = int((src_w - trg_w) / 2)
        tfm_list.append(CropTransform(crop_w, 0, trg_w, src_h))
        tfm_list.append(PadTransform(src_h, trg_w, trg_h, trg_w))

    else:
        tfm_list.append(PadTransform(src_h, src_w, trg_h, trg_w))

    new_img = copy.deepcopy(img)
    for tfm in tfm_list:
        new_img = tfm.apply_image(new_img)

    for anno in annos:
        if "segmentation" in anno:
            segm = anno["segmentation"]
            if isinstance(segm, list):
                # polygons
                polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
                for tfm in tfm_list:
                    polygons = tfm.apply_polygons(polygons)
                anno['segmentation'] = [p.reshape(-1) for p in polygons]
            elif isinstance(segm, dict):
                # RLE
                mask = mask_util.decode(segm)
                for tfm in tfm_list:
                    mask = tfm.apply_segmentation(mask)
                assert tuple(mask.shape[:2]) == image_size
                anno["segmentation"] = mask
            else:
                raise ValueError(
                    "Cannot transform segmentation of type '{}'!"
                    "Supported types are: polygons as list[list[float] or ndarray],"
                    " COCO-style RLE as a dict.".format(type(segm))
                )

    return new_img, annos

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
