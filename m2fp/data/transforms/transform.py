# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

"""
See "Data Augmentation" tutorial for an overview of the system:
https://detectron2.readthedocs.io/tutorials/augmentation.html
"""

import numpy as np
import torch
import torch.nn.functional as F
from fvcore.transforms.transform import (
    CropTransform,
    HFlipTransform,
    NoOpTransform,
    Transform,
    TransformList,
)
from PIL import Image

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass

__all__ = [
    "RotationTransform",
    "PadTransform",
]


class RotationTransform(Transform):
    """
    This method returns a copy of this image, rotated the given
    number of degrees counter clockwise around its center.
    """

    def __init__(self, h, w, angle, expand=True, center=None, interp=None):
        """
        Args:
            h, w (int): original image size
            angle (float): degrees for rotation
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (tuple (width, height)): coordinates of the rotation center
                if left to None, the center will be fit to the center of each image
                center has no effect if expand=True because it only affects shifting
            interp: cv2 interpolation method, default cv2.INTER_LINEAR
        """
        super().__init__()
        image_center = np.array((w / 2, h / 2))
        if center is None:
            center = image_center
        if interp is None:
            interp = cv2.INTER_LINEAR
        abs_cos, abs_sin = (abs(np.cos(np.deg2rad(angle))), abs(np.sin(np.deg2rad(angle))))
        if expand:
            # find the new width and height bounds
            bound_w, bound_h = np.rint(
                [h * abs_sin + w * abs_cos, h * abs_cos + w * abs_sin]
            ).astype(int)
        else:
            bound_w, bound_h = w, h

        self._set_attributes(locals())
        self.rm_coords = self.create_rotation_matrix()
        # Needed because of this problem https://github.com/opencv/opencv/issues/11784
        self.rm_image = self.create_rotation_matrix(offset=-0.5)

    def apply_image(self, img, interp=None, border_value=(128, 128, 128)):
        """
        img should be a numpy array, formatted as Height * Width * Nchannels
        """
        if len(img) == 0 or self.angle % 360 == 0:
            return img
        assert img.shape[:2] == (self.h, self.w)
        interp = interp if interp is not None else self.interp
        return cv2.warpAffine(img, self.rm_image, (self.bound_w, self.bound_h), flags=interp, borderValue=border_value)

    def apply_coords(self, coords):
        """
        coords should be a N * 2 array-like, containing N couples of (x, y) points
        """
        coords = np.asarray(coords, dtype=float)
        if len(coords) == 0 or self.angle % 360 == 0:
            return coords
        return cv2.transform(coords[:, np.newaxis, :], self.rm_coords)[:, 0, :]

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=cv2.INTER_NEAREST, border_value=(255, 255, 255))
        return segmentation

    def create_rotation_matrix(self, offset=0):
        center = (self.center[0] + offset, self.center[1] + offset)
        rm = cv2.getRotationMatrix2D(tuple(center), self.angle, 1)
        if self.expand:
            # Find the coordinates of the center of rotation in the new image
            # The only point for which we know the future coordinates is the center of the image
            rot_im_center = cv2.transform(self.image_center[None, None, :] + offset, rm)[0, 0, :]
            new_center = np.array([self.bound_w / 2, self.bound_h / 2]) + offset - rot_im_center
            # shift the rotation center to the new coordinates
            rm[:, 2] += new_center
        return rm

    def inverse(self):
        """
        The inverse is to rotate it back with expand, and crop to get the original shape.
        """
        if not self.expand:  # Not possible to inverse if a part of the image is lost
            raise NotImplementedError()
        rotation = RotationTransform(
            self.bound_h, self.bound_w, -self.angle, True, None, self.interp
        )
        crop = CropTransform(
            (rotation.bound_w - self.w) // 2, (rotation.bound_h - self.h) // 2, self.w, self.h
        )
        return TransformList([rotation, crop])


class PadTransform(Transform):
    def __init__(self, src_h, src_w, trg_h, trg_w):
        super().__init__()
        assert src_h <= trg_h, "expect src_h <= trg_h"
        assert src_w <= trg_w, "expect src_w <= trg_w"

        pad_left = int((trg_w - src_w) / 2)
        pad_right = trg_w - src_w - pad_left
        pad_top = int((trg_h - src_h) / 2)
        pad_bottom = trg_h - src_h - pad_top

        self._set_attributes(locals())

    def apply_image(self, img, pad_value=128):
        if self.pad_left == 0 and self.pad_top == 0:
            return img

        if len(img.shape) == 2:
            return np.pad(
                img,
                ((self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right)),
                "constant",
                constant_values=((pad_value, pad_value), (pad_value, pad_value))
            )
        elif len(img.shape) == 3:
            return np.pad(
                img,
                ((self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right), (0, 0)),
                "constant",
                constant_values=((pad_value, pad_value), (pad_value, pad_value), (pad_value, pad_value))
            )

    def apply_coords(self, coords):
        coords = np.asarray(coords, dtype=float)
        if len(coords) == 0:
            return coords
        if self.pad_left == 0 and self.pad_top == 0:
            return coords

        coords[:, 0] += self.pad_left
        coords[:, 1] += self.pad_top
        return coords

    def apply_segmentation(self, segmentation, pad_value=255):
        segmentation = self.apply_image(segmentation, pad_value=pad_value)
        return segmentation
