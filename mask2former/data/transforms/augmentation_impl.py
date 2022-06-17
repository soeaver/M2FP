# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Implement many useful :class:`Augmentation`.
"""
import numpy as np
import sys
from typing import Tuple
import torch
from fvcore.transforms.transform import (
    BlendTransform,
    CropTransform,
    HFlipTransform,
    NoOpTransform,
    PadTransform,
    Transform,
    TransformList,
    VFlipTransform,
)
from PIL import Image
import cv2, random
from detectron2.data import transforms as T
from .transform import RotationTransform

__all__ = [
    "ResizeByAspectRatio",
    "ResizeByScale",
    "RandomCenterRotation",
]



class ResizeByAspectRatio(T.Augmentation):
    """Resize image to a fixed aspect ratio"""

    def __init__(self, aspect_ratio, interp=Image.LINEAR):
        """
        Args:
            aspect_ratio: float, w/h
            interp: PIL interpolation method
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image: np.ndarray) -> Transform:
        h, w = image.shape[0], image.shape[1]

        if w > self.aspect_ratio * h:
            h = np.round(w * 1.0 / self.aspect_ratio).astype(int)
        elif w < self.aspect_ratio * h:
            w = np.round(h * self.aspect_ratio).astype(int)

        return T.ResizeTransform(image.shape[0], image.shape[1], h, w, self.interp)


class ResizeByScale(T.Augmentation):
    """Resize image to a fixed aspect ratio"""

    def __init__(self, scale_factor, interp=Image.LINEAR):
        """
        Args:
            aspect_ratio: float, w/h
            interp: PIL interpolation method
        """
        super().__init__()
        self._init(locals())

    def _get_resize(self, image: np.ndarray, scale: float) -> Transform:
        h, w = image.shape[0], image.shape[1]

        h = np.round(h * scale).astype(int)
        w = np.round(w * scale).astype(int) 

        return T.ResizeTransform(image.shape[0], image.shape[1], h, w, self.interp)

    def get_transform(self, image: np.ndarray) -> Transform:
        random_scale = np.clip(np.random.randn() * self.scale_factor + 1, 1 - self.scale_factor, 1 + self.scale_factor) 
        return self._get_resize(image, random_scale)
    

class RandomCenterRotation(T.Augmentation):
    """
    This method returns a copy of this image, rotated the given
    number of degrees counter clockwise around the given center.
    """

    def __init__(self, angle_range, expand=True, interp=cv2.INTER_LINEAR):
        """
        Args:
            angle (float): ratation factor
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (list[[float, float]]):  If ``sample_style=="range"``,
                a [[minx, miny], [maxx, maxy]] relative interval from which to sample the center,
                [0, 0] being the top left of the image and [1, 1] the bottom right.
                If ``sample_style=="choice"``, a list of centers to sample from
                Default: None, which means that the center of rotation is the center of the image
                center has no effect if expand=True because it only affects shifting
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        angle = np.clip(
            np.random.randn() * self.angle_range, -self.angle_range * 2, self.angle_range * 2
        ) if random.random() <= 0.6 else 0
        return RotationTransform(h, w, angle, expand=self.expand, center=None, interp=self.interp)
