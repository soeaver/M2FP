# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch.nn import functional as F


def single_human_sem_seg_postprocess(result, img_size, crop_box, output_height, output_width):
    result = result[:, : img_size[0], : img_size[1]]
    result = result[:, crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]].expand(1, -1, -1, -1)
    result = F.interpolate(
        result, size=(output_height, output_width), mode="bilinear", align_corners=False
    )[0]
    return result
