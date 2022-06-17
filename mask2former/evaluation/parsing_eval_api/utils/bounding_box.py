import math
import numpy as np
from enum import Enum, unique
from typing import Any, List, Set, Tuple, Union

import torch
from torch import BoolTensor, Tensor

from .base_struct import FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM, BaseStruct


_RawBoxType = Union[List[float], Tuple[float, ...], torch.Tensor, np.ndarray]


@unique
class BoxMode(Enum):
    """
    Enum of different ways to represent a box.
    """

    XYXY = 'xyxy'
    """
    (x0, y0, x1, y1) in absolute floating points coordinates.
    The coordinates in range [0, width or height].
    """

    XYWH = 'xywh'
    """
    (x0, y0, w, h) in absolute floating points coordinates.
    """

    XYWHA = 'xywha'
    """
    (xc, yc, w, h, a) in absolute floating points coordinates.
    (xc, yc) is the center of the rotated box, and the angle a is in degrees ccw.
    """

    @staticmethod
    def convert(box: _RawBoxType, from_mode: 'BoxMode', to_mode: 'BoxMode') -> _RawBoxType:
        """
        Args:
            box: can be a k-tuple, k-list or an Nxk array/tensor, where k = 4 or 5
            from_mode, to_mode (BoxMode)

        Returns:
            The converted box of the same type.
        """
        if from_mode == to_mode:
            return box

        original_type = type(box)
        is_numpy = isinstance(box, np.ndarray)
        single_box = isinstance(box, (list, tuple))
        if single_box:
            assert len(box) == 4 or len(box) == 5, (
                "BoxMode.convert takes either a k-tuple/list or an Nxk array/tensor,"
                " where k == 4 or 5"
            )
            arr = torch.tensor(box)[None, :]
        else:
            # avoid modifying the input box
            if is_numpy:
                arr = torch.from_numpy(np.asarray(box)).clone()
            else:
                arr = box.clone()

        if from_mode == BoxMode.XYWHA and to_mode == BoxMode.XYXY:
            assert (
                arr.shape[-1] == 5
            ), "The last dimension of input shape must be 5 for XYWHA format"
            original_dtype = arr.dtype
            arr = arr.double()

            w = arr[:, 2]
            h = arr[:, 3]
            a = arr[:, 4]
            c = torch.abs(torch.cos(a * math.pi / 180.0))
            s = torch.abs(torch.sin(a * math.pi / 180.0))
            # This basically computes the horizontal bounding rectangle of the rotated box
            new_w = c * w + s * h
            new_h = c * h + s * w

            # convert center to top-left corner
            arr[:, 0] -= new_w / 2.0
            arr[:, 1] -= new_h / 2.0
            # bottom-right corner
            arr[:, 2] = arr[:, 0] + new_w
            arr[:, 3] = arr[:, 1] + new_h

            arr = arr[:, :4].to(dtype=original_dtype)
        elif from_mode == BoxMode.XYWH and to_mode == BoxMode.XYWHA:
            original_dtype = arr.dtype
            arr = arr.double()
            arr[:, 0] += arr[:, 2] / 2.0
            arr[:, 1] += arr[:, 3] / 2.0
            angles = torch.zeros((arr.shape[0], 1), dtype=arr.dtype)
            arr = torch.cat((arr, angles), axis=1).to(dtype=original_dtype)
        else:
            if to_mode == BoxMode.XYXY and from_mode == BoxMode.XYWH:
                arr[:, 2] += arr[:, 0]
                arr[:, 3] += arr[:, 1]
            elif from_mode == BoxMode.XYXY and to_mode == BoxMode.XYWH:
                arr[:, 2] -= arr[:, 0]
                arr[:, 3] -= arr[:, 1]
            else:
                raise NotImplementedError(
                    f"Conversion from BoxMode {from_mode} to {to_mode} is not supported yet")

        if single_box:
            return original_type(arr.flatten().tolist())
        if is_numpy:
            return arr.numpy()
        else:
            return arr


class BoundingBox(BaseStruct):
    __ann_type__: str = 'bbox'
    __modes__: Set[BoxMode] = {BoxMode.XYXY, BoxMode.XYWH}

    def __init__(self, size: Tuple[int, int], boxes: Any, mode: BoxMode = BoxMode.XYXY) -> None:
        super(BoundingBox, self).__init__(size)

        device = boxes.device if isinstance(boxes, torch.Tensor) else 'cpu'
        boxes = torch.as_tensor(boxes, dtype=torch.float32, device=device)

        if len(boxes) == 0:
            boxes = boxes.view(0, 4)

        self._check_boxes_shape(boxes)
        self._check_boxes_mode(mode)

        self.tensor: Tensor = boxes
        self.mode = mode

    def _check_boxes_shape(self, boxes: Tensor) -> None:
        if not isinstance(boxes, torch.Tensor):
            raise TypeError(f"'boxes' should be torch.Tensor, got {type(boxes)}.")
        if boxes.ndim != 2:
            raise ValueError(f"'boxes' should have 2 dims, got {boxes.ndim}.")
        if boxes.size(-1) != 4:
            raise ValueError(f"Last dim of 'boxes' should have a size of 4, got {boxes.size(-1)}.")

    def _check_boxes_mode(self, mode: BoxMode) -> None:
        if mode not in self.__modes__:
            raise ValueError(f"mode should be ({', '.join(map(str, self.__modes__))}), got {mode}")

    def area(self) -> Tensor:
        boxes = self.tensor
        if self.mode == BoxMode.XYXY:
            aeras = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        elif self.mode == BoxMode.XYWH:
            aeras = boxes[:, 2] * boxes[:, 3]
        else:
            boxes = BoxMode.convert(boxes, self.mode, BoxMode.XYWH)
            aeras = boxes[:, 2] * boxes[:, 3]

        return aeras

    def convert(self, mode: BoxMode, deepcopy: bool = False) -> 'BoundingBox':
        self._check_boxes_mode(mode)

        if self.mode == mode:
            return self.clone() if deepcopy else self

        return type(self)(self.size, BoxMode.convert(self.tensor, self.mode, mode), mode=mode)

    @classmethod
    def cat(cls, structs) -> 'BoundingBox':
        modes = set(struct.mode for struct in structs)
        if len(modes) > 1:
            raise ValueError(f"multi modes: {modes}.")

        cated_boxes = torch.cat([struct.tensor for struct in structs], dim=0)
        return cls(structs[0].size, cated_boxes, mode=structs[0].mode)

    def nonempty(self, threshold: float = 0.0) -> BoolTensor:
        box = self.tensor
        widths = box[:, 2] - box[:, 0]
        heights = box[:, 3] - box[:, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def crop(self, box: Tuple[int, int, int, int], **kwargs) -> 'BoundingBox':
        left, top, right, bottom = box
        crop_width, crop_height = right - left, bottom - top

        x0, y0, x1, y1 = BoxMode.convert(self.tensor, self.mode, BoxMode.XYXY).unbind(dim=-1)
        cropped_xmin = (x0 - left).clamp(min=0, max=crop_width)
        cropped_xmax = (x1 - left).clamp(min=0, max=crop_width)
        cropped_ymin = (y0 - top).clamp(min=0, max=crop_height)
        cropped_ymax = (y1 - top).clamp(min=0, max=crop_height)

        cropped_boxes = torch.stack(
            (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1)
        cropped_boxes = BoxMode.convert(cropped_boxes, BoxMode.XYXY, self.mode)

        return type(self)((crop_width, crop_height), cropped_boxes, mode=self.mode)

    def resize(self, size: Tuple[int, int], **kwargs) -> 'BoundingBox':
        self._check_size(size)

        width, height = self.size
        ratio_width, ratio_height = size[0] / width, size[1] / height

        if ratio_width == ratio_height:
            resized_boxes = self.tensor * ratio_width
        else:  # ratio_width != ratio_height
            resized_boxes = self.tensor.clone()
            resized_boxes[:, 0] *= ratio_width
            resized_boxes[:, 1] *= ratio_height
            resized_boxes[:, 2] *= ratio_width
            resized_boxes[:, 3] *= ratio_height

        return type(self)(size, resized_boxes, mode=self.mode)

    def to(self, device: Union[str, torch.device]) -> 'BoundingBox':
        return type(self)(self.size, self.tensor.to(device), self.mode)

    def transpose(self, method: int, to_remove: int = 0, **kwargs) -> 'BoundingBox':
        width, height = self.size

        x0, y0, x1, y1 = BoxMode.convert(self.tensor, self.mode, BoxMode.XYXY).unbind(dim=-1)

        if method == FLIP_LEFT_RIGHT:
            transposed_x0 = (width - to_remove) - x1
            transposed_x1 = (width - to_remove) - x0
            transposed_y0 = y0
            transposed_y1 = y1
        elif method == FLIP_TOP_BOTTOM:
            transposed_x0 = x0
            transposed_x1 = x1
            transposed_y0 = (height - to_remove) - y1
            transposed_y1 = (height - to_remove) - y0
        else:
            raise NotImplementedError(
                f"Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented, got {method}.")

        transposed_boxes = (transposed_x0, transposed_y0, transposed_x1, transposed_y1)
        transposed_boxes = torch.stack(transposed_boxes, dim=-1)
        transposed_boxes = BoxMode.convert(transposed_boxes, BoxMode.XYXY, self.mode)

        return type(self)(self.size, transposed_boxes, mode=self.mode)

    def __len__(self) -> int:
        return len(self.tensor)

    def __getitem__(self, item) -> 'BoundingBox':
        return type(self)(self.size, self.tensor[item], self.mode)

    def extra_repr(self) -> str:
        return f"mode={self.mode}"


def boundingbox_iof(bbox1: BoundingBox, bbox2: BoundingBox) -> Tensor:
    if bbox1.size != bbox2.size:
        raise ValueError(f"Size mismatch: {bbox1.size}, {bbox2.size}.")

    bbox1 = bbox1.convert(BoxMode.XYXY)
    bbox2 = bbox2.convert(BoxMode.XYXY)

    area1 = bbox1.area()
    boxes1, boxes2 = bbox1.tensor, bbox2.tensor

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    iof = inter / area1[:, None]

    return iof


def boundingbox_iou(bbox1: BoundingBox, bbox2: BoundingBox) -> Tensor:
    """Return iou of bbox1 and bbox1, torch version.
    """
    if bbox1.size != bbox2.size:
        raise ValueError(f"Size mismatch: {bbox1.size}, {bbox2.size}.")

    bbox1 = bbox1.convert(BoxMode.XYXY)
    bbox2 = bbox2.convert(BoxMode.XYXY)

    area1, area2 = bbox1.area(), bbox2.area()
    boxes1, boxes2 = bbox1.tensor, bbox2.tensor

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, None] + area2 - inter)

    return iou


def boundingbox_overlap(bbox1: BoundingBox, bbox2: BoundingBox) -> BoolTensor:
    overlap = boundingbox_iou(bbox1, bbox2) > 0
    return overlap


def boundingbox_partly_overlap(bbox1: BoundingBox, bbox2: BoundingBox) -> BoolTensor:
    if bbox1.size != bbox2.size:
        raise ValueError(f"Size mismatch: {bbox1.size}, {bbox2.size}.")

    bbox1 = bbox1.convert(BoxMode.XYXY)
    bbox2 = bbox2.convert(BoxMode.XYXY)

    area1, area2 = bbox1.area(), bbox2.area()
    boxes1, boxes2 = bbox1.tensor, bbox2.tensor

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, None] + area2 - inter)

    overlap = iou > 0
    not_complete_overlap = (inter - area1[:, None]) * (inter - area2[None, :]) != 0
    partly_overlap = overlap & not_complete_overlap

    return partly_overlap
