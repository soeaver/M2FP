import copy
import warnings
from abc import ABCMeta, abstractmethod
from enum import Enum, unique
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np
import pycocotools.mask as mask_utils

import torch
import torch.nn.functional as F
from torch import Tensor


from .base_struct import FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM, BaseStruct


@unique
class MaskMode(Enum):
    """
    Enum of different ways to represent a mask.
    """

    POLY = 'poly'
    """
    List[np.ndarray] or List[List[np.ndarray]]
    """

    MASK = 'mask'
    """
    torch.Tensor
    shape: (h, w) or (n, h, w)
    dtype: torch.uint8
    """

    @staticmethod
    def convert(instance, from_mode: 'MaskMode', to_mode: 'MaskMode', size: Optional[Tuple[int, int]] = None):
        if from_mode == to_mode:
            return instance

        if from_mode == MaskMode.MASK and to_mode == MaskMode.POLY:
            assert isinstance(instance, (torch.Tensor, np.ndarray)), (
                f"MaskMode.MASK only support torch.Tensor or np.ndarray, got {type(instance)}."
            )
            assert instance.ndim in (2, 3), (
                f"masks should have 3 dims, got {instance.ndim}."
            )
            single_instance = instance.ndim == 2
            if isinstance(instance, torch.Tensor):
                instance = instance.cpu().numpy()
            if single_instance:
                instance = instance[None, ...]

            instance = instance.astype('uint8', copy=False)

            contours = []
            for mask in instance:
                contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

                reshaped_contour = []
                for entity in contour:
                    assert len(entity.shape) == 3
                    assert entity.shape[1] == 1, "Hierarchical contours are not allowed"
                    reshaped_contour.append(entity.reshape(-1))
                contours.append(reshaped_contour)

            if single_instance:
                contours = contours[0]

            return contours
        elif from_mode == MaskMode.POLY and to_mode == MaskMode.MASK:
            type_error = "MaskMode.POLY only support List[np.ndarray] or List[List[np.ndarray]]."
            assert isinstance(instance, (list, tuple)), type_error

            width, height = size

            if len(instance) == 0:
                return torch.empty((height, width), dtype=torch.uint8)

            single_instance = not isinstance(instance[0], (list, tuple))
            if single_instance:
                instance = [instance]

            masks = []
            for polygon in instance:
                assert isinstance(polygon[0], np.ndarray), type_error

                rles = mask_utils.frPyObjects(polygon, height, width)
                rle = mask_utils.merge(rles)
                mask = mask_utils.decode(rle)
                mask = torch.from_numpy(mask)
                masks.append(mask)

            if single_instance:
                masks = masks[0]
            else:
                masks = torch.stack(masks, dim=0)

            return masks
        else:
            raise NotImplementedError(
                f"Conversion from MaskMode {from_mode} to {to_mode} is not supported yet.")


class Mask(BaseStruct, metaclass=ABCMeta):
    __ann_type__: str = 'mask'
    __modes__: Set[MaskMode] = {MaskMode.MASK, MaskMode.POLY}

    def __init__(self, size: Tuple[int, int], instances: Any, mode: MaskMode):
        super(Mask, self).__init__(size)

        self._check_mask_mode(mode)

        self.instances = instances
        self.mode = mode

    def _check_mask_mode(self, mode: MaskMode) -> None:
        if mode not in self.__modes__:
            raise ValueError(f"mode should be ({', '.join(map(str, self.__modes__))}), got {mode}")

    def _check_boxes_shape(self, boxes: Tensor) -> None:
        if not isinstance(boxes, torch.Tensor):
            raise TypeError(f"boxes should be torch.Tensor, got {type(boxes)}.")
        if boxes.ndim != 2:
            raise ValueError(f"boxes should have 2 dims, got {boxes.ndim}.")
        if boxes.size(-1) != 4:
            raise ValueError(f"last dim of boxes should have a size of 4, got {boxes.size(-1)}.")

    def get_mask_tensor(self, **kwargs):
        return self.convert(MaskMode.MASK, **kwargs).instances

    def get_visable_half_body(self, upper_body_ids) -> Tuple[Tensor, Tensor]:
        if len(self) == 1:  # for instance, return shape (num_points, 2)
            return torch.empty(0, 2), torch.empty(0, 2)
        else:
            raise NotImplementedError

    def convert(self, mode: MaskMode, deepcopy=False, **kwargs) -> 'Mask':
        self._check_mask_mode(mode)

        if mode == self.mode:
            return self.clone() if deepcopy else self

        convert_func = f'_convert_to_{mode.value}'

        if not hasattr(self, convert_func):
            raise NotImplementedError(f'{convert_func}')

        return getattr(self, convert_func)(**kwargs)

    @classmethod
    def flip_prob(cls, prob: Tensor, shift: bool = False) -> Tensor:
        flipped_prob = prob.flip(3)
        if shift:
            flipped_prob[:, :, :, 1:] = flipped_prob[:, :, :, :-1]
        return flipped_prob

    @abstractmethod
    def crop_and_resize(self, boxes: Tensor, size: Tuple[int, int], **kwargs) -> 'Mask':
        pass

    @classmethod
    def cat(cls, structs: Sequence['Mask']) -> Any:
        warnings.warn(f"'{cls.__name__}.cat' not implemented.")
        return structs

    def __len__(self) -> int:
        return len(self.instances)

    def extra_repr(self) -> str:
        return f"mode={self.mode}"


_polygons_type = Sequence[Sequence[Union[torch.Tensor, np.ndarray, Sequence[float]]]]


class PolygonList(Mask):
    def __init__(self, size: Tuple[int, int], polygons: _polygons_type) -> None:

        def _make_array(t):
            if isinstance(t, torch.Tensor):
                t = t.cpu().numpy()
            polygon = np.array(t, dtype='float64', copy=True)
            assert len(polygon) % 2 == 0 and len(polygon) >= 6
            return polygon

        def _process_polygons(polygons_per_instance) -> List[np.ndarray]:
            if not isinstance(polygons_per_instance, (list, tuple, torch.Tensor, np.ndarray)):
                raise ValueError("'polygons_per_instance' must be a list, tuple, torch.Tensor "
                                 f"or np.ndarray, got {type(polygons_per_instance)}.")

            return [_make_array(p) for p in polygons_per_instance]

        if not isinstance(polygons, (list, tuple)):
            raise ValueError(f"'polygons' must be a list or tuple, got {type(polygons)}.")

        polygons: List[List[np.ndarray]] = \
            [_process_polygons(polygons_per_instance) for polygons_per_instance in polygons]

        super(PolygonList, self).__init__(size, polygons, MaskMode.POLY)

    def _convert_to_mask(self, size: Optional[Tuple[int, int]] = None, **kwargs) -> 'BinaryMaskList':
        if size is None:
            size = self.size

        if len(self.instances) == 0:
            width, height = size
            masks = torch.empty((0, height, width), dtype=torch.uint8)
        else:
            masks = MaskMode.convert(self.instances, self.mode, MaskMode.MASK, size=size)

        return BinaryMaskList(size, masks)

    def affine(self, trans: Union[np.ndarray, List[np.ndarray]], size: Tuple[int, int], **kwargs) -> 'PolygonList':
        self._check_size(size)

        if not isinstance(trans, (list, tuple)):
            trans = [trans] * len(self.instances)

        affined_polygons = []
        for polygons_per_instance, t in zip(self.instances, trans):
            affined_polygons_per_instance = []
            for polygon in polygons_per_instance:
                points: np.ndarray = polygon.reshape(-1, 2)
                affined_points = np.column_stack((points, np.ones(len(points))))
                affined_points = affined_points.dot(t.T).reshape(-1)
                affined_polygons_per_instance.append(affined_points)
            affined_polygons.append(affined_polygons_per_instance)
        return type(self)(size, affined_polygons)

    def crop_and_resize(self, boxes: Tensor, size: Tuple[int, int], **kwargs) -> 'PolygonList':
        self._check_boxes_shape(boxes)

        boxes = boxes.cpu().numpy()
        polygons = self.clone().instances
        for polygons_per_instance, box in zip(polygons, boxes):
            left, top, right, bottom = box
            crop_width, crop_height = right - left, bottom - top

            ratio_width = size[0] / max(crop_width, 0.1)
            ratio_height = size[1] / max(crop_height, 0.1)

            if ratio_height == ratio_width:
                for p in polygons_per_instance:
                    p[0::2] -= left
                    p[1::2] -= top
                    p *= ratio_width
            else:
                for p in polygons_per_instance:
                    p[0::2] -= left
                    p[1::2] -= top
                    p[0::2] *= ratio_width
                    p[1::2] *= ratio_height

        return type(self)(size, polygons)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == 'instances':
                copyed = [[copy.deepcopy(p) for p in polygons_per_instance]
                          for polygons_per_instance in self.instances]
            else:
                copyed = copy.deepcopy(v, memo)
            setattr(result, k, copyed)
        return result

    def crop(self, box: Sequence[Union[int, float]], **kwargs) -> 'PolygonList':
        # TODO box: Tuple[int, int, int, int]
        left, top, right, bottom = box
        crop_width, crop_height = int(right - left), int(bottom - top)

        width, height = self.size
        left = min(max(left, 0), width - 1)  # TODO to remove -1
        top = min(max(top, 0), height - 1)  # TODO to remove -1
        # right = min(max(right, 0), width)
        # bottom = min(max(bottom, 0), height)

        polygons = self.clone().instances

        if not left == top == 0:
            for polygons_per_instance in polygons:
                for p in polygons_per_instance:
                    p[0::2] -= left
                    p[1::2] -= top

        return type(self)((crop_width, crop_height), polygons)

    def resize(self, size: Tuple[int, int], **kwargs) -> 'PolygonList':
        self._check_size(size)

        width, height = self.size
        ratio_width = size[0] / max(width, 1e-4)
        ratio_height = size[1] / max(height, 1e-4)

        polygons = []
        if ratio_width == ratio_height:
            for i in range(len(self.instances)):
                polygons_per_instance = []
                for p in self.instances[i]:
                    poly = p * ratio_width
                    polygons_per_instance.append(poly)
                polygons.append(polygons_per_instance)
        else:
            for i in range(len(self.instances)):
                polygons_per_instance = []
                for p in self.instances[i]:
                    poly = np.empty_like(p)
                    poly[0::2] = p[0::2] * ratio_width
                    poly[1::2] = p[1::2] * ratio_height
                    polygons_per_instance.append(poly)
                polygons.append(polygons_per_instance)

        return type(self)(size, polygons)

    def to(self, device: Union[str, torch.device]) -> 'PolygonList':
        return self

    def transpose(self, method: int, to_remove: int = 0, **kwargs) -> 'PolygonList':
        if method == FLIP_LEFT_RIGHT:
            dim = 0
        elif method == FLIP_TOP_BOTTOM:
            dim = 1
        else:
            raise NotImplementedError("Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented.")

        polygons = self.clone().instances

        for polygons_per_instance in polygons:
            for p in polygons_per_instance:
                p[dim::2] = (self.size[dim] - to_remove) - p[dim::2]

        return type(self)(self.size, polygons)

    def __getitem__(self, item) -> 'PolygonList':
        if isinstance(item, int):
            selected_polygons = [self.instances[item]]
        elif isinstance(item, slice):
            selected_polygons = self.instances[item]
        else:
            # advanced indexing on a single dimension
            selected_polygons = []
            if isinstance(item, torch.Tensor) and (item.dtype == torch.uint8 or item.dtype == torch.bool):
                item = item.nonzero(as_tuple=False)
                item = item.squeeze(1) if item.numel() > 0 else item
                item = item.tolist()
            for i in item:
                selected_polygons.append(self.instances[i])
        return type(self)(self.size, selected_polygons)

    def __iter__(self) -> Iterator:
        for i in range(len(self)):
            yield type(self)(self.size, [self.instances[i]])


_masks_type = Union[torch.Tensor, Sequence[Union[torch.Tensor, Dict]]]


