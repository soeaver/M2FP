import numpy as np
import warnings
from typing import Any, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from pycocotools import mask as maskUtils


from .base_struct import FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM, BaseStruct

def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

class Segmentation(BaseStruct):
    ignore_label: int = 0
    label_shift: int = 1

    @classmethod
    def cat(cls, structs: Sequence['Segmentation']) -> Any:
        warnings.warn(f"'{cls.__name__}.cat' not implemented.")
        return structs


class SemanticSegmentation(Segmentation):
    __ann_type__: str = 'semseg'

    def __init__(self, size: Tuple[int, int], semseg, length: int = 0) -> None:
        super(SemanticSegmentation, self).__init__(size)

        device = semseg.device if isinstance(semseg, torch.Tensor) else 'cpu'
        semseg = torch.as_tensor(semseg, dtype=torch.uint8, device=device)

        # single channel
        if semseg.ndim == 2:
            semseg = semseg.unsqueeze(0)
        elif semseg.ndim != 3:
            raise ValueError(f"semseg should have 2 or 3 dims, got {semseg.ndim}.")

        if semseg.size(0) != 1:
            raise ValueError(f"first dim of semseg should be 1, got {semseg.size(0)}.")

        self.tensor: Tensor = semseg
        self.length: int = length

    @classmethod
    def flip_prob(cls, prob: Tensor, shift: bool = False,
                  flip_map: Optional[Tuple[Tuple[int, int], ...]] = None) -> Tensor:
        flipped_prob = prob.flip(3)
        if flip_map is None:
            flip_map = cls.__flip_map__
        for left, right in flip_map:
            flipped_right = flipped_prob[:, left, :, :]
            flipped_left = flipped_prob[:, right, :, :].clone()
            flipped_prob[:, right, :, :] = flipped_right
            flipped_prob[:, left, :, :] = flipped_left
        if shift:
            flipped_prob[:, :, :, 1:] = flipped_prob[:, :, :, :-1]
        return flipped_prob

    def crop(self, box: Sequence[Union[int, float]], **kwargs) -> 'SemanticSegmentation':
        # TODO box: Tuple[int, int, int, int]
        left, top, right, bottom = map(int, map(round, box))

        # bad box
        if right < left or bottom < top:
            raise ValueError(f"Bad box: {box}.")

        width, height = self.size
        crop_width, crop_height = right - left, bottom - top

        padding = (-left, right - width, -top, bottom - height)
        cropped_semseg = F.pad(self.tensor, padding)

        return type(self)((crop_width, crop_height), cropped_semseg, length=self.length)

    def resize(self, size: Tuple[int, int], **kwargs) -> 'SemanticSegmentation':
        self._check_size(size)

        width, height = size
        resized_semseg = F.interpolate(
            self.tensor[None].to(torch.float32),
            size=(height, width),
            mode='nearest',
        )[0].type_as(self.tensor)

        return type(self)(size, resized_semseg, length=self.length)

    def to(self, device: Union[str, torch.device]) -> 'SemanticSegmentation':
        return type(self)(self.size, self.tensor.to(device), length=self.length)

    def transpose(self, method: int, **kwargs) -> 'SemanticSegmentation':
        if method == FLIP_LEFT_RIGHT:
            dim = 2
        elif method == FLIP_TOP_BOTTOM:
            dim = 1
        else:
            raise NotImplementedError("Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented.")

        flipped_semseg = self.tensor.flip(dim)

        for ori_label, new_label in self.__flip_map__:
            left = flipped_semseg == ori_label
            right = flipped_semseg == new_label
            flipped_semseg[left] = new_label
            flipped_semseg[right] = ori_label

        return type(self)(self.size, flipped_semseg, length=self.length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, item) -> 'SemanticSegmentation':
        return type(self)(self.size, self.tensor, length=self.length)


class PanopticSegmentation(Segmentation):
    __ann_type__: str = 'panoseg'


# TODO refactor
def semseg_batch_resize(tensors, size_divisible=0, scale=1 / 8, ignore_value=255):
    assert isinstance(tensors, list)
    if size_divisible > 0:
        max_size = tuple(max(s) for s in zip(*[semseg.shape for semseg in tensors]))
        # TODO Ideally, just remove this and let me model handle arbitrary
        # input sizs
        if size_divisible > 0:
            import math

            stride = size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)

        batch_shape = (len(tensors),) + max_size
        batched_semsegs = tensors[0].new(*batch_shape).zero_() + ignore_value
        for semseg, pad_semseg in zip(tensors, batched_semsegs):
            pad_semseg[: semseg.shape[0], : semseg.shape[1], : semseg.shape[2]].copy_(semseg)

        _, _, height, width = batched_semsegs.shape

        width, height = int(width * float(scale) + 0.5), int(height * float(scale) + 0.5)
        # Height comes first here!
        batched_resized_semsegs = F.interpolate(
            batched_semsegs.float(),
            size=(height, width),
            mode="nearest",
        ).type_as(batched_semsegs)

        return batched_resized_semsegs


def convert_pano_to_semseg(semseg, pano_anns, ignore_label, label_format, file_name):
    """Pre-process the panoptic annotations to semantic annotations,
    it can convert coco or cityscape format, but now, it only support
    coco format. It supports two mainstream process ways, get all
    thing's ids to one or not, which is using 'extra_field['convert_format']'
    to control.
    """
    categories = pano_anns['categories']
    thing_ids = [c['id'] for c in categories if c['isthing'] == 1]
    stuff_ids = [c['id'] for c in categories if c['isthing'] == 0]

    panoptic = rgb2id(semseg[..., ::-1])
    output = np.full(panoptic.shape, ignore_label, dtype=np.uint8)

    if label_format == 'stuff_only':
        assert len(thing_ids) < ignore_label
        id_map = {stuff_id: i for i, stuff_id in enumerate(stuff_ids, 1)}
        thing_map = {thing_id: 0 for thing_id in thing_ids}
        id_map.update(thing_map)
    elif label_format == 'stuff_thing':
        pano_ids = stuff_ids + thing_ids
        assert len(pano_ids) < ignore_label
        id_map = {pano_id: i for i, pano_id in enumerate(pano_ids)}
    else:
        raise NotImplementedError
    id_map.update({0: ignore_label})

    # TODO find a ideal way to get specific annotation
    for ann in pano_anns['annotations']:
        if ann['file_name'] == file_name:
            segments = ann['segments_info']
            break

    for seg in segments:
        cat_id = seg["category_id"]
        output[panoptic == seg["id"]] = id_map[cat_id]

    return output


def paste_poly_anno(semseg: np.ndarray, poly_segs: Sequence[Any], classes: Sequence[int]) -> None:
    assert len(poly_segs) == len(classes)

    h, w = semseg.shape
    for poly_seg, class_id in zip(poly_segs, classes):
        if type(poly_seg) == list:
            rles = maskUtils.frPyObjects(poly_seg, h, w)
            rle = maskUtils.merge(rles)
        elif type(poly_seg['counts']) == list:
            rle = maskUtils.frPyObjects(poly_seg, h, w)
        else:
            rle = poly_seg
        m = maskUtils.decode(rle)
        semseg[m == 1] = class_id


def convert_poly_to_semseg(size, label_format, label_shift=0, thing_classes=[], thing_segs=[], stuff_cat_ids=None,
                           stuff_anns=None, json_category_id_to_contiguous_id=None):
    w, h = size
    init_label = 0
    stuff_segs = []
    stuff_classes = []

    if label_format == 'thing_only':
        thing_classes = [c + label_shift for c in thing_classes]
    else:
        assert stuff_cat_ids is not None
        assert stuff_anns is not None

        stuff_segs = [obj["segmentation"] for obj in stuff_anns]

        if label_format == 'stuff_only':
            thing_classes = [0 for _ in thing_classes]
            cat_ids_map = {v: i for i, v in enumerate(stuff_cat_ids)}
            init_label = cat_ids_map[stuff_cat_ids[-1]] + 1
        elif label_format == 'stuff_thing':
            assert json_category_id_to_contiguous_id is not None
            assert label_shift == 0
            cat_ids_map = json_category_id_to_contiguous_id
            init_label = cat_ids_map[stuff_cat_ids[-1]]  # category 'other'
        else:
            raise ValueError(f"Unsupported label_format: {label_format}")

        stuff_classes = [cat_ids_map[obj["category_id"]] + label_shift for obj in stuff_anns]

    semseg = np.full((h, w), init_label, dtype=np.int64)    # in case classes > 255
    paste_poly_anno(semseg, stuff_segs, stuff_classes)
    paste_poly_anno(semseg, thing_segs, thing_classes)
    return semseg
