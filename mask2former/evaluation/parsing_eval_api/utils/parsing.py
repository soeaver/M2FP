import cv2
import numpy as np
import warnings
from typing import Any, Iterator, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from .base_struct import FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM, BaseStruct


class Parsing(BaseStruct):
    __ann_type__: str = 'parsing'

    def __init__(self, size: Tuple[int, int], parsing: Any) -> None:
        super(Parsing, self).__init__(size)

        device = parsing.device if isinstance(parsing, torch.Tensor) else 'cpu'
        parsing = torch.as_tensor(parsing, device=device)

        if parsing.ndim == 2:
            parsing = parsing[None]

        self._check_parsing_shape(parsing, size)

        self.tensor: Tensor = parsing

    def _check_parsing_shape(self, parsing: Tensor, size: Tuple[int, int]) -> None:
        if not isinstance(parsing, torch.Tensor):
            raise TypeError(f"'parsing' should be torch.Tensor, got {type(parsing)}.")
        if parsing.ndim != 3:
            raise ValueError(f"'parsing' should have 2 dims, got {parsing.ndim}.")
        width, height = size
        if parsing.shape[-2:] != (height, width):
            raise ValueError("'parsing' shape mismatch: "
                             f"expect height {parsing.size(1)} and width {parsing.size(2)}, "
                             f"got height {height} and width {width}.")

    def get_visable_half_body(self, upper_body_ids) -> Tuple[Tensor, Tensor]:
        device = self.tensor.device
        parsing_ids = torch.unique(self.tensor)

        upper_points, lower_points = [], []
        for parsing_id in parsing_ids:
            mask = torch.where(self.tensor == parsing_id, 1, 0)
            if mask.sum() > 100:
                bbox = mask_to_bbox(mask)
                if bbox is not None:
                    if parsing_id in upper_body_ids:
                        upper_points += bbox
                    else:
                        lower_points += bbox

        if len(self) == 1:  # for instance, return shape (num_points, 2)
            upper_points = torch.as_tensor(upper_points, dtype=torch.float32, device=device)
            lower_points = torch.as_tensor(lower_points, dtype=torch.float32, device=device)
            return upper_points.view(-1, 2), lower_points.view(-1, 2)
        else:
            raise NotImplementedError

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

    def affine(self, trans: Union[np.ndarray, List[np.ndarray]], size: Tuple[int, int], **kwargs) -> 'Parsing':
        self._check_size(size)

        if not isinstance(trans, (list, tuple)):
            trans = [trans] * len(self.tensor)

        parsing = self.tensor.cpu().numpy()
        dtype, device = self.tensor.dtype, self.tensor.device

        affined_parsing = []
        for mask, t in zip(parsing, trans):
            affined_mask = cv2.warpAffine(mask, t, size, flags=cv2.INTER_NEAREST)
            affined_parsing.append(affined_mask)
        affined_parsing = torch.as_tensor(affined_parsing, dtype=dtype, device=device)
        return type(self)(size, affined_parsing)

    @classmethod
    def cat(cls, structs: Sequence['Parsing']) -> 'Parsing':
        warnings.warn("'Parsing.cat' not implemented.")
        return structs

    def crop(self, box: Sequence[Union[int, float]], **kwargs) -> 'Parsing':
        # TODO box: Tuple[int, int, int, int]
        left, top, right, bottom = map(int, map(round, box))

        # bad box
        if right < left or bottom < top:
            raise ValueError(f"Bad box: {box}.")

        width, height = self.size
        crop_width, crop_height = right - left, bottom - top

        padding = (-left, right - width, -top, bottom - height)
        cropped_parsing = F.pad(self.tensor, padding)

        return type(self)((crop_width, crop_height), cropped_parsing)

    def resize(self, size: Tuple[int, int], **kwargs) -> 'Parsing':
        self._check_size(size)

        width, height = size
        resized_parsing = F.interpolate(
            self.tensor[None].to(torch.float32),
            size=(height, width),
            mode='nearest',
        )[0].type_as(self.tensor)

        return type(self)(size, resized_parsing)

    def to(self, device: Union[str, torch.device]) -> 'Parsing':
        return type(self)(self.size, self.tensor.to(device))

    def transpose(self, method: int, **kwargs) -> 'Parsing':
        if method == FLIP_LEFT_RIGHT:
            dim = 2
        elif method == FLIP_TOP_BOTTOM:
            dim = 1
        else:
            raise NotImplementedError("Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented.")

        flipped_parsing = self.tensor.flip(dim)

        for ori_label, new_label in self.__flip_map__:
            left = flipped_parsing == ori_label
            right = flipped_parsing == new_label
            flipped_parsing[left] = new_label
            flipped_parsing[right] = ori_label

        return type(self)(self.size, flipped_parsing)

    def __len__(self) -> int:
        return len(self.tensor)

    def __getitem__(self, item) -> 'Parsing':
        return type(self)(self.size, self.tensor[item])

    def __iter__(self) -> Iterator:
        for i in range(len(self)):
            yield type(self)(self.size, self.tensor[i][None])


def mask_to_bbox(mask: Tensor) -> Tuple[int, int, int, int]:
    xs = mask.sum(dim=0).nonzero(as_tuple=True)[0]
    ys = mask.sum(dim=1).nonzero(as_tuple=True)[0]

    if len(xs) == 0 or len(ys) == 0:
        return None

    return int(xs[0]), int(xs[-1]), int(ys[0]), int(ys[-1])
