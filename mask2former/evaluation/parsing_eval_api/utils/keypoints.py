import functools
import numpy as np
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import Tensor

from .base_struct import FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM, BaseStruct


@functools.lru_cache()
def get_gaussian_kernel2d(radius: int) -> np.ndarray:
    sigma = radius / 3
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    x = np.exp(-((x / sigma) ** 2) / 2)
    return x * x[:, None]


class Keypoints(BaseStruct):
    __ann_type__: str = 'keypoints'
    __connections__: Tuple[Tuple[int, int], ...] = ()

    def __init__(self, size: Tuple[int, int], keypoints: Any) -> None:
        super(Keypoints, self).__init__(size)

        device = keypoints.device if isinstance(keypoints, torch.Tensor) else 'cpu'
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32, device=device)

        if len(keypoints) == 0:
            keypoints = keypoints.view(0, 0, 3)

        if keypoints.ndim == 2:
            keypoints = keypoints.view(keypoints.size(0), keypoints.size(1) // 3, 3)
        elif keypoints.ndim != 3:
            raise ValueError(f"keypoints should have 2 or 3 dims, got {keypoints.ndim}.")

        self.tensor: Tensor = keypoints

    @functools.lru_cache()
    def get_flip_inds(self, inds_length: int, device: Union[str, torch.device] = 'cpu') -> Tensor:
        flip_inds = list(range(inds_length))
        for left, right in self.__flip_map__:
            flip_inds[left] = right
            flip_inds[right] = left
        return torch.as_tensor(flip_inds, dtype=torch.int64, device=device)

    def get_visable_half_body(self, upper_body_ids) -> Tuple[Tensor, Tensor]:
        device = self.tensor.device
        upper_inds = torch.zeros(self.tensor.shape[:2], dtype=torch.bool, device=device)
        upper_body_ids = torch.as_tensor(upper_body_ids, dtype=torch.int64, device=device)
        upper_inds[:, upper_body_ids] = True
        lower_inds = ~upper_inds

        visable = self.tensor[:, :, 2] > 0

        upper_inds &= visable
        lower_inds &= visable

        if len(self) == 1:  # for instance, return shape (num_points, 2)
            joints = self.tensor[:, :, :2]
            return joints[upper_inds], joints[lower_inds]
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

    def get_heatmap(self, output_size: Tuple[int, int], radius) -> Tuple[Tensor, Tensor]:
        width, height = self.size
        ow, oh = output_size
        sw, sh = width / ow, height / oh

        num_instance, num_keypoints = self.tensor.shape[:2]
        heatmap = np.zeros((num_instance, num_keypoints, oh, ow), dtype=np.float32)
        gaussian = get_gaussian_kernel2d(radius)

        weight = self.tensor[:, :, 2:3].clamp(max=1)
        for ins_id in range(num_instance):
            for pt_id in range(num_keypoints):
                x, y, v = self.tensor[ins_id, pt_id]

                if v < 0.5:
                    weight[ins_id, pt_id] = 0
                    continue

                x, y = int(x / sw + 0.5), int(y / sh + 0.5)

                left, right = min(x, radius), min(ow - x, radius + 1)
                top, bottom = min(y, radius), min(oh - y, radius + 1)

                if left + right <= 0 or top + bottom <= 0:
                    weight[ins_id, pt_id] = 0
                    continue

                heatmap[ins_id, pt_id, y - top:y + bottom, x - left:x + right] = \
                    gaussian[radius - top:radius + bottom, radius - left:radius + right]

        heatmap = torch.as_tensor(heatmap, dtype=torch.float32, device=self.tensor.device)

        return heatmap, weight

    def affine(self, trans: Union[np.ndarray, List[np.ndarray]], size: Tuple[int, int], **kwargs) -> 'Keypoints':
        self._check_size(size)

        if not isinstance(trans, (list, tuple)):
            trans = [trans] * len(self.tensor)

        affined_keypoints = self.tensor.cpu().numpy().copy()
        affined_keypoints[:, :, 2] = 1
        for points, t in zip(affined_keypoints, trans):
            points[:, :2] = points.dot(t.T)
        affined_keypoints = torch.as_tensor(affined_keypoints,
                                            dtype=self.tensor.dtype,
                                            device=self.tensor.device)
        affined_keypoints[:, :, 2] = self.tensor[:, :, 2]
        return type(self)(size, affined_keypoints)

    @classmethod
    def cat(cls, structs) -> 'Keypoints':
        cated_keypoints = torch.cat([struct.tensor for struct in structs], dim=0)
        return cls(structs[0].size, cated_keypoints)

    def crop(self, box: Tuple[int, int, int, int], **kwargs) -> 'Keypoints':
        left, top, right, bottom = box
        crop_width, crop_height = right - left, bottom - top

        x, y = self.tensor[..., 0], self.tensor[..., 1]
        inds = (x >= left) & (x <= right) & (y >= top) & (y <= bottom)
        cropped_keypoints = torch.zeros_like(self.tensor)
        cropped_keypoints[inds] = self.tensor[inds]

        return type(self)((crop_width, crop_height), cropped_keypoints)

    def resize(self, size: Tuple[int, int], **kwargs) -> 'Keypoints':
        self._check_size(size)

        width, height = self.size
        ratio_width, ratio_height = size[0] / width, size[1] / height

        resized_keypoints = self.tensor.clone()
        if ratio_width == ratio_height:
            resized_keypoints[..., :2] *= ratio_width
        else:
            resized_keypoints[..., 0] *= ratio_width
            resized_keypoints[..., 1] *= ratio_height

        return type(self)(size, resized_keypoints)

    def to(self, device: Union[str, torch.device]) -> 'Keypoints':
        return type(self)(self.size, self.tensor.to(device))

    def transpose(self, method: int, to_remove: int = 0, **kwargs) -> 'Keypoints':
        if method == FLIP_LEFT_RIGHT or method == FLIP_TOP_BOTTOM:
            flip_inds = self.get_flip_inds(self.tensor.size(1), device=self.tensor.device)
            flipped_data = self.tensor[:, flip_inds]  # copyed
            flipped_data[..., 0] = (self.size[0] - to_remove) - flipped_data[..., 0]

            # Maintain COCO convention that if visibility == 0, then x, y = 0
            inds = flipped_data[..., 2] == 0
            flipped_data[inds] = 0
        else:
            raise NotImplementedError(
                f"Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented, got {method}.")

        return type(self)(self.size, flipped_data)

    def __len__(self) -> int:
        return len(self.tensor)

    def __getitem__(self, item) -> 'Keypoints':
        return type(self)(self.size, self.tensor[item])


class PersonKeypoints(Keypoints):

    NAMES: Tuple[str, ...] = (
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    )


# TODO refactor
# TODO make this nicer, this is a direct translation from C2 (but removing the inner loop)
def keypoints_to_heatmap(keypoints, rois, h, w):
    if rois.numel() == 0:
        return rois.new().long(), rois.new().long()

    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = w / (rois[:, 2] - rois[:, 0])
    scale_y = h / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]

    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()

    x[x_boundary_inds] = w - 1
    y[y_boundary_inds] = h - 1

    valid_loc = (x >= 0) & (y >= 0) & (x < w) & (y < h)
    vis = keypoints[..., 2] > 0
    valid = (valid_loc & vis).long()

    lin_ind = y * w + x
    heatmaps = lin_ind * valid

    return heatmaps, valid
