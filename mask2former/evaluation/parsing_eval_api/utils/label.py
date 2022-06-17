from typing import Any, Tuple, Union

import torch
from torch import Tensor

from .base_struct import BaseStruct


class Label(BaseStruct):
    def __init__(self, size: Tuple[int, int], labels: Any) -> None:
        super(Label, self).__init__(size)

        device = labels.device if isinstance(labels, torch.Tensor) else 'cpu'
        labels = torch.as_tensor(labels, dtype=torch.int64, device=device)
        self._check_labels_shape(labels)

        self.tensor: Tensor = labels

    def _check_labels_shape(self, labels: Tensor) -> None:
        if not isinstance(labels, torch.Tensor):
            raise TypeError(f"labels should be torch.Tensor, got {type(labels)}.")
        if labels.ndim != 1:
            raise ValueError(f"labels should have 1 dims, got {labels.ndim}.")

    def get_visable_half_body(self, upper_body_ids) -> Tuple[Tensor, Tensor]:
        if len(self) == 1:  # for instance, return shape (num_points, 2)
            return torch.empty(0, 2), torch.empty(0, 2)
        else:
            raise NotImplementedError

    def affine(self, trans, size, **kwargs):
        return type(self)(size, self.tensor)

    @classmethod
    def cat(cls, structs) -> 'Label':
        cated_labels = torch.cat([struct.tensor for struct in structs], dim=0)
        return cls(structs[0].size, cated_labels)

    def crop(self, box: Tuple[int, int, int, int], **kwargs) -> 'Label':
        return type(self)((box[2] - box[0], box[3] - box[1]), self.tensor)

    def resize(self, size: Tuple[int, int], **kwargs) -> 'Label':
        return type(self)(size, self.tensor)

    def to(self, device: Union[str, torch.device]) -> 'Label':
        return type(self)(self.size, self.tensor.to(device))

    def transpose(self, method: int, **kwargs) -> 'Label':
        if len(self.__flip_map__) == 0:
            return type(self)(self.size, self.tensor)

        labels = self.tensor.clone()
        for ori_label, new_label in self.__flip_map__:
            left = labels == ori_label
            right = labels == new_label
            labels[left] = new_label
            labels[right] = ori_label
        return type(self)(self.size, labels)

    def __len__(self) -> int:
        return len(self.tensor)

    def __getitem__(self, item) -> 'Label':
        return type(self)(self.size, self.tensor[item])
