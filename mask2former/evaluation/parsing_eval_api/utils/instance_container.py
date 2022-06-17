
from typing import Any

import torch

from .base_struct import FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM
from .image_container import ImageContainer


TO_REMOVE = 1


class InstanceContainer(ImageContainer):

    @property
    def regions(self) -> torch.Tensor:
        try:
            return self['regions']
        except:
            raise RuntimeError("Init error: 'InstanceContainer' need regions in field.")

    @regions.setter
    def regions(self, boxes: Any) -> None:
        device = boxes.device if isinstance(boxes, torch.Tensor) else 'cpu'
        boxes = torch.as_tensor(boxes, dtype=torch.float64, device=device)

        if len(boxes) == 0:
            boxes = boxes.view(0, 5)

        if boxes.ndim == 1:
            boxes = boxes.view(1, -1)
        elif boxes.ndim != 2:
            raise ValueError(f"'boxes' should have 1 or 2 dims, got {boxes.ndim}.")

        if boxes.size(-1) == 4:
            a = boxes.new_zeros(boxes.size(0))
            boxes = torch.cat((boxes, a), dim=-1)
        elif boxes.size(-1) != 5:
            raise ValueError(
                f"Last dim of 'boxes' should have a size of 4 or 5, got {boxes.size(-1)}.")

        self['regions'] = boxes

    def get_regions_by_aspect_ratio(self, aspect_ratio: float) -> torch.Tensor:
        regions = self.regions.clone()
        _, _, w, h, _ = regions.unbind(dim=-1)  # not copyed

        inds0 = w > h * aspect_ratio
        inds1 = ~inds0

        h[inds0] = w[inds0] / aspect_ratio
        w[inds1] = h[inds1] * aspect_ratio

        return regions

    def affine(self, trans, size, **kwargs) -> 'InstanceContainer':
        container = type(self)(size)
        self._apply_structs('affine', container, trans, size, **kwargs)
        container._fields.update(self._fields)
        return container

    def transpose(self, method: int, to_remove: int = TO_REMOVE, **kwargs) -> 'InstanceContainer':
        container = super(InstanceContainer, self).transpose(method, to_remove=to_remove, **kwargs)

        width, height = container.size
        xc, yc, w, h, a = container.regions.unbind(dim=-1)

        if method == FLIP_LEFT_RIGHT:
            xc = (width - to_remove) - xc
        elif method == FLIP_TOP_BOTTOM:
            yc = (height - to_remove) - yc
        else:
            raise NotImplementedError("Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented")

        container.regions = torch.stack((xc, yc, w, h, a), dim=-1)

        return container
