import copy
from abc import ABCMeta, abstractclassmethod, abstractmethod
from enum import Enum, unique
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


@unique
class AnnType(Enum):
    BBOX = 'bbox'

    MASK = 'mask'

    KEYPOINTS = 'keypoints'

    PARSING = 'parsing'

    UV = 'uv'

    SEMSEG = 'semseg'

    PANOSEG = 'panoseg'

    @staticmethod
    def all_types():
        return set(map(str, AnnType))


class BaseStruct(metaclass=ABCMeta):
    __ann_type__: Optional[str] = None
    __flip_map__: Tuple[Tuple[int, int], ...] = ()

    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size

    def _check_size(self, size: Tuple[int, int]) -> None:
        if not (isinstance(size, tuple) and len(size) == 2 and  # noqa: W504
                isinstance(size[0], int) and isinstance(size[1], int)):
            raise TypeError("size must be Tuple[int, int].")

    @property
    def size(self) -> Tuple[int, int]:
        return self._size

    @size.setter
    def size(self, size) -> None:
        self._check_size(size)
        self._size = tuple(size)

    @property
    def width(self) -> int:
        return self._size[0]

    @property
    def height(self) -> int:
        return self._size[1]

    def clip(self, size: Optional[Tuple[int, int]] = None) -> 'BaseStruct':
        width, height = self.size if size is None else size
        return self.crop((0, 0, width, height))

    @classmethod
    def safe_cat(cls, structs) -> 'BaseStruct':
        if not isinstance(structs, (list, tuple)) or len(structs) == 0:
            raise ValueError("'structs' must be a list or tuple, "
                             "and contains at least one element.")

        types = set(type(struct) for struct in structs)
        if len(types) > 1:
            raise TypeError(f"multi types: {types}.")
        if cls not in types:
            raise TypeError(f"wrong type: {types.pop()}.")

        sizes = set(struct.size for struct in structs)
        if len(sizes) > 1:
            raise ValueError(f"multi sizes: {sizes}.")

        return cls.cat(structs)

    @abstractclassmethod
    def cat(cls, structs) -> 'BaseStruct':
        pass

    def clone(self) -> 'BaseStruct':
        return copy.deepcopy(self)

    def nonempty(self, threshold: float = 0.0) -> Optional[Tensor]:
        return None

    @abstractmethod
    def crop(self, box: Tuple[int, int, int, int], **kwargs) -> 'BaseStruct':
        pass

    @abstractmethod
    def resize(self, size: Tuple[int, int], **kwargs) -> 'BaseStruct':
        pass

    @abstractmethod
    def to(self, device: Union[str, torch.device]) -> 'BaseStruct':
        pass

    @abstractmethod
    def transpose(self, method: int, **kwargs) -> 'BaseStruct':
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, item) -> 'BaseStruct':
        pass

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        extra_string = self.extra_repr()
        if len(extra_string) > 0:
            extra_string = ", " + extra_string
        return self.__class__.__name__ + f"(size={self.size}, length={len(self)}{extra_string})"
