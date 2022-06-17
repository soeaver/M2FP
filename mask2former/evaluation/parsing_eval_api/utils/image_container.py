import itertools
import warnings
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

import torch
from torch import Tensor


from .base_struct import BaseStruct
from .bounding_box import BoundingBox, BoxMode, boundingbox_iou
from .label import Label

_container_type = Union[BaseStruct, Tensor]
_seq_str = Union[str, Sequence[str]]
_extra_data_type = Union[int, float, bool]

BBOX_FIELD = 'bbox'
LABEL_FIELD = 'label'
SCORE_FIELD = 'bbox_scores'


class ImageContainer(BaseStruct):
    def __init__(self, size: Tuple[int, int], **kwargs) -> None:
        super(ImageContainer, self).__init__(size)

        self._structs: Dict[str, BaseStruct] = {}  # Label/BoundingBox/Mask/Keypoints
        self._fields: Dict[str, Tensor] = {}
        self._extra_data: Dict[str, _extra_data_type] = {}

        self._ignore_length: Set[str] = set()

        for key, value in kwargs.items():
            self[key] = value

    @property
    def containers(self) -> Tuple:
        return tuple(itertools.chain.from_iterable((self._structs, self._fields)))

    @property
    def ann_types(self) -> Set[str]:
        ann_types = set()
        for struct in self._structs.values():
            if struct.__ann_type__ is not None:
                ann_types.add(struct.__ann_type__)
        return ann_types

    def named_structs(self) -> Iterator[Tuple[str, BaseStruct]]:
        yield from self._structs.items()

    def named_fields(self) -> Iterator[Tuple[str, Tensor]]:
        yield from self._fields.items()

    def named_extra_data(self) -> Iterator[Tuple[str, _extra_data_type]]:
        yield from self._extra_data.items()

    def structs(self) -> Iterator[BaseStruct]:
        yield from self._structs.values()

    def fields(self) -> Iterator[Tensor]:
        yield from self._fields.values()

    def extra_data(self) -> Iterator[_extra_data_type]:
        yield from self._extra_data.values()

    def get(self, key: str, default=None) -> Union[_container_type, _extra_data_type, None]:
        if not isinstance(key, str):
            raise TypeError(f"Unsupported 'key' type: '{type(key)}'")

        if key in self._structs:
            return self._structs[key]
        if key in self._fields:
            return self._fields[key]
        if key in self._extra_data:
            warnings.warn(f"'{key}' is a extra data, use 'get_extra_data' please.")
            return self._extra_data[key]
        return default

    def pop(self, key: str) -> Union[_container_type, _extra_data_type]:
        if not isinstance(key, str):
            raise TypeError(f"Unsupported 'key' type: '{type(key)}'")

        if key in self._structs:
            return self._structs.pop(key)
        if key in self._fields:
            return self._fields.pop(key)
        if key in self._extra_data:
            return self._extra_data.pop(key)
        raise KeyError(key)

    def ignore_length_of(self, names: Union[str, Tuple[str], List[str], Set[str]]) -> None:
        if isinstance(names, str):
            names = [names]
        self._ignore_length.update(names)

    def add_extra_data(self, name: str, data: _extra_data_type) -> None:
        if name in self._extra_data:
            raise KeyError(name)
        self._extra_data[name] = data

    def set_extra_data(self, name: str, data: _extra_data_type) -> None:
        if name in self._extra_data:
            expected_type = type(self._extra_data[name])
            if not isinstance(data, type(self._extra_data[name])):
                raise TypeError(f"Type mismatch: expect {expected_type}, got {type(data)}.")
        self._extra_data[name] = data

    def get_extra_data(self, name: str, default=None) -> _extra_data_type:
        return self._extra_data.get(name, default)

    def pop_extra_data(self, name: str) -> _extra_data_type:
        return self._extra_data.pop(name)

    def has_extra_data(self, name: str) -> bool:
        return name in self._extra_data

    def has_field(self, name: str) -> bool:
        return name in self._fields

    def _apply_structs(self, _func: str, _out: 'ImageContainer', *args, **kwargs) -> None:
        for key, value in self._structs.items():
            _out._structs[key] = getattr(value, _func)(*args, **kwargs)

    def _apply_fields(self, _func: str, _out: 'ImageContainer', *args, **kwargs) -> None:
        for key, value in self._fields.items():
            _out._fields[key] = getattr(value, _func)(*args, **kwargs)

    def update(self, container: Union['ImageContainer', Dict[str, _container_type]]) -> None:
        if isinstance(container, ImageContainer):
            if self.size != container.size:
                raise ValueError(f"Size mismatch: expect {self.size}, got {container.size}.")
            self._ignore_length.update(container._ignore_length)
            for name, value in container._extra_data.items():
                self.set_extra_data(name, value)
            data = container
        elif isinstance(container, dict):
            data = container.items()
        else:
            raise TypeError(f"Unsupported 'container' type: '{type(container)}'.")

        for key, value in data:
            self[key] = value

    def set_size(self, size: Tuple[int, int], ignore_warning: bool = False) -> None:
        if not ignore_warning:
            warnings.warn("Force modify size!")
        self.size = size
        for struct in self._structs.values():
            struct.size = size

    def select(self, with_data: _seq_str = (), deepcopy: bool = False) -> 'ImageContainer':
        if isinstance(with_data, str):
            with_data = (with_data, )
        if not isinstance(with_data, (list, tuple)):
            raise TypeError(f"Unsupported 'with_data' type: {type(with_data)}.")

        if len(with_data) == 0:
            if deepcopy:
                return self.clone()
            with_data = self.containers

        container = type(self)(self.size)
        container._ignore_length.update(self._ignore_length)
        container._extra_data.update(self._extra_data)

        for key in with_data:
            container[key] = self[key].clone() if deepcopy else self[key]

        return container

    def copy(self, with_data: _seq_str = (), deepcopy: bool = True) -> 'ImageContainer':
        return self.select(with_data=with_data, deepcopy=deepcopy)

    def clip_to_image(self, threshold: float = 0.0, remove_empty: bool = True) -> 'ImageContainer':
        container = self.copy(deepcopy=False)

        if BBOX_FIELD not in self._structs:
            return container

        bbox_struct = self._structs[BBOX_FIELD].clip()
        container._structs[BBOX_FIELD] = bbox_struct
        if remove_empty:
            container = container[bbox_struct.nonempty(threshold)]
        return container

    @classmethod
    def cat(cls, containers: Sequence['ImageContainer']) -> 'ImageContainer':
        ele = containers[0]
        container = cls(ele.size)
        no_length_set = set(itertools.chain.from_iterable(
            cnt._ignore_length for cnt in containers))
        container._ignore_length.update(no_length_set)

        extra_structs = set(itertools.chain.from_iterable(
            cnt._structs for cnt in containers))
        for key in extra_structs:
            structs = [cnt._structs[key] for cnt in containers]
            container._structs[key] = ele[key].cat(structs)

        extra_fields = set(itertools.chain.from_iterable(
            cnt._fields for cnt in containers))
        for key in extra_fields:
            fields = [cnt._fields[key] for cnt in containers]
            container._fields[key] = torch.cat(fields, dim=0)

        extra_data = set(itertools.chain.from_iterable(
            cnt._extra_data for cnt in containers))
        for key in extra_data:
            data = None
            for cnt in containers:
                if key in cnt._extra_data:
                    cur_data = cnt._extra_data[key]
                    if data is None:
                        data = cur_data
                    else:
                        assert data == cur_data
            container._extra_data[key] = data

        return container

    def crop(self, box: Tuple[int, int, int, int], **kwargs) -> 'ImageContainer':
        container = type(self)((box[2] - box[0], box[3] - box[1]))
        container._ignore_length.update(self._ignore_length)
        container._extra_data.update(self._extra_data)
        self._apply_structs('crop', container, box, **kwargs)
        container._fields.update(self._fields)
        return container

    def resize(self, size: Tuple[int, int], **kwargs) -> 'ImageContainer':
        if size == self.size:
            return self.copy(deepcopy=False)
        container = type(self)(size)
        container._ignore_length.update(self._ignore_length)
        container._extra_data.update(self._extra_data)
        self._apply_structs('resize', container, size, **kwargs)
        container._fields.update(self._fields)
        return container

    def to(self, device: Union[str, torch.device]) -> 'ImageContainer':
        container = type(self)(self.size)
        container._ignore_length.update(self._ignore_length)
        container._extra_data.update(self._extra_data)
        self._apply_structs('to', container, device)
        self._apply_fields('to', container, device)
        return container

    def transpose(self, method: int, **kwargs) -> 'ImageContainer':
        container = type(self)(self.size)
        container._ignore_length.update(self._ignore_length)
        container._extra_data.update(self._extra_data)
        self._apply_structs('transpose', container, method, **kwargs)
        container._fields.update(self._fields)
        return container

    def __len__(self) -> int:
        lengths = set(len(value) for key, value in self if key not in self._ignore_length)

        if len(lengths) == 0:
            return 0
        if len(lengths) == 1:
            return lengths.pop()

        raise ValueError(lengths)

    def __getitem__(self, item) -> Union['ImageContainer', _container_type]:
        if item is None:
            return self.copy(deepcopy=False)
        elif isinstance(item, str):
            val = self.get(item)
            if val is None:
                raise KeyError(item)
            return val
        else:
            container = type(self)(self.size)
            container._ignore_length.update(self._ignore_length)
            container._extra_data.update(self._extra_data)
            self._apply_structs('__getitem__', container, item)
            self._apply_fields('__getitem__', container, item)
            return container

    def __setitem__(self, key: str, value: _container_type) -> None:
        if not isinstance(key, str):
            raise TypeError(f"Unsupported 'key' type: '{type(key)}'")

        if key in self._extra_data:
            raise KeyError(f"'{key}' is a extra data, use 'set_extra_data' please.")

        to_check_length = False
        if key not in self._ignore_length:
            length = len(self)
            if length != 0:
                to_check_length = True
                length_value = len(value)

        if isinstance(value, BaseStruct) and not isinstance(value, ImageContainer):
            if to_check_length and length != length_value:
                raise ValueError(f"Length mismatch: expect {length}, got {length_value}.")
            if self.size != value.size:
                raise ValueError(f"Size mismatch: expect {self.size}, got {value.size}.")
            if key in self._fields:
                raise ValueError(f"Multiple assignment: "
                                 f"({key}: torch.Tensor) already exists, "
                                 f"got ({key}: {type(value)}).")
            self._structs[key] = value
        elif isinstance(value, torch.Tensor):
            if to_check_length and length != length_value:
                raise ValueError(f"Length mismatch: expect {length}, got {length_value}.")
            if key in self._structs:
                raise ValueError(f"Multiple assignment: "
                                 f"({key}: {self._structs[key]}) already exists, "
                                 f"got ({key}: torch.Tensor).")
            self._fields[key] = value
        else:
            raise TypeError(f"Unsupported 'value' type: '{type(value)}'")

    def __delitem__(self, key: str) -> None:
        self.pop(key)

    def __contains__(self, key) -> bool:
        return key in self._structs or key in self._fields or key in self._extra_data

    def __iter__(self) -> Iterator[Tuple[str, _container_type]]:
        yield from self.named_structs()
        yield from self.named_fields()

    def extra_repr(self) -> str:
        containers = self.containers
        if len(containers) == 1:
            return f"data=('{containers[0]}')"
        if len(containers) > 1:
            return f"data={str(containers)}"
        return ""

