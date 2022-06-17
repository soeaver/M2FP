import cv2
import dpath.util
import json
import numpy as np
import os
import warnings
from pycocotools.coco import COCO
from typing import Callable, Dict, List, Optional, Set, Tuple
from PIL import Image, ImageFile

import torch
from torchvision.datasets.coco import CocoDetection

from .utils import (BoundingBox, BoxMode, ImageContainer,
                                     InstanceContainer, Label, MaskMode,
                                     Parsing, PersonKeypoints, PolygonList,
                                     SemanticSegmentation,
                                     convert_pano_to_semseg,
                                     convert_poly_to_semseg)

from .utils import (count_visible_keypoints, get_parsing, has_valid_bbox,
                    has_valid_person, has_visible_hier, is_not_crowd,
                    print_instances_class_histogram)


ImageFile.LOAD_TRUNCATED_IMAGES = True


class ParsingGT(CocoDetection):
    def __init__(self, root: str, ann_file: str, ann_types: Set[str],
                 ann_fields: Dict[str, Dict], transforms: Optional[Callable] = None, is_train: bool = False,
                 filter_crowd_ann: bool = True, filter_invalid_ann: bool = True, filter_empty_ann: bool = True,
                 **kwargs) -> None:
        super(ParsingGT, self).__init__(root, ann_file)
        coco = self.coco
        self.img_to_anns = [coco.imgToAnns[idx] for idx in self.ids]
        cat_ids = sorted(coco.getCatIds())
        self.classes = [c['name'] for c in coco.loadCats(cat_ids)]

        Label.__flip_map__ = dpath.util.get(ann_fields, "/bbox/flip_map", default=())
        PersonKeypoints.__flip_map__ = dpath.util.get(ann_fields, "/keypoints/flip_map", default=())
        PersonKeypoints.__connections__ = dpath.util.get(ann_fields, "/keypoints/connections", default=())
        Parsing.__flip_map__ = dpath.util.get(ann_fields, "/parsing/flip_map", default=())
        SemanticSegmentation.__flip_map__ = dpath.util.get(ann_fields, "/semseg/flip_map", default=())

        if 'semseg' in ann_types:
            self.semseg_format = dpath.util.get(ann_fields, "/semseg/semseg_format", default="mask")
            self.label_shift = dpath.util.get(ann_fields, "/semseg/label_shift", default=0)
            self.label_format = dpath.util.get(ann_fields, "/semseg/label_format", default="thing_only")

            if self.semseg_format == "mask":
                self.seg_root = dpath.util.get(ann_fields, "/semseg/seg_root", default=self.root.replace('img', 'seg'))
                self.name_trans = dpath.util.get(ann_fields, "/semseg/name_trans", default=('jpg', 'png'))

                if 'panoseg' in ann_types:
                    self.ignore_label = dpath.util.get(ann_fields, "/semseg/ignore_label")
                    seg_json = dpath.util.get(ann_fields, "/semseg/seg_json")
                    self.pano_anns = json.load(open(seg_json, 'r'))
                    panoseg_cat_ids = sorted(cat['id'] for cat in self.pano_anns['categories'])
                    self.pano_id_to_contiguous_id = {v: i for i, v in enumerate(panoseg_cat_ids)}
                    self.contiguous_id_to_pano_id = {i: v for i, v in enumerate(panoseg_cat_ids)}
            elif self.semseg_format == "poly":
                seg_json = dpath.util.get(ann_fields, "/semseg/seg_json")
                self.semseg_gt = COCO(seg_json)
                self.semseg_cat_ids = sorted(self.semseg_gt.getCatIds())
                cat_ids = sorted(np.unique(self.semseg_cat_ids + cat_ids).tolist())
            else:
                raise ValueError(f"Unsupported semseg gt format: '{self.semseg_format}'.")

        self.category_id_to_contiguous_id = {v: i for i, v in enumerate(cat_ids)}
        self.contiguous_id_to_category_id = {i: v for i, v in enumerate(cat_ids)}

        self.ann_types = ann_types
        self.ann_fields = ann_fields
        self._transforms = transforms
        self.kwargs = kwargs

        if filter_crowd_ann:
            self.filter_crowd()

        if filter_invalid_ann:
            self.filter_invalid()

        if filter_empty_ann:
            self.filter_empty()

    @property
    def extra_fields(self):
        warnings.warn("'extra_fields' will be replaced by 'ann_fields'.")
        return self.ann_fields

    def filter_crowd(self) -> None:
        ids, img_to_anns = [], []
        for img_id, anns in zip(self.ids, self.img_to_anns):
            anns = list(filter(is_not_crowd, anns))
            ids.append(img_id)
            img_to_anns.append(anns)
        self.ids, self.img_to_anns = ids, img_to_anns

    def filter_invalid(self) -> None:
        ann_types = self.ann_types
        check_bbox = 'bbox' in ann_types
        check_keypoints = 'keypoints' in ann_types
        check_hier = 'hier' in ann_types

        ids, img_to_anns = [], []
        for img_id, anns in zip(self.ids, self.img_to_anns):
            if ((not check_bbox or has_valid_bbox(anns))
                and (not check_keypoints or count_visible_keypoints(anns) >= 10)
                and (not check_hier or has_visible_hier(anns))):  # noqa: E129
                ids.append(img_id)
                img_to_anns.append(anns)
        self.ids, self.img_to_anns = ids, img_to_anns

    def filter_empty(self) -> None:
        ids, img_to_anns = [], []
        for img_id, anns in zip(self.ids, self.img_to_anns):
            if len(anns) > 0 or (len(self.ann_types) == 1 and "semseg" in self.ann_types):
                ids.append(img_id)
                img_to_anns.append(anns)
        self.ids, self.img_to_anns = ids, img_to_anns

    def __getitem__(self, idx: int) -> Tuple[Image.Image, ImageContainer, int]:
        img = self.pull_image(idx)
        anns = self.pull_target(idx)

        size = img.size
        target = ImageContainer(size)

        cat_ids_map = self.category_id_to_contiguous_id
        classes = [cat_ids_map[obj["category_id"]] for obj in anns]
        target['label'] = Label(size, classes)

        if 'bbox' in self.ann_types:
            boxes = [obj["bbox"] for obj in anns]
            boxes = torch.as_tensor(boxes, dtype=torch.float32).view(-1, 4)
            boxes = BoxMode.convert(boxes, BoxMode.XYWH, BoxMode.XYXY)
            target['bbox'] = BoundingBox(size, boxes)

        if 'mask' in self.ann_types:
            masks = [obj["segmentation"] for obj in anns]
            target['mask'] = PolygonList(size, masks)

        if 'keypoints' in self.ann_types:
            keypoints = [obj["keypoints"] for obj in anns]
            keypoints = torch.tensor(keypoints, dtype=torch.float32)
            # COCO's segmentation coordinates are floating points in [0, H or W],
            # but keypoint coordinates are integers in [0, H-1 or W-1]
            # Therefore we assume the coordinates are "pixel indices" and
            # add 0.5 to convert to floating point coordinates.
            if len(keypoints) > 0:
                keypoints[:, 0::3] += 0.5
                keypoints[:, 1::3] += 0.5
            target['keypoints'] = PersonKeypoints(size, keypoints)

        if 'parsing' in self.ann_types:
            parsing_ids = [obj["parsing_id"] for obj in anns]
            parsing = get_parsing(self.root, self.get_img_info(idx)['file_name'], parsing_ids)
            target['parsing'] = Parsing(size, parsing)

        if 'semseg' in self.ann_types:
            target.ignore_length_of('semseg')
            if self.semseg_format == "mask":
                file_name = self.get_img_info(idx)['file_name'].replace(*self.name_trans)
                if 'panoseg' in self.ann_types:
                    panoseg = cv2.imread(os.path.join(self.seg_root, file_name))
                    semseg = convert_pano_to_semseg(panoseg, self.pano_anns, self.ignore_label,
                                                    self.label_format, file_name)
                else:
                    semseg = cv2.imread(os.path.join(self.seg_root, file_name), 0) + self.label_shift
            else:  # "poly"
                thing_segs = [obj["segmentation"] for obj in anns]
                stuff_anns = self.semseg_gt.imgToAnns[self.ids[idx]] if self.semseg_gt else None
                semseg = convert_poly_to_semseg(
                    size, self.label_format, label_shift=self.label_shift,
                    thing_classes=classes, thing_segs=thing_segs,
                    stuff_cat_ids=self.semseg_cat_ids, stuff_anns=stuff_anns,
                    json_category_id_to_contiguous_id=cat_ids_map)
            target['semseg'] = SemanticSegmentation(size, semseg, length=len(target))

        target = target.clip_to_image(threshold=1.0, remove_empty=True)

        # transform
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        # We convert format after transform because polygon format is faster than bitmask
        if 'mask' in self.ann_types and self.ann_fields['mask']['mask_format'] == 'mask':
            target['mask'] = target['mask'].convert(MaskMode.MASK)

        return img, target, idx

    def get_img_info(self, idx: int) -> Dict:
        img_id = self.ids[idx]
        return self.coco.imgs[img_id]

    def pull_image(self, idx: int) -> Image.Image:
        """Returns the original image object at index in PIL form
        """
        img_info = self.get_img_info(idx)
        try:
            file_name = img_info['file_name']
        except:
            file_name = img_info['coco_url'].split('.org/')[-1]  # for lvis

        return Image.open(os.path.join(self.root, file_name)).convert('RGB')

    def pull_target(self, idx: int) -> List[Dict]:
        anns = self.img_to_anns[idx]
        anns = [ann for ann in anns if not ann.get('ignore', False)]
        return anns

    def analyse_dataset(self) -> None:
        cat_ids_map = self.category_id_to_contiguous_id
        dataset_dicts = []
        for anns in self.img_to_anns:
            new_anns = []
            for ann in anns:
                new_ann = {
                    'category_id': cat_ids_map[ann['category_id']],
                    'iscrowd': ann.get('iscrowd', 0),
                }
                new_anns.append(new_ann)
            dataset_dicts.append(new_anns)

        print_instances_class_histogram(dataset_dicts, self.classes)



