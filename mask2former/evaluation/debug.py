# -*- coding: UTF-8 -*-

import os
import copy, glob
import json

import torch
from torchvision.datasets.coco import CocoDetection



image_root = "/home/user/Program/m2f/M2FP/datasets/pascal-person-part/Testing/Images"
json_file = "/home/user/Program/m2f/M2FP/datasets/pascal-person-part/annotations/PASCAL-Person-Part_imgIds_test.json"

dataset = CocoDetection(image_root, json_file)


image_ids = dataset.coco.getImgIds()
image_ids.sort()

for image_id in image_ids:
    file_name = dataset.coco.loadImgs(image_id)[0]['file_name']
    # print(image_id, ' | ', file_name)


# ids_src = json.load(open(json_file, 'r'))
# print(ids_src['images'])

gt_root = '/home/user/Program/m2f/M2FP/datasets/pascal-person-part/Testing/Human_ids'

img_names = [x.split("/")[-1].split(".")[0] for x in glob.glob(gt_root + '/*') if x[-3:] == 'png']
print(img_names)
