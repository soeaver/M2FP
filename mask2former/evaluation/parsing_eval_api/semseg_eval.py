import sys

import cv2
import dpath.util
import numpy as np
import os
from tqdm import tqdm
import logging

from .utils import convert_pano_to_semseg, convert_poly_to_semseg

from detectron2.utils.logger import create_small_table


class SemSegEvaluator(object):
    """
    Evaluate semantic segmentation
    """

    def __init__(self, dataset, preds, root_dir, num_classes, gt_dir=None, metrics=("mIoU", "IoU", "F1Score")):
        """
        Initialize SemSegEvaluator
        :return: None
        """

        self.preds = preds
        # self.pre_dir = pre_dir
        self.dataset = dataset
        self.num_classes = num_classes
        ann_fields = dataset.ann_fields
        self.semseg_format = dpath.util.get(ann_fields, "/semseg/semseg_format", default="mask")
        self.ignore_label = dpath.util.get(ann_fields, "/semseg/ignore_label", default=0)
        self.label_shift = dpath.util.get(ann_fields, "/semseg/label_shift", default=1)
        self.name_trans = dpath.util.get(ann_fields, "/semseg/name_trans", default=('jpg', 'png'))
        if "panoseg" in dataset.ann_types:
            self.pano_anns = dataset.pano_anns
        self.ids = dataset.ids
        self.metrics = metrics

        if gt_dir is not None:
            self.gt_dir = gt_dir
        else:
            self.gt_dir = dpath.util.get(ann_fields, "/semseg/seg_root", default=root_dir.replace('img', 'seg'))
        self.stats = dict()
        self._logger = logging.getLogger(__name__)

    def fast_hist(self, a, b):
        # print('gt & pre shape: ', a.shape, b.shape)
        k = (a >= 0) & (a < self.num_classes)
        return np.bincount(
            self.num_classes * a[k].astype(int) + b[k], minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

    def generate_gt_png(self, i, image_name, size):
        # print(self.dataset.ann_types, self.semseg_format, '\n+\n\n\n')
        if "panoseg" not in self.dataset.ann_types:
            if self.semseg_format == "mask":
                gt = cv2.imread(os.path.join(self.gt_dir, image_name), 0) + self.label_shift
                gt_png = gt.astype(np.uint8, copy=False)
            elif self.semseg_format == "poly":
                anns = self.dataset.coco.imgToAnns[i]
                cat_ids_map = self.dataset.category_id_to_contiguous_id
                classes = [cat_ids_map[obj["category_id"]] for obj in anns]
                stuff_anns = self.dataset.semseg_gt.imgToAnns[i] if self.dataset.semseg_gt else None
                thing_segs = [obj["segmentation"] for obj in anns]
                gt_png = convert_poly_to_semseg(
                    (size[1], size[0]), self.dataset.label_format, label_shift=1,
                    thing_classes=classes, thing_segs=thing_segs,
                    stuff_cat_ids=self.dataset.semseg_cat_ids, stuff_anns=stuff_anns,
                    json_category_id_to_contiguous_id=cat_ids_map)
            else:
                raise ValueError(f"Unsupported semseg gt format: '{self.semseg_format}'.")
        else:
            semseg = cv2.imread(os.path.join(self.gt_dir, image_name))
            gt_png = convert_pano_to_semseg(semseg, self.pano_anns, self.ignore_label,
                                            self.dataset.label_format, image_name)

        return gt_png

    def evaluate(self):
        self._logger.info('Evaluating Semantic Segmentation predictions')
        hist = np.zeros((self.num_classes, self.num_classes))
        pred = []
        for ss in os.listdir(os.path.join(self.preds, 'semseg')):
            pred.append({ss: cv2.imread(os.path.join(self.preds,'semseg',ss),-1)})
            # print(pred)


        for i in tqdm(self.ids, desc='Calculating IoU ..'):
            image_name = self.dataset.coco.imgs[i]['file_name'].replace(*self.name_trans)
            semseg_res = [x for x in pred if image_name in x]
            if len(semseg_res) == 0:
                continue
            pre_png = semseg_res[0][image_name]
            gt_png = self.generate_gt_png(i, image_name, pre_png.shape)

            assert gt_png.shape == pre_png.shape, '{} VS {}'.format(str(gt_png.shape), str(pre_png.shape))
            gt = gt_png.flatten()
            pre = pre_png.flatten()
            hist += self.fast_hist(gt, pre)

        def mean_iou(overall_h):
            iu = np.diag(overall_h) / (overall_h.sum(1) + overall_h.sum(0) - np.diag(overall_h) + 1e-10)
            return iu, np.nanmean(iu)

        def mean_f1score(overall_h):
            precision = np.diag(overall_h) / (overall_h.sum(1) + 1e-10)
            recall = np.diag(overall_h) / (overall_h.sum(0) + 1e-10)
            f1score = 2 * precision * recall / (precision + recall + 1e-10)
            return f1score, np.nanmean(f1score)

        def per_class_acc(overall_h):
            acc = np.diag(overall_h) / (overall_h.sum(1) + 1e-10)
            return np.nanmean(acc)

        def pixel_wise_acc(overall_h):
            return np.diag(overall_h).sum() / overall_h.sum()

        iou, miou = mean_iou(hist)
        mean_acc = per_class_acc(hist)
        pixel_acc = pixel_wise_acc(hist)
        f1_score, mean_f1score = mean_f1score(hist)
        self.stats.update(dict(IoU=iou, mIoU=miou, MeanACC=mean_acc, PixelACC=pixel_acc,
                               F1Score=f1_score, MeanF1Score=mean_f1score))

    def accumulate(self, p=None):
        pass

    def summarize(self):
        iStr = ' {:<18} @[area={:>6s}] = {:0.4f}'
        if "F1Score" in self.metrics:
            for i, score in enumerate(self.stats['F1Score']):
                print('Class {}   (F1 Score) = {:.3f}'.format(i + 1, score))
            print('=' * 80)

        if "IoU" in self.metrics:
            for i, iou in enumerate(self.stats['IoU']):
                print('Class {}        (IoU) = {:.3f}'.format(i + 1, iou))
            print('=' * 80)

        if "mIoU" in self.metrics:
            for k, v in self.stats.items():
                if k == 'IoU' or k == 'F1Score':
                    continue
                self._logger.info(iStr.format(k, 'all', v))

    def __str__(self):
        self.summarize()


def semseg_png(score, dataset=None, img_info=None, output_folder=None, semseg=None, target=None):
    semseg_pres_dir = os.path.join(output_folder, 'semseg_pres')
    if not os.path.exists(semseg_pres_dir):
        os.makedirs(semseg_pres_dir)

    try:
        file_name = img_info['file_name']
    except:
        file_name = img_info['coco_url'].split('.org/')[-1]  # for lvis = img_info['file_name']
    extra_fields = dataset.extra_fields
    name_trans = extra_fields['name_trans'] if 'name_trans' in extra_fields else ['jpg', 'png']
    save_semseg_pres = os.path.join(semseg_pres_dir, file_name.replace(name_trans[0], name_trans[1]))
    cv2.imwrite(save_semseg_pres, score.astype(np.uint8))

    if target is not None:
        semseg_gt_dir = os.path.join(output_folder, 'semseg_gt')
        label = target['semseg'].tensor.squeeze(0).numpy()
        if not os.path.exists(semseg_gt_dir):
            os.makedirs(semseg_gt_dir)
        save_semseg_gt = os.path.join(semseg_gt_dir, file_name.replace(name_trans[0], name_trans[1]))
        cv2.imwrite(save_semseg_gt, label.astype(np.uint8))
