import sys

import cv2
import numpy as np
import os, glob
from tqdm import tqdm
import logging


class SemSegEvaluator(object):
    """
    Evaluate semantic segmentation
    """

    def __init__(self, metadata, pred_dir, num_classes, metrics=("mIoU", "IoU", "F1Score")):
        """
        Initialize SemSegEvaluator
        :return: None
        """

        self.pred_dir = pred_dir
        self.gt_dir = metadata.category_gt_root
        self.num_classes = num_classes

        assert metadata.semseg is not None
        self.semseg_format =  metadata.semseg["semseg_format"] \
            if metadata.semseg["semseg_format"] is not None else "mask"
        self.ignore_label = metadata.semseg["ignore_label"] \
            if metadata.semseg["ignore_label"] is not None else 0
        self.label_shift = metadata.semseg["label_shift"] \
            if metadata.semseg["label_shift"] is not None else 1
        self.name_trans = metadata.semseg["name_trans"] \
            if metadata.semseg["name_trans"] is not None else ('jpg', 'png')

        self.metrics = metrics

        self.stats = dict()
        self._logger = logging.getLogger(__name__)

    def fast_hist(self, a, b):
        k = (a >= 0) & (a < self.num_classes)
        return np.bincount(
            self.num_classes * a[k].astype(int) + b[k], minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

    def generate_gt_png(self, image_name):
        if self.semseg_format == "mask":
            gt = cv2.imread(os.path.join(self.gt_dir, image_name), 0) + self.label_shift
            gt_png = gt.astype(np.uint8, copy=False)
        else:
            raise ValueError(f"poly or other unsupported semseg gt format: '{self.semseg_format}'.")

        return gt_png

    def evaluate(self):
        self._logger.info('Evaluating Semantic Segmentation predictions')
        hist = np.zeros((self.num_classes, self.num_classes))

        pred = []
        for ss in os.listdir(os.path.join(self.pred_dir, 'semantic')):
            pred.append({ss: cv2.imread(os.path.join(self.pred_dir, 'semantic', ss), -1)})

        image_names = [x.split("/")[-1] for x in glob.glob(self.gt_dir + '/*') if x[-3:] == 'png']
        self._logger.info('The Global Parsing Images: {}'.format(len(image_names)))

        for image_name in image_names:
            semseg_res = [x for x in pred if image_name in x]
            if len(semseg_res) == 0:
                continue
            pre_png = semseg_res[0][image_name]
            gt_png = self.generate_gt_png(image_name)

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
