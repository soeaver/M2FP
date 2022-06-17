# -*- coding: UTF-8 -*-

import contextlib
import copy
import io
import itertools
import json
import logging
import shutil
import sys

import cv2
import numpy as np
import os
import pickle
from scipy.sparse import csr_matrix
from collections import OrderedDict, defaultdict
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table

from detectron2.evaluation.evaluator import DatasetEvaluator

try:
    from detectron2.evaluation.fast_eval_api import COCOeval_opt
except ImportError:
    COCOeval_opt = COCOeval

from .parsing_eval_api import ParsingEval
from .parsing_eval_api import ParsingGT
from .parsing_eval_api import get_ann_fields


class ParsingEvaluator(DatasetEvaluator):
    def __init__(
            self,
            dataset_name,
            tasks='parsing',
            distributed=True,
            output_dir=None,
            *,
            max_dets_per_image=None,
            parsing_metrics=('mIoU', 'APp', 'APr', 'APh'),
    ):
        """

        :param dataset_name:
        :param tasks:
        :param distributed:
        :param output_dir:
        :param max_dets_per_image:
        :param parsing_metrics:
        """
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir

        if max_dets_per_image is None:
            max_dets_per_image = [1, 10, 100]
        else:
            max_dets_per_image = [1, 10, max_dets_per_image]

        self._max_dets_per_image = max_dets_per_image
        self._tasks = tasks
        self.metrics = parsing_metrics

        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)

        if not hasattr(self._metadata, "json_file"):
            if output_dir is None:
                raise ValueError(
                    "output_dir must be provided to COCOEvaluator "
                    "for datasets not in COCO format."
                )
            self._logger.info(f"Trying to convert '{dataset_name}' to COCO format ...")

            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)
            ann_fields = defaultdict(dict, get_ann_fields(dataset_name))
            ann_fields['parsing'].update({'semseg_format': "mask"})
            ann_fields = dict(ann_fields)
            self.parsing_GT = ParsingGT(self._metadata.image_root, self._metadata.json_file, set('parsing'), ann_fields)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = True  # "annotations" in self._coco_api.dataset

    def reset(self):
        self._parsing_predictions = []


    def process(self, inputs, outputs):

        output_dict = outputs[-1]['parsing']

        # save results to png
        output_root = self._output_dir

        # semseg_predictions
        semseg_path = os.path.join(output_root, 'semseg')
        os.makedirs(semseg_path, exist_ok=True)

        semseg_img = output_dict["semseg_outputs"].argmax(dim=0).cpu().numpy()
        semseg_name = os.path.join(semseg_path, os.path.splitext(inputs[0]["file_name"])[0].split('/')[-1] + '.png')
        cv2.imwrite(semseg_name, semseg_img)

        # part predictions
        part_path = os.path.join(output_root, 'part')
        os.makedirs(part_path, exist_ok=True)
        part_list = []

        for part_output in output_dict["part_outputs"]:
            part_prediction = {"image_id": inputs[0]["image_id"]}
            part_prediction["img_name"] = inputs[0]["file_name"].split('/')[-1].split('.')[0]
            part_prediction["category_id"] = part_output["category_id"]
            part_prediction["score"] = part_output["score"]
            _rle = mask_util.encode(np.array((part_output["mask"] > 0)[:, :, None], order="F", dtype="uint8"))[0]
            _rle["counts"] = _rle["counts"].decode("utf-8")
            part_prediction['mask'] = _rle
            if len(part_prediction) > 1:
                part_list.append(part_prediction)

        part_name = os.path.join(part_path, '{}.json'.format(inputs[0]["image_id"]))
        json.dump(part_list, open(part_name, 'w'))

        # human predictions
        human_path = os.path.join(output_root, 'human')
        os.makedirs(human_path, exist_ok=True)
        human_list = []

        for human_output in output_dict["human_outputs"]:
            human_prediction = {"image_id": inputs[0]["image_id"]}
            human_prediction["img_name"] = inputs[0]["file_name"].split('/')[-1].split('.')[0]
            human_prediction["category_id"] = human_output["category_id"]
            human_prediction["score"] = human_output["score"]
            _rle2 = mask_util.encode(np.array((human_output["mask"] > 0)[:, :, None], order="F", dtype="uint8"))[0]
            _rle2["counts"] = _rle2["counts"].decode("utf-8")
            human_prediction["mask"] = _rle2
            if len(human_prediction) > 1:
                human_list.append(human_prediction)

        human_name = os.path.join(human_path, '{}.json'.format(inputs[0]["image_id"]))
        json.dump(human_list, open(human_name, 'w'))

        for parsing_output in output_dict["parsing_outputs"]:
            parsing_prediction = {"image_id": inputs[0]["image_id"]}
            parsing_prediction['parsing'] = csr_matrix(parsing_output["parsing"].numpy())
            parsing_prediction['score'] = parsing_output["instance_score"]
            if len(parsing_prediction) > 1:
                self._parsing_predictions.append(parsing_prediction)

    def evaluate(self):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """

        if self._distributed:
            self._logger.info("gathering results from all devices....")
            comm.synchronize()

            parsing_prediction = comm.gather(self._parsing_predictions, dst=0)
            parsing_prediction = list(itertools.chain(*parsing_prediction))

            if not comm.is_main_process():
                return {}
        else:
            self._logger.info("gathering results from single devices....")

            parsing_prediction = self._parsing_predictions

        self._logger.info("gather results from all devices done")

        if len(parsing_prediction) == 0:
            self._logger.warning("[ParsingEvaluator] Did not receive valid parsing predictions.")

        self._results = OrderedDict()

        self._eval_parsing_predictions(parsing_prediction)

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_parsing_predictions(self, pars_predictions, img_ids=None):
        """
        Evaluate parsing predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")

        parsing_results = pars_predictions

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        _evaluate_predictions_on_parsing(
            self.parsing_GT,
            parsing_results,
            self._metadata,
            self._output_dir,
            self.metrics,
        )


def _evaluate_predictions_on_parsing(
        parsing_gt,
        parsing_results,
        metadata,
        output_folder,
        metrics,
):
    """
    Evaluate the parsing results using ParsingEval API.
    """
    model_parsing_score_threse = 0.01
    model_parsing_num_parsing = metadata.num_parsing

    pet_eval = ParsingEval(
        parsing_gt,
        parsing_results,
        metadata.image_root, output_folder,
        model_parsing_score_threse,
        model_parsing_num_parsing,
        metrics=metrics
    )
    pet_eval.evaluate()
    pet_eval.accumulate()
    pet_eval.summarize()
