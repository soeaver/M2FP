# -*- coding: UTF-8 -*-

import os
import copy
import itertools
import json
import logging
from collections import OrderedDict

import cv2
import numpy as np
from scipy.sparse import csr_matrix

import pycocotools.mask as mask_util

import torch
from torchvision.datasets.coco import CocoDetection

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator

from .parsing_eval import ParsingEval


class ParsingEvaluator(DatasetEvaluator):
    def __init__(
            self,
            dataset_name,
            distributed=True,
            output_dir=None,
            *,
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

        self.metrics = parsing_metrics

        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)

        if output_dir is None:
            raise ValueError("output_dir must be provided to ParsingEvaluator.")

        self.dataset = CocoDetection(self._metadata.image_root, self._metadata.json_file)

        self._do_evaluation = True

    def reset(self):
        self._parsing_predictions = []


    def process(self, inputs, outputs):

        output_dict = outputs[-1]['parsing']

        # save results to png
        output_root = self._output_dir

        # semseg_predictions
        semseg_path = os.path.join(output_root, 'semantic')
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

    def _eval_parsing_predictions(self, pars_predictions):
        """
        Evaluate parsing predictions. Fill self._results with the metrics of the tasks.
        """
        parsing_results = pars_predictions

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        _evaluate_predictions_for_parsing(
            self.dataset,
            parsing_results,
            self._metadata,
            self._output_dir,
            self.metrics,
        )


def _evaluate_predictions_for_parsing(
        dataset,
        parsing_results,
        metadata,
        output_folder,
        metrics,
):
    """
    Evaluate the parsing results using ParsingEval API.
    """
    model_parsing_score_threse = 0.01

    parsing_eval = ParsingEval(
        dataset,
        parsing_results,
        metadata,
        output_folder,
        model_parsing_score_threse,
        metrics=metrics
    )
    parsing_eval.evaluate()
    parsing_eval.accumulate()
    parsing_eval.summarize()
