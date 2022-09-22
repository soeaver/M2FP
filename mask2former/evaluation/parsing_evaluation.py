# -*- coding: UTF-8 -*-

import os
import copy
import itertools
import json
import logging
import sys
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
            parsing_metrics=("mIoU", "APp", "APr", "APh"),
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

        self._do_evaluation = True

    def reset(self):
        self._parsing_predictions = []

    def process(self, inputs, outputs):

        output_dict = outputs[-1]["parsing"]

        image_name = inputs[0]["file_name"].split("/")[-1].split(".")[0]
        output_root = self._output_dir
        image_shape = (inputs[0]["height"], inputs[0]["width"])

        # prepare semantic, part and human prediction
        semseg_path = os.path.join(output_root, "semantic")
        os.makedirs(semseg_path, exist_ok=True)

        part_path = os.path.join(output_root, "part")
        os.makedirs(part_path, exist_ok=True)
        part_png = np.zeros(image_shape, dtype=np.uint8)
        part_info = []

        human_path = os.path.join(output_root, "human")
        os.makedirs(human_path, exist_ok=True)
        human_png = np.zeros(image_shape, dtype=np.uint8)
        human_info = []

        # prepare category labels and probs
        semseg_probs = output_dict["semseg_outputs"]
        global_ins_scores_map = output_dict["ins_scores_map"].cpu().numpy()
        global_category_labels = copy.deepcopy(semseg_probs).argmax(dim=0).cpu().numpy()

        human_ins_preds = output_dict["human_outputs"]

        # prepare human ids
        human_ids_map = np.zeros(image_shape, dtype=np.uint8)
        human_scores = []
        sorted_human_idx = np.array([_s["score"] for _s in human_ins_preds]).argsort().tolist()
        human_ins_id = 1
        for human_idx in sorted_human_idx:
            human_output = output_dict["human_outputs"][human_idx]
            if human_output["score"] < 0.:
                continue
            human_ids_map = np.where(human_output["mask"].numpy() > 0, human_ins_id, human_ids_map)
            human_scores.append(human_output["score"])
            human_ins_id += 1

        human_ids = np.unique(human_ids_map)
        bg_id_index = np.where(human_ids == 0)[0]
        human_ids = np.delete(human_ids, bg_id_index)

        # calculate prediction
        # semantic prediction
        semseg_png = copy.deepcopy(semseg_probs).argmax(dim=0).cpu().numpy()

        # part predictions
        sorted_part_id = np.array([_s['score'] for _s in output_dict['part_outputs']]).argsort()
        for _num_part, part_id in enumerate(sorted_part_id):
            part_output = output_dict['part_outputs'][part_id]
            # reserve id 0 for background
            if _num_part >= 255:
                part_png = np.where(part_output['mask'] > 0, 0, part_png)
            else:
                part_png = np.where(part_output['mask'] > 0, _num_part + 1, part_png)
            part_info_tmp = {
                'img_name': image_name,
                'category_id': part_output["category_id"],
                'score': part_output["score"],
                'part_id': int(_num_part + 1),
            }
            part_info.append(part_info_tmp)

        part_info = filter_out_covered_ins_info(part_info, part_png)

        # human & parsing prediction
        total_human_num = 1
        for human_id in human_ids:
            # human prediction
            human_score = human_scores[human_id - 1]
            human_png = np.where(human_ids_map == human_id, total_human_num, human_png)

            human_info.append(
                {
                    "img_name": image_name,
                    "score": human_score,
                    "human_id": total_human_num
                }
            )
            total_human_num += 1

            # parsing prediction
            human_ins_scores_map = (np.where(human_ids_map == human_id, 1, 0) * global_ins_scores_map)
            human_part_label = (np.where(human_ids_map == human_id, 1, 0) * global_category_labels).astype(np.uint8)
            human_part_classes = np.unique(human_part_label)

            _scores_for_parsing = []
            for part_id in human_part_classes:
                # part ins scores
                part_ins_scores_map = human_ins_scores_map[part_id, :, :]
                part_ins_scores = np.unique(part_ins_scores_map)
                bg_id_index = np.where(part_ins_scores == 0)[0]
                part_ins_scores = np.delete(part_ins_scores, bg_id_index)

                if part_ins_scores.shape[0] == 0:
                    part_score = 0
                else:
                    part_score = np.max(part_ins_scores)

                if part_score > 0:
                    _scores_for_parsing.append(part_score)

            mean_part_score = np.mean(np.asarray(_scores_for_parsing))
            parsing_score = mean_part_score * human_score

            self._parsing_predictions.append(
                {
                    "img_name": image_name,
                    "parsing": csr_matrix(human_part_label),
                    "score": parsing_score
                }
            )

        # save semantic, part and human prediction
        cv2.imwrite(os.path.join(semseg_path, image_name + ".png"), semseg_png)

        cv2.imwrite(os.path.join(part_path, "{}.png".format(image_name)), part_png)
        json.dump(part_info, open(os.path.join(part_path, "{}.json".format(image_name)), "w"))

        cv2.imwrite(os.path.join(human_path, "{}.png".format(image_name)), human_png)
        json.dump(human_info, open(os.path.join(human_path, "{}.json".format(image_name)), "w"))

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

        eval_res = self._eval_parsing_predictions(parsing_prediction)

        if "mIoU" in self.metrics:
            self._results["mIoU"] = eval_res["mIoU"]
        if "APr" in self.metrics:
            self._results["APr"]  = np.nanmean(np.array(list(eval_res["APr"].values())))
        if "APh" in self.metrics:
            self._results["APh"]  = np.nanmean(np.array(list(eval_res["APh"].values())))
        if "APp" in self.metrics:
            self._results["APp"]  = np.nanmean(np.array(list(eval_res["APp"].values())))

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_parsing_predictions(self, pars_predictions):
        """
        Evaluate parsing predictions. Fill self._results with the metrics of the tasks.
        """
        parsing_results = pars_predictions

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return {}

        return _evaluate_predictions_for_parsing(
            parsing_results,
            self._metadata,
            self._output_dir,
            self.metrics,
        )


def _evaluate_predictions_for_parsing(
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
        parsing_results,
        metadata,
        output_folder,
        model_parsing_score_threse,
        metrics=metrics
    )
    eval_res = parsing_eval.evaluate()
    parsing_eval.accumulate()
    parsing_eval.summarize()

    return eval_res


def filter_out_covered_ins_info(info_list, map):
    filtered_info_list = []
    for info in info_list:
        idx = info["part_id"]

        mask = np.where(map == idx, 1, 0)
        if np.max(mask) == 0:
            continue
        else:
            filtered_info_list.append(info)
    return filtered_info_list
