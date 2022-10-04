# -*- coding: UTF-8 -*-

import os
import copy
import itertools
import json
import logging
import sys
import time
from collections import OrderedDict

import cv2
import numpy as np
from scipy.sparse import csr_matrix

import pycocotools.mask as mask_util

import torch
import torch.nn.functional as F
from torchvision.datasets.coco import CocoDetection

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator

from .parsing_eval import ParsingEval


class ParsingEvaluator(DatasetEvaluator):
    def __init__(
            self,
            dataset_name,
            ins_score_thr,
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
        self.ins_score_thr = ins_score_thr

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

        #################### prepare prediction outputs ####################
        # prepare category labels and probs
        semantic_probs = output_dict["semantic_outputs"]
        global_category_labels = copy.deepcopy(semantic_probs).argmax(dim=0)

        semseg_path = os.path.join(output_root, "semantic")
        os.makedirs(semseg_path, exist_ok=True)
        part_path = os.path.join(output_root, "part")
        os.makedirs(part_path, exist_ok=True)
        part_png = torch.zeros(image_shape, dtype=torch.int64, device=semantic_probs.device)
        part_info = []
        human_path = os.path.join(output_root, "human")
        os.makedirs(human_path, exist_ok=True)
        human_png = torch.zeros(image_shape, dtype=torch.int64, device=semantic_probs.device)
        human_info = []

        # prepare human ids
        human_ins_preds = output_dict["human_outputs"]
        human_ids_map = torch.zeros(image_shape, dtype=torch.int64, device=semantic_probs.device)
        human_scores = []
        sorted_human_idx = np.array([_s["score"] for _s in human_ins_preds]).argsort().tolist()
        human_ins_id = 1
        for human_idx in sorted_human_idx:
            human_output = output_dict["human_outputs"][human_idx]
            if human_output["score"] < 0.:
                continue
            human_ids_map = torch.where(human_output["mask"] > 0, human_ins_id, human_ids_map)
            human_scores.append(human_output["score"])
            human_ins_id += 1
        human_ids = torch.unique(human_ids_map)
        human_id_index = torch.where(human_ids != 0)[0]
        human_ids = human_ids[human_id_index]

        # prepare part ins scores map
        part_ins_preds = prepare_part_instances(copy.deepcopy(output_dict['part_outputs']))
        ins_scores_map = [torch.zeros(image_shape, dtype=torch.float32, device=semantic_probs.device)]
        for cls_ind in range(1, self._metadata.num_parsing):
            ins_scores_cate = torch.zeros(image_shape, dtype=torch.float32, device=semantic_probs.device)
            try:
                scores_cate = part_ins_preds[cls_ind]["scores"]
                mask_probs_cate = part_ins_preds[cls_ind]["masks"]
            except:
                scores_cate, mask_probs_cate = [], []
            if len(scores_cate) > 0:
                scores_cate = torch.tensor(scores_cate, dtype=torch.float32, device=semantic_probs.device)
                mask_probs_cate = torch.stack(mask_probs_cate, dim=0).sigmoid()
                index = scores_cate.argsort()
                for k in range(len(index)):
                    if scores_cate[index[k]] < self.ins_score_thr:
                        continue
                    ins_mask_probs = mask_probs_cate[index[k], :, :] * scores_cate[index[k]]
                    ins_scores_cate = torch.where(
                        ins_mask_probs > 0.5, torch.max(scores_cate[index[k]], ins_scores_cate), ins_scores_cate
                    )
            ins_scores_map.append(ins_scores_cate)
        global_ins_scores_map = torch.stack(ins_scores_map, dim=0)

        #################### calculate prediction outputs ####################
        # semantic prediction
        semseg_png = copy.deepcopy(semantic_probs).argmax(dim=0).cpu().numpy()

        # part predictions
        sorted_part_id = np.array([_s["score"] for _s in output_dict["part_outputs"]]).argsort()
        for _num_part, part_id in enumerate(sorted_part_id):
            part_pred = output_dict["part_outputs"][part_id]
            if _num_part >= 255:
                part_png = torch.where(part_pred["mask"] > 0, 0, part_png)
            else:
                part_png = torch.where(part_pred["mask"] > 0, _num_part + 1, part_png)
            part_info_tmp = {
                "img_name": image_name,
                "category_id": part_pred["category_id"],
                "score": part_pred["score"],
                "part_id": int(_num_part + 1),
            }
            part_info.append(part_info_tmp)
        part_png = part_png.cpu().numpy().astype(np.uint8)
        part_info = filter_out_covered_ins_info(part_info, part_png)

        # human & parsing prediction
        total_human_num = 1
        for human_id in human_ids:
            # human prediction
            human_score = human_scores[human_id - 1]
            human_png = torch.where(human_ids_map == human_id, total_human_num, human_png)
            human_info.append(
                {
                    "img_name": image_name,
                    "score": human_score,
                    "human_id": total_human_num
                }
            )
            total_human_num += 1

            # parsing prediction
            human_part_label = (torch.where(human_ids_map == human_id, 1, 0) * global_category_labels)
            human_ins_scores_map = (torch.where(human_ids_map == human_id, 1, 0) * global_ins_scores_map)
            human_part_classes = torch.unique(human_part_label)
            human_ins_scores_map_selected = human_ins_scores_map[human_part_classes, :, :]  # select channels by labels
            human_part_ins_scores = torch.max(human_ins_scores_map_selected.flatten(1, 2), dim=1)[0]
            valid_parts_num = len(torch.where(human_part_ins_scores != 0)[0])
            parsing_score = (torch.sum(human_part_ins_scores) * human_score) / valid_parts_num

            self._parsing_predictions.append(
                {
                    "img_name": image_name,
                    "parsing": csr_matrix(human_part_label.cpu().numpy()),
                    "score": parsing_score.cpu(),
                }
            )
        human_png = human_png.cpu().numpy().astype(np.uint8)

        #################### save prediction outputs ####################
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


def prepare_part_instances(part_ins_preds):
    part_preds_classes = {}
    for part_ins_pred in part_ins_preds:
        if part_ins_pred["category_id"] not in part_preds_classes:
            part_preds_classes[part_ins_pred["category_id"]] = {
                "scores": [part_ins_pred["score"]], "masks": [part_ins_pred["mask"]]
            }
        else:
            part_preds_classes[part_ins_pred["category_id"]]["scores"].append(part_ins_pred["score"])
            part_preds_classes[part_ins_pred["category_id"]]["masks"].append(part_ins_pred["mask"])
    return part_preds_classes


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
