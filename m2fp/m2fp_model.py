# Copyright (c) Facebook, Inc. and its affiliates.
import os.path
import sys
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher

import copy, cv2
import numpy as np
from .modeling.postprocessing import single_human_sem_seg_postprocess


@META_ARCH_REGISTRY.register()
class M2FP(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
            self,
            *,
            backbone: Backbone,
            sem_seg_head: nn.Module,
            criterion: nn.Module,
            num_queries: int,
            object_mask_threshold: float,
            overlap_threshold: float,
            metadata,
            size_divisibility: int,
            sem_seg_postprocess_before_inference: bool,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            # inference
            semantic_on: bool,
            parsing_on: bool,
            test_topk_per_image: int,
            single_human: bool,
            with_human_instance: bool,
            parsing_ins_score_thr: float
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.test_topk_per_image = test_topk_per_image
        self.single_human = single_human

        # parsing inference
        self.parsing_on = parsing_on
        self.with_human_instance = with_human_instance
        self.parsing_ins_score_thr = parsing_ins_score_thr

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.M2FP.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.M2FP.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.M2FP.CLASS_WEIGHT
        dice_weight = cfg.MODEL.M2FP.DICE_WEIGHT
        mask_weight = cfg.MODEL.M2FP.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.M2FP.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.M2FP.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.M2FP.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.M2FP.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.M2FP.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.M2FP.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.M2FP.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.M2FP.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.M2FP.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                    cfg.MODEL.M2FP.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                    or cfg.MODEL.M2FP.TEST.PARSING_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.M2FP.TEST.SEMANTIC_ON,
            "parsing_on": cfg.MODEL.M2FP.TEST.PARSING_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "single_human": True if "lip" in cfg.DATASETS.TEST[0] else False,
            "with_human_instance": cfg.MODEL.M2FP.WITH_HUMAN_INSTANCE,
            "parsing_ins_score_thr": cfg.MODEL.M2FP.TEST.PARSING_INS_SCORE_THR
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            mask_cls_results = outputs["pred_logits"]  # (B, Q, C+1)
            mask_pred_results = outputs["pred_masks"]  # (B, Q, H, W)
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                    mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})  # for each image

                if self.sem_seg_postprocess_before_inference:
                    if not self.single_human:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                    else:
                        mask_pred_result = retry_if_cuda_oom(single_human_sem_seg_postprocess)(
                            mask_pred_result, image_size, input_per_image["crop_box"], height, width
                        )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        if not self.single_human:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                        else:
                            r = retry_if_cuda_oom(single_human_sem_seg_postprocess)(
                                r, image_size, input_per_image["crop_box"], height, width
                            )
                        processed_results[-1]["sem_seg"] = r

                # parsing inference
                if self.parsing_on:
                    parsing_r = retry_if_cuda_oom(self.instance_parsing_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["parsing"] = parsing_r

            return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def instance_parsing_inference(self, mask_cls, mask_pred):
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(
            self.sem_seg_head.num_classes, device=self.device
        ).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)

        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        mask_pred = mask_pred[topk_indices]

        binary_pred_masks = (mask_pred > 0).float()
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * binary_pred_masks.flatten(1)).sum(1) / \
                                (binary_pred_masks.flatten(1).sum(1) + 1e-6)

        pred_scores = scores_per_image * mask_scores_per_image
        pred_labels = labels_per_image
        pred_masks = mask_pred

        # prepare outputs
        part_instance_res = []
        human_instance_res = []

        # bkg and part instances
        bkg_part_index = torch.where(pred_labels != self.metadata.num_parsing)[0]
        bkg_part_labels = pred_labels[bkg_part_index]
        bkg_part_scores = pred_scores[bkg_part_index]
        bkg_part_masks = pred_masks[bkg_part_index, :, :]

        # human instances
        human_index = torch.where(pred_labels == self.metadata.num_parsing)[0]
        human_labels = pred_labels[human_index]
        human_scores = pred_scores[human_index]
        human_masks = pred_masks[human_index, :, :]

        semantic_res = self.paste_instance_to_semseg_probs(bkg_part_labels, bkg_part_scores, bkg_part_masks)

        # part instances
        part_index = torch.where(bkg_part_labels != 0)[0]
        part_labels = bkg_part_labels[part_index]
        part_scores = bkg_part_scores[part_index]
        part_masks = bkg_part_masks[part_index, :, :]

        # part instance results
        for idx in range(part_labels.shape[0]):
            if part_scores[idx] < 0.1:
                continue
            part_instance_res.append(
                {
                    "category_id": part_labels[idx].cpu().tolist(),
                    "score": part_scores[idx].cpu().tolist(),
                    "mask": part_masks[idx],
                }
            )

        # human instance results
        for human_idx in range(human_scores.shape[0]):
            if human_scores[human_idx] > 0.1:
                human_instance_res.append(
                    {
                        "category_id": human_labels[human_idx].cpu().tolist(),
                        "score": human_scores[human_idx].cpu().tolist(),
                        "mask": human_masks[human_idx],
                    }
                )

        return {
            "semantic_outputs": semantic_res,
            "part_outputs": part_instance_res,
            "human_outputs": human_instance_res,
        }

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]  # discard non-sense category
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def paste_instance_to_semseg_probs(self, labels, scores, mask_probs):
        im_h, im_w = mask_probs.shape[-2:]
        semseg_im = []
        for cls_ind in range(self.metadata.num_parsing):
            cate_inds = torch.where(labels == cls_ind)[0]
            cate_scores = scores[cate_inds]
            cate_mask_probs = mask_probs[cate_inds, :, :].sigmoid()
            semseg_im.append(self.paste_category_probs(cate_scores, cate_mask_probs, im_h, im_w))

        return torch.stack(semseg_im, dim=0)

    def paste_category_probs(self, scores, mask_probs, h, w):
        category_probs = torch.zeros((h, w), dtype=torch.float32, device=mask_probs.device)
        paste_times = torch.zeros((h, w), dtype=torch.float32, device=mask_probs.device)

        index = scores.argsort()
        for k in range(len(index)):
            if scores[index[k]] < self.parsing_ins_score_thr:
                continue
            ins_mask_probs = mask_probs[index[k], :, :] * scores[index[k]]
            category_probs = torch.where(ins_mask_probs > 0.5, ins_mask_probs + category_probs, category_probs)
            paste_times += torch.where(ins_mask_probs > 0.5, 1, 0)

        paste_times = torch.where(paste_times == 0, paste_times + 1, paste_times)
        category_probs /= paste_times

        return category_probs
