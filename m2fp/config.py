# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_m2fp_config(cfg):
    """
    Add config for M2FP.
    """
    # NOTE: configs from original M2FP
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "m2fp_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # m2fp model config
    cfg.MODEL.M2FP = CN()

    # loss
    cfg.MODEL.M2FP.DEEP_SUPERVISION = True
    cfg.MODEL.M2FP.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.M2FP.CLASS_WEIGHT = 1.0
    cfg.MODEL.M2FP.DICE_WEIGHT = 1.0
    cfg.MODEL.M2FP.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.M2FP.NHEADS = 8
    cfg.MODEL.M2FP.DROPOUT = 0.1
    cfg.MODEL.M2FP.DIM_FEEDFORWARD = 2048
    cfg.MODEL.M2FP.ENC_LAYERS = 0
    cfg.MODEL.M2FP.DEC_LAYERS = 6
    cfg.MODEL.M2FP.PRE_NORM = False

    cfg.MODEL.M2FP.HIDDEN_DIM = 256
    cfg.MODEL.M2FP.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.M2FP.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.M2FP.ENFORCE_INPUT_PROJ = False

    cfg.MODEL.M2FP.WITH_HUMAN_INSTANCE = True

    # M2FP inference config
    cfg.MODEL.M2FP.TEST = CN()
    cfg.MODEL.M2FP.TEST.SEMANTIC_ON = True
    cfg.MODEL.M2FP.TEST.PARSING_ON = False
    cfg.MODEL.M2FP.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.M2FP.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.M2FP.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False
    cfg.MODEL.M2FP.TEST.PARSING_INS_SCORE_THR = 0.5
    cfg.MODEL.M2FP.TEST.METRICS = ("mIoU", "APr", "APp")

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.M2FP.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # transformer module
    cfg.MODEL.M2FP.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0
    cfg.INPUT.ROTATION = 0

    # Single human parsing aug
    cfg.INPUT.SINGLE_HUMAN = CN()
    cfg.INPUT.SINGLE_HUMAN.ENABLED = False
    cfg.INPUT.SINGLE_HUMAN.SIZES = ([384, 512],)
    cfg.INPUT.SINGLE_HUMAN.SCALE_FACTOR = 0.8
    cfg.INPUT.SINGLE_HUMAN.ROTATION = 40
    cfg.INPUT.SINGLE_HUMAN.COLOR_AUG_SSD = False
    cfg.INPUT.SINGLE_HUMAN.TEST_SCALES = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.M2FP.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.M2FP.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.M2FP.IMPORTANCE_SAMPLE_RATIO = 0.75

    # WandB
    cfg.WANDB = CN({"ENABLED": False})
    cfg.WANDB.ENTITY = ""
    cfg.WANDB.NAME = ""
    cfg.WANDB.PROJECT = "M2FP"
