# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config

# dataset loading
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import COCOPanopticNewBaselineDatasetMapper
from .data.dataset_mappers.mask_former_instance_dataset_mapper import (
    MaskFormerInstanceDatasetMapper,
)
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper,
)
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)

from .data.dataset_mappers.mask_former_semantic_hp_dataset_mapper import (
    MaskFormerSemanticHPDatasetMapper,
)

from .data.dataset_mappers.mask_former_parsing_dataset_mapper import (
    MaskFormerParsingDatasetMapper,
)

from .data.dataset_mappers.mask_former_single_human_test_dataset_mapper import (
    MaskFormerSingleHumanTestDatasetMapper,
)

from .data.dataset_mappers.mask_former_parsing_lsj_dataset_mapper import (
    MaskFormerParsingLSJDatasetMapper,
)

from .data.build import build_detection_test_loader


# models
from .maskformer_model import MaskFormer
from .test_time_augmentation import SemanticSegmentorWithTTA, ParsingWithTTA

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator
from .evaluation.parsing_evaluation import ParsingEvaluator
from .evaluation.utils import load_image_into_numpy_array
