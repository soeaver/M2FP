import os.path as osp

# Root directory of project
ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))

# Path to data dir
_DATA_DIR = osp.abspath(osp.join(ROOT_DIR, 'data'))

# Required dataset entry keys
_IM_DIR = 'image_directory'
_ANN_FN = 'annotation_file'
_ANN_TYPES = 'annotation_types'
_ANN_FIELDS = 'annotation_fields'

# TODO: coco_panoptic, coco_densepose, cocohumanparts, voc, cityscape, object365v2, MHP, VIP, LaPa, ATR, PPP, VSPW, MSL
COMMON_DATASETS = {
    'cihp_semseg_val': {
        _ANN_FIELDS: {
            'num_images': 5000,
            'bbox': {
                'num_classes': 1,  # only person
                'num_instances': 17520,
            },
            'mask': {
                'num_classes': 1,  # only person
                'num_instances': 17520,
            },
            'parsing': {
                'num_classes': 20,
                'num_instances': 17520,
                'flip_map': ((13, 14), (15, 16), (17, 18)),
            },
            'semseg': {
                'num_classes': 20,
                'num_instances': -1,  # no statistics
                'seg_root': _DATA_DIR + '/CIHP/Validation/Category_ids',
                'flip_map': ((14, 15), (16, 17), (18, 19)),
                'ignore_label': 255,
                'label_shift': 0,
                'semseg_format': 'mask',
            },
        },
    },
    'cihp_parsing_val': {
        _IM_DIR: _DATA_DIR + '/CIHP/Validation/Images',
        _ANN_FN: _DATA_DIR + '/CIHP/annotations/CIHP_val.json',
        _ANN_TYPES: ('bbox', 'mask', 'parsing', 'semseg'),
        _ANN_FIELDS: {
            'num_images': 5000,
            'bbox': {
                'num_classes': 1,  # only person
                'num_instances': 17520,
            },
            'mask': {
                'num_classes': 1,  # only person
                'num_instances': 17520,
            },
            'parsing': {
                'num_classes': 20,
                'num_instances': 17520,
                'flip_map': ((14, 15), (16, 17), (18, 19)),
            },
            'semseg': {
                'num_classes': 20,
                'num_instances': -1,  # no statistics
                'seg_root': _DATA_DIR + '/CIHP/Validation/Category_ids',
                'flip_map': ((14, 15), (16, 17), (18, 19)),
                'ignore_label': 255,
                'label_shift': 0,
                'semseg_format': 'mask',
            },
        },
    },
    'lip_semseg_val': {
        _IM_DIR: _DATA_DIR + '/CIHP/Validation/Images',
        _ANN_FN: _DATA_DIR + '/CIHP/annotations/CIHP_val.json',
        _ANN_TYPES: ('bbox', 'mask', 'parsing', 'semseg'),
        _ANN_FIELDS: {
            'num_images': 10000,
            'bbox': {
                'num_classes': 1,  # only person
                'num_instances': 10000,
            },
            'mask': {
                'num_classes': 1,  # only person
                'num_instances': 10000,
            },
            'parsing': {
                'num_classes': 20,
                'num_instances': -1,
                'flip_map': ((14, 15), (16, 17), (18, 19)),
            },
            'semseg': {
                'num_classes': 20,
                'num_instances': -1,  # no statistics
                'seg_root': _DATA_DIR + '/LIP/Validation/Category_ids',
                'flip_map': ((14, 15), (16, 17), (18, 19)),
                'ignore_label': 255,
                'label_shift': 0,
                'semseg_format': 'mask',
            },
        },
    },
    'lip_parsing_val': {
        _IM_DIR: _DATA_DIR + '/LIP/Validation/Images',
        _ANN_FN: _DATA_DIR + '/LIP/annotations/LIP_val.json',
        _ANN_TYPES: ('bbox', 'mask', 'parsing', 'semseg'),
        _ANN_FIELDS: {
            'num_images': 10000,
            'bbox': {
                'num_classes': 1,  # only person
                'num_instances': 10000,
            },
            'mask': {
                'num_classes': 1,  # only person
                'num_instances': 10000,
            },
            'parsing': {
                'num_classes': 20,
                'num_instances': -1,  # no statistics
                'flip_map': ((14, 15), (16, 17), (18, 19)),
            },
            'semseg': {
                'num_classes': 20,
                'num_instances': -1,  # no statistics
                'seg_root': _DATA_DIR + '/LIP/Validation/Category_ids',
                'flip_map': ((14, 15), (16, 17), (18, 19)),
                'ignore_label': 255,
                'label_shift': 0,
                'semseg_format': 'mask',
            },
        },
    },
    "mhpv2_parsing_val": {
        _IM_DIR: _DATA_DIR + '/MHP-v2/Validation/Images',
        _ANN_FN: _DATA_DIR + '/MHP-v2/annotations/LIP_val.json',
        _ANN_TYPES: ('bbox', 'mask', 'parsing', 'semseg'),
        _ANN_FIELDS: {
            'num_images': 5000,
            'bbox': {
                'num_classes': 1,  # only person
                'num_instances': 17520,
            },
            'mask': {
                'num_classes': 1,  # only person
                'num_instances': 17520,
            },
            'parsing': {
                'num_classes': 59,
                'num_instances': 17520,
                'flip_map': ((5, 6), (7, 8), (22, 23), (24, 25), (26, 27), (28, 29), (30, 31), (32, 33)),
            },
            'semseg': {
                'num_classes': 59,
                'num_instances': -1,  # no statistics
                'seg_root': _DATA_DIR + '/MHP-v2/Validation/Category_ids',
                'flip_map': ((5, 6), (7, 8), (22, 23), (24, 25), (26, 27), (28, 29), (30, 31), (32, 33)),
                'ignore_label': 255,
                'label_shift': 0,
                'semseg_format': 'mask',
            },
        },
    },
}


def datasets():
    """Retrieve the list of available dataset names."""
    return COMMON_DATASETS.keys()


def contains(name):
    """Determine if the dataset is in the catalog."""
    return name in COMMON_DATASETS.keys()


def get_im_dir(name):
    """Retrieve the image directory for the dataset."""
    return COMMON_DATASETS[name][_IM_DIR]


def get_ann_fn(name):
    """Retrieve the annotation file for the dataset."""
    return COMMON_DATASETS[name][_ANN_FN]


def get_ann_types(name):
    """Retrieve the annotation types for the dataset."""
    return COMMON_DATASETS[name][_ANN_TYPES]


def get_ann_fields(name):
    """Retrieve the annotation fields for the dataset."""
    return COMMON_DATASETS[name].get(_ANN_FIELDS, {})
