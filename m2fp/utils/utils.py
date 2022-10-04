import numpy as np
import torch
from pycocotools import mask as mask_util

__all__ = [
    'predictions_merge',
]


def msk2rle(maskin):
    '''
    :param mask:(frames,h,w)
    :return:
    '''
    mask = maskin > 0
    _rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    _rle["counts"] = _rle["counts"].decode("utf-8")
    return _rle


def mask_merge(masks_in):
    '''
    :param masks_in:(list)
    :return: merged mask
    '''
    ms_tmp = None
    for i, ms in enumerate(masks_in):
        if i == 0:
            ms_tmp = ms
        else:
            ms_tmp += ms

    return ms_tmp / len(masks_in)


def score_merge(scores_in):
    '''
    :param scores_in: (list)
    :return: merged score
    '''
    return np.max(scores_in)


def predictions_merge(catdict_in, IOU_thr=0.5, device=None):
    '''

    Args:
        catdict_in: predictions attributed by category

    Returns:

    '''
    for k, v in catdict_in.items():
        # sort the values by scores
        if len(v['scores']) > 1:
            index_s = np.argsort(-np.asarray(v['scores']))
            v['scores'] = np.asarray(v['scores'])[index_s]
            v['masks'] = [v['masks'][s] for s in index_s]
            v['rle'] = [msk2rle(r) for r in v['masks']]

            # apply nms
            supressed = np.zeros(len(v['masks']))
            score_m = []
            mask_m = []
            for i in range(len(v['masks'])):
                if supressed[i] == 1:
                    continue
                keep = [i]

                rle_1 = v['rle'][i]

                if i != len(v['masks']) - 1:
                    for j in range(i + 1, len(v['masks'])):
                        if supressed[j] == 1:
                            continue

                        rle_2 = v['rle'][j]

                        iou = mask_util.iou([rle_1], [rle_2], [False])
                        v['scores'][j] *= np.exp(-iou ** 2 / 0.5)

                        if iou >= IOU_thr:
                            supressed[j] = 1
                            keep.append(j)
                score_m.append(score_merge(v['scores'][np.asarray(keep)]))

                mask_m.append(mask_merge([v['masks'][s] for s in keep]))

            v['scores'] = score_m
            v['masks'] = mask_m

    merged_output = []
    for k, v in catdict_in.items():
        for _s, _m in zip(v['scores'], v['masks']):
            merged_output.append(
                {
                    "category_id": k,
                    "score": _s,
                    "mask": torch.as_tensor(_m, device=device),
                }
            )
    return merged_output
