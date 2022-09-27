import numpy as np
import torch
from pycocotools import mask as mask_util

__all__ = [
    "compute_parsing_IoP",
    'predictions_supress',
    'predictions_merge',
    'parsing_nms',
]

def compute_parsing_IoP(person_binary_mask, part_binary_mask):
    # both person_binary_mask and part_binary_mask are binary mask in shape (H, W)
    person = person_binary_mask.cpu()[:, :, None]
    person = mask_util.encode(np.array(person, order="F", dtype="uint8"))[0]
    person["counts"] = person["counts"].decode("utf-8")

    part = part_binary_mask.cpu()[:, :, None]
    part = mask_util.encode(np.array(part, order="F", dtype="uint8"))[0]
    part["counts"] = part["counts"].decode("utf-8")

    area_part = mask_util.area(part)
    i = mask_util.area(mask_util.merge([person, part], True))
    return i / area_part


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


def predictions_merge(catdict_in, IOU_thr=0.5):
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
                    "mask": np.array(_m),
                }
            )
    return merged_output


def predictions_supress(catdict_in, score_thr=0.3):
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

            for i in range(len(v['masks'])):
                rle_1 = v['rle'][i]

                if i != len(v['masks']) - 1:
                    for j in range(i + 1, len(v['masks'])):
                        rle_2 = v['rle'][j]

                        iou = mask_util.iou([rle_1], [rle_2], [False])
                        v['scores'][j] *= np.exp(-iou ** 2 / 0.5)

            keep_idx = np.where(v['scores'] > score_thr)
            v['scores'] = v['scores'][keep_idx[0]]
            v['masks'] = [v['masks'][_ki] for _ki in keep_idx[0]]

    merged_output = []
    for k, v in catdict_in.items():
        for _s, _m in zip(v['scores'], v['masks']):
            merged_output.append(
                {
                    "category_id": k,
                    "score": _s,
                    "mask": torch.from_numpy(_m),
                }
            )
    return merged_output


def parsing_nms(parsings, instance_scores, nms_thresh=0.6, num_parsing=20):
    def fast_hist(a, b, n):
        k = (a >= 0) & (a < n)
        return np.bincount(n * np.array(a[k]).astype(int) + np.array(b[k]), minlength=n ** 2).reshape(n, n)

    def cal_one_mean_iou(image_array, label_array, _num_parsing):
        hist = fast_hist(label_array, image_array, _num_parsing).astype(np.float)
        num_cor_pix = np.diag(hist)
        num_gt_pix = hist.sum(1)
        union = num_gt_pix + hist.sum(0) - num_cor_pix
        iu = num_cor_pix / union
        return iu

    def parsing_iou(src, dsts, num_classes=20):
        ious = []
        for d in range(dsts.shape[0]):
            iou = cal_one_mean_iou(src, dsts[d], num_classes)
            ious.append(np.nanmean(iou))
        return ious

    sorted_ids = (-instance_scores).argsort()
    sorted_parsings = parsings[sorted_ids]
    sorted_instance_scores = instance_scores[sorted_ids]

    keepped = [True] * sorted_instance_scores.shape[0]
    for i in range(sorted_instance_scores.shape[0] - 1):
        if not keepped[i]:
            continue
        ious = parsing_iou(sorted_parsings[i], (sorted_parsings[i + 1:])[keepped[i + 1:]], num_parsing)
        for idx, iou in enumerate(ious):
            if iou >= nms_thresh:
                keepped[i + 1 + idx] = False
    parsings = sorted_parsings[keepped]
    instance_scores = sorted_instance_scores[keepped]

    return parsings, instance_scores
