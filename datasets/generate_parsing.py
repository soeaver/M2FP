import os
import scipy.io as scio
import numpy as np
import cv2
import shutil
import glob
import json


def _mkdir(_dir):
    if not os.path.exists(_dir):
        os.mkdir(_dir)


if __name__ == '__main__':
    _categories = {0: 'background', 1: 'Lower-arm', 2: 'Head', 3: 'Upper-leg', 4: 'Torso', 5: 'Lower-leg',
                   6: 'Upper-arm'}

    key = 'test'
    datasets = {'train': ['train', 'Training'], 'val': ['val', 'Validation'], 'test': ['test', 'Testing']}
    key_list = datasets[key]

    ids_file = 'pascal-person-part/tools/{}_id.txt'.format(key_list[0])
    image_root = 'pascal-person-part/{}/Images/'.format(key_list[1])
    human_dir = 'pascal-person-part/{}/Human_ids/'.format(key_list[1])
    cate_dir = 'pascal-person-part/{}/Category_ids/'.format(key_list[1])
    save_json_file = 'pascal-person-part/annotations/PASCAL-Person-Part_parsing_{}.json'.format(key_list[0])

    ids_list = []
    f = open(ids_file, 'r')
    for l in f:
        img_id = l.strip()
        ids_list.append(img_id)
    f.close()

    # generate json
    categories = [{'id': 0, 'name': 'Background'}, {'id': 1, 'name': 'Lower-arm'}, {'id': 2, 'name': 'Head'},
                  {'id': 3, 'name': 'Upper-leg'}, {'id': 4, 'name': 'Torso'}, {'id': 5, 'name': 'Lower-leg'},
                  {'id': 6, 'name': 'Upper-arm'}, {'id': 7, 'name': 'Human'}]
    images = []
    annotations = []
    ins_id = 1
    for idx, img_id in enumerate(ids_list):
        img = cv2.imread(image_root + '{}.jpg'.format(img_id))
        height, width, _ = img.shape
        image = {'height': height, 'width': width, 'id': idx + 1, 'file_name': '{}.jpg'.format(img_id)}
        images.append(image)

        # if key == 'test':
        #     print(idx, img_id)
        #     continue

        semseg = cv2.imread(cate_dir + '{}.png'.format(img_id), 0)
        h, w = semseg.shape
        assert height == h and width == w

        mask = cv2.imread(human_dir + '{}.png'.format(img_id), 0)
        human_ids = np.unique(mask)
        # bg_id_index = np.where(human_ids == 0)[0]
        # human_ids = np.delete(human_ids, bg_id_index)
        for h_id in human_ids:
            human_mask = np.where(mask == h_id, 1, 0).astype(np.uint8)
            contours, _ = cv2.findContours(human_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            segmentation = []
            for j in contours:
                if len(j) < 20:
                    continue
                _segs = []
                for p in j:
                    _segs.extend([int(p[0][1]), int(p[0][0])])
                    # img[p[0][1]][p[0][0]] = [0, 0, 255]
                segmentation.append(_segs)
            rect = cv2.boundingRect(human_mask.copy())
            x, y, w, h = rect

            if h_id == 0:
                print(idx, img_id, ins_id, h_id, 0)
                anno = {'segmentation': segmentation, 'iscrowd': 0, 'area': w * h, 'image_id': idx + 1,
                        'bbox': [x, y, w, h], 'category_id': 0, 'id': ins_id, 'parsing_id': int(h_id),
                        'ispart': 0, 'isfg': 0}
                annotations.append(anno)
                ins_id += 1
            else:
                print(idx, img_id, ins_id, h_id, 7)
                anno = {'segmentation': segmentation, 'iscrowd': 0, 'area': w * h, 'image_id': idx + 1,
                        'bbox': [x, y, w, h], 'category_id': 7, 'id': ins_id, 'parsing_id': int(h_id),
                        'ispart': 0, 'isfg': 1}
                annotations.append(anno)
                ins_id += 1

                human_semseg = semseg * human_mask
                part_ids = np.unique(human_semseg)
                bg_id_index = np.where(part_ids == 0)[0]
                part_ids = np.delete(part_ids, bg_id_index)
                for p_id in part_ids:
                    part_mask = np.where(human_semseg == p_id, 1, 0).astype(np.uint8)
                    contours, _ = cv2.findContours(part_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    segmentation = []
                    for j in contours:
                        if len(j) < 10:
                            continue
                        _segs = []
                        for p in j:
                            _segs.extend([int(p[0][1]), int(p[0][0])])
                            # img[p[0][1]][p[0][0]] = [0, 0, 255]
                        segmentation.append(_segs)
                    rect = cv2.boundingRect(part_mask.copy())
                    x, y, w, h = rect

                    print(idx, img_id, ins_id, h_id, p_id)
                    anno = {'segmentation': segmentation, 'iscrowd': 0, 'area': w * h, 'image_id': idx + 1,
                            'bbox': [x, y, w, h], 'category_id': int(p_id), 'id': ins_id, 'parsing_id': int(h_id),
                            'ispart': 1, 'isfg': 1}
                    annotations.append(anno)
                    ins_id += 1

    json.dump({'images': images, 'categories': categories, 'annotations': annotations}, open(save_json_file, 'w'))
