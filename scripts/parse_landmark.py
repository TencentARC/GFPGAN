import cv2
import json
import numpy as np
import torch
from basicsr.utils import FileClient, imfrombytes
from collections import OrderedDict

print('Load JSON metadata...')
# use the json file in FFHQ dataset
with open('ffhq-dataset-v2.json', 'rb') as f:
    json_data = json.load(f, object_pairs_hook=OrderedDict)

print('Open LMDB file...')
# read ffhq images
file_client = FileClient('lmdb', db_paths='datasets/ffhq/ffhq_512.lmdb')
with open('datasets/ffhq/ffhq_512.lmdb/meta_info.txt') as fin:
    paths = [line.split('.')[0] for line in fin]

save_img = False
scale = 0.5  # 0.5 for official FFHQ (512x512), 1 for others
enlarge_ratio = 1.4  # only for eyes
save_dict = {}

for item_idx, item in enumerate(json_data.values()):
    print(f'\r{item_idx} / {len(json_data)}, {item["image"]["file_path"]} ', end='', flush=True)

    # parse landmarks
    lm = np.array(item['image']['face_landmarks'])
    lm = lm * scale

    item_dict = {}
    # get image
    if save_img:
        img_bytes = file_client.get(paths[item_idx])
        img = imfrombytes(img_bytes, float32=True)

    map_left_eye = list(range(36, 42))
    map_right_eye = list(range(42, 48))
    map_mouth = list(range(48, 68))

    # eye_left
    mean_left_eye = np.mean(lm[map_left_eye], 0)  # (x, y)
    half_len_left_eye = np.max((np.max(np.max(lm[map_left_eye], 0) - np.min(lm[map_left_eye], 0)) / 2, 16))
    item_dict['left_eye'] = [mean_left_eye[0], mean_left_eye[1], half_len_left_eye]
    # mean_left_eye[0] = 512 - mean_left_eye[0]  # for testing flip
    half_len_left_eye *= enlarge_ratio
    loc_left_eye = np.hstack((mean_left_eye - half_len_left_eye + 1, mean_left_eye + half_len_left_eye)).astype(int)
    if save_img:
        eye_left_img = img[loc_left_eye[1]:loc_left_eye[3], loc_left_eye[0]:loc_left_eye[2], :]
        cv2.imwrite(f'tmp/{item_idx:08d}_eye_left.png', eye_left_img * 255)

    # eye_right
    mean_right_eye = np.mean(lm[map_right_eye], 0)
    half_len_right_eye = np.max((np.max(np.max(lm[map_right_eye], 0) - np.min(lm[map_right_eye], 0)) / 2, 16))
    item_dict['right_eye'] = [mean_right_eye[0], mean_right_eye[1], half_len_right_eye]
    # mean_right_eye[0] = 512 - mean_right_eye[0]  # # for testing flip
    half_len_right_eye *= enlarge_ratio
    loc_right_eye = np.hstack(
        (mean_right_eye - half_len_right_eye + 1, mean_right_eye + half_len_right_eye)).astype(int)
    if save_img:
        eye_right_img = img[loc_right_eye[1]:loc_right_eye[3], loc_right_eye[0]:loc_right_eye[2], :]
        cv2.imwrite(f'tmp/{item_idx:08d}_eye_right.png', eye_right_img * 255)

    # mouth
    mean_mouth = np.mean(lm[map_mouth], 0)
    half_len_mouth = np.max((np.max(np.max(lm[map_mouth], 0) - np.min(lm[map_mouth], 0)) / 2, 16))
    item_dict['mouth'] = [mean_mouth[0], mean_mouth[1], half_len_mouth]
    # mean_mouth[0] = 512 - mean_mouth[0]  # for testing flip
    loc_mouth = np.hstack((mean_mouth - half_len_mouth + 1, mean_mouth + half_len_mouth)).astype(int)
    if save_img:
        mouth_img = img[loc_mouth[1]:loc_mouth[3], loc_mouth[0]:loc_mouth[2], :]
        cv2.imwrite(f'tmp/{item_idx:08d}_mouth.png', mouth_img * 255)

    save_dict[f'{item_idx:08d}'] = item_dict

print('Save...')
torch.save(save_dict, './FFHQ_eye_mouth_landmarks_512.pth')
