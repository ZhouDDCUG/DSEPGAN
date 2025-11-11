import os
# import cv2
import math
import numpy as np
from enum import Enum, auto, unique

import torch
from torch.utils.data import Dataset


def get_pair_path(root_dir, target_dir_name, ref_dir_name):
    paths = [None, None, None, None]
    target_dir = root_dir + '/' + target_dir_name
    for filename in os.listdir(target_dir):
        if filename[0] == 'M':
            paths[0] = os.path.join(target_dir, filename)
        else:
            paths[1] = os.path.join(target_dir, filename)

    target_dir = root_dir + '/' + ref_dir_name
    for filename in os.listdir(target_dir):
        if filename[0] == 'M':
            paths[2] = os.path.join(target_dir, filename)
        else:
            paths[3] = os.path.join(target_dir, filename)

    return paths


def load_image_pair(root_dir, target_dir_name, ref_dir_name):
    paths = get_pair_path(root_dir, target_dir_name, ref_dir_name)
    images = []
    for p in paths:
        im = np.load(p)
        images.append(im)

    return images


def transform_image(image, flip_num, rotate_num0, rotate_num):
    # 掩模处理：该代码将图像切片中小于零和大于10000的值的像素替换成0和10000（在后面的归一化操作中有用），
    # 并对掩模（门控函数）像素进行初始化。
    # 对于图像中的每个掩模像素，掩模张量都会为该像素设置一个值，控制该像素是否被用于训练。
    # 在这里，我们设置了一个当前图像中所有非零数值的张量，用于以后的掩模操作。
    image = image.astype(np.float32)
    image_mask = np.ones(image.shape)
    negtive_mask = np.where(image < 0)
    inf_mask = np.where(image > 10000.)

    image_mask[negtive_mask] = 0.0
    image_mask[inf_mask] = 0.0
    image[negtive_mask] = 0.0
    image[inf_mask] = 10000.0
    # image = image.astype(np.float32)

    # flip_num对应一个随机变量（0或1），表示是否在水平方向翻转图像
    if flip_num == 1:
        image = image[:, :, ::-1]

    C, H, W = image.shape
    if rotate_num0 == 1:
        # -90
        if rotate_num == 2:
            image = image.transpose(0, 2, 1)[::-1, :]
        # 90
        elif rotate_num == 1:
            image = image.transpose(0, 2, 1)[:, ::-1]
        # 180
        else:
            image = image.reshape(C, H * W)[:, ::-1].reshape(C, H, W)

    image = torch.from_numpy(image.copy())
    image_mask = torch.from_numpy(image_mask)

    image.mul_(0.001)
    image = image * 2 - 1
    return image, image_mask


# Data Augment, flip、rotate
class PatchSet(Dataset):  #从root_dir得到真正的训练数据，那image_dates的作用是什么？只是为了得到total_index吗？
    def __init__(self, root_dir, image_dates, image_size, patch_size, patch_stride):
        super(PatchSet, self).__init__()
        self.root_dir = root_dir
        self.image_dates = image_dates
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        max_end_h = image_size[0] - patch_size  # 最大下标是可以取到值的
        max_end_w = image_size[1] - patch_size
        h_index_list = [i for i in range(0, max_end_h + 1, patch_stride)]
        w_index_list = [i for i in range(0, max_end_w + 1, patch_stride)]

        if max_end_h % patch_stride != 0:
            h_index_list.append(image_size[0] - patch_size)
        if max_end_w % patch_stride != 0:
            w_index_list.append(image_size[1] - patch_size)

        self.total_index = len(self.image_dates) * len(h_index_list) * len(w_index_list)
        print(self.total_index)

    def __getitem__(self, item):
        images = []

        im = np.load(os.path.join(self.root_dir, str(item) + '.npy'))
        for i in range(4):
            images.append(im[i * 4: i * 4 + 4, :, :])
        patches = [None] * len(images)
        masks = [None] * len(images)

        flip_num = np.random.choice(2)
        rotate_num0 = np.random.choice(2)
        rotate_num = np.random.choice(3)
        for i in range(len(patches)):
            im = images[i]
            im, im_mask = transform_image(im, flip_num, rotate_num0, rotate_num)
            patches[i] = im
            masks[i] = im_mask

        gt_mask = masks[0] * masks[1] * masks[2] * masks[3]

        return patches[0], patches[1], patches[2], patches[3], gt_mask

    def __len__(self):
        return self.total_index
