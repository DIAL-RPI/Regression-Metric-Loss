# -*- coding: utf-8 -*-
# @Author  : Abner Chao
# @Time    : 2022/5/25 11:43

import ctypes
import multiprocessing as mp

import cv2
import numpy as np
import torch.utils.data as tordata
from PIL import Image
from torchvision import transforms as tortrans


class DataSet(tordata.Dataset):
    IMG_H = 520
    IMG_W = 400
    IMG_SIZE = IMG_H * IMG_W

    def __init__(self, subset_dict, aug=False, cache=False):
        super(DataSet, self).__init__()
        self.data_path = subset_dict['data_path']
        self.sex = subset_dict['sex']
        self.label = subset_dict['label']
        self.data_size = len(self.label)
        self.if_aug = aug
        img_transform = []
        if aug:
            img_transform += [
                tortrans.RandomAffine(
                    20, translate=(0.2, 0.2), scale=(0.8, 1.2)),
                tortrans.ColorJitter(contrast=0.5),
                tortrans.RandomHorizontalFlip(),
            ]
        img_transform.append(tortrans.ToTensor())
        self.img_transform = tortrans.Compose(img_transform)

        self.if_cache = cache
        if cache:
            self.cache_flag = mp.Array(ctypes.c_bool, self.data_size)
            shared_data_base = mp.Array(ctypes.c_uint8, self.data_size * self.IMG_SIZE)
            shared_data = np.ctypeslib.as_array(shared_data_base.get_obj())
            self.data = shared_data.reshape(self.data_size, self.IMG_H, self.IMG_W)

    def load_all_data(self):
        for i in range(self.data_size):
            self.load_data(i)

    def load_data(self, index):
        return self.__getitem__(index)

    def clear_cache(self):
        self.data = [None] * self.data_size

    def __loader__(self, path):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    def __getitem__(self, index):

        if self.if_cache:
            if self.cache_flag[index] is False:
                self.cache_flag[index] = True
                self.data[index] = self.__loader__(self.data_path[index])
            data = self.data[index].copy()
        else:
            data = self.__loader__(self.data_path[index])

        data = self.img_transform(Image.fromarray(data))

        return data, self.sex[index], self.label[index]

    def __len__(self):
        return len(self.label)
