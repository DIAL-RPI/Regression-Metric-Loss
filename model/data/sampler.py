# -*- coding: utf-8 -*-
# @Author  : Abner Chao
# @Time    : 2022/5/25 11:43


import random

import torch.utils.data as tordata


class SoftmaxSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        while (True):
            sample_indices = random.sample(
                range(self.dataset.data_size),
                self.batch_size)
            yield sample_indices

    def __len__(self):
        return self.dataset.data_size
