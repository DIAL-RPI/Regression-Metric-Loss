# -*- coding: utf-8 -*-
# @Author  : Abner Chao
# @Time    : 2022/5/25 11:43

import os.path as osp

import pandas as pd

from .dataset import DataSet


def load_dataset(
        dataset_path, datainfo_path,
        label_mean, label_std,
        validation_groups, test_groups,
        cache, train_aug):
    assert len(set(test_groups) & set(validation_groups)) <= 0, 'Overlap between validation and test'
    data_info_list = pd.read_csv(datainfo_path, index_col=None)
    test_set = {'data_path': list(), 'sex': list(), 'label': list()}
    validation_set = {'data_path': list(), 'sex': list(), 'label': list()}
    train_set = {'data_path': list(), 'sex': list(), 'label': list()}

    def subset_insert(_subset, _data, _sex, _label):
        _subset['data_path'].append(_data)
        _subset['sex'].append(_sex)
        _subset['label'].append(_label)

    data_size = data_info_list.shape[0]
    for i in range(data_size):
        _info = data_info_list.iloc[i]
        label = (_info['boneage'] - label_mean) / label_std
        group = _info['group']
        pid = _info['id']
        sex = _info['sex']
        data_name = str(pid) + '.png'
        data_path = osp.join(dataset_path, data_name)
        if sex.lower() == 'm':
            sex = 1
        else:
            sex = 0
        if group in test_groups:
            subset_insert(test_set, data_path, sex, label)
        elif group in validation_groups:
            subset_insert(validation_set, data_path, sex, label)
        else:
            subset_insert(train_set, data_path, sex, label)

    return DataSet(
        train_set, aug=train_aug, cache=cache['train']), DataSet(
        validation_set, aug=False, cache=cache['val']), DataSet(
        test_set, aug=False, cache=cache['test'])
