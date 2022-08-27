# -*- coding: utf-8 -*-
# @Author  : Abner Chao
# @Time    : 2022/5/25 11:39

conf = {
    'WORK_PATH': './work',
    'CUDA_VISIBLE_DEVICES': "0",
    'data': {
        'dataset_path': './data/img',
        'datainfo_path': './data/data_info.csv',
        'label_mean': 127.3,
        'label_std': 41.2,
        'validation_groups': [1],
        'test_groups': [2],
        # If cache data in memory
        'cache': {
            'train': True,
            'val': False,
            'test': False,
        }
    },

    'model': {
        'lr': 1e-4,
        'num_workers': 8,
        'batch_size': 64,
        'restore_iter': 0,
        'total_iter': 15000,
        'model_name': 'BAA-RMLoss',
        'sigma': 0.5,
        'w_leak': 0.1,
        'p': 0.9,
    }
}
