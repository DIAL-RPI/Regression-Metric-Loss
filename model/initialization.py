# -*- coding: utf-8 -*-
# @Author  : Abner Chao
# @Time    : 2022/5/25 16:53

import os
import os.path as osp
from copy import deepcopy

from .data import load_dataset
from .model import Model


def initialize_data(data_config, train_aug):
    print("Initializing data source...")
    train_source, val_source, test_source = load_dataset(**data_config, train_aug=train_aug)
    print("Data initialization complete.")
    return train_source, val_source, test_source


def initialize_model(model_config, train_source, val_source, test_source):
    print("Initializing model...")
    model_param = deepcopy(model_config)
    model_param['train_source'] = train_source
    model_param['val_source'] = val_source
    model_param['test_source'] = test_source
    model_param['save_name'] = '_'.join([
        '{}'.format(model_config['model_name']),
        '{}'.format(model_config['sigma']),
        '{}'.format(model_config['w_leak']),
        '{}'.format(model_config['p']),
        '{}'.format(model_config['lr']),
        '{}'.format(model_config['batch_size']),
    ])

    m = Model(**model_param)
    print("Model initialization complete.")
    return m


def initialization(config, train_aug=False):
    print("Initialzing...")
    os.makedirs(osp.join(config['WORK_PATH'], 'checkpoints', config['model']['model_name']), exist_ok=True)
    os.chdir(config['WORK_PATH'])
    os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]
    train_source, val_source, test_source = initialize_data(config['data'], train_aug=train_aug)
    return initialize_model(config['model'], train_source, val_source, test_source)
