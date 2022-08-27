# -*- coding: utf-8 -*-
# @Author  : Abner Chao
# @Time    : 2022/5/25 11:39

import argparse
from datetime import datetime

import numpy as np
from sklearn.metrics import r2_score

from configs import conf
from model import initialization, feature2prediction, best_RV, D5

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--iter', default='14500', type=int,
                        help='iter: iteration of the checkpoint to load. Default: 14500')
    parser.add_argument('--batch_size', default='64', type=int,
                        help='batch_size: batch size for parallel test. Default: 64')
    opt = parser.parse_args()

    m = initialization(conf, train_aug=False)

    # load model checkpoint of iteration opt.iter
    print('Loading the model of iteration %d...' % opt.iter)
    m.load_model(opt.iter)

    print('Transforming...')
    print('\tTraining Set...')
    time = datetime.now()
    train_label, train_feature, train_sex = m.transform(
        'train', opt.batch_size,
        label_rescale_mean=conf['data']['label_mean'],
        label_rescale_std=conf['data']['label_std'])
    print('\t', datetime.now() - time)
    print('\tValidation Set...')
    time = datetime.now()
    val_label, val_feature, val_sex = m.transform(
        'val', opt.batch_size,
        label_rescale_mean=conf['data']['label_mean'],
        label_rescale_std=conf['data']['label_std'])
    print('\t', datetime.now() - time)
    print('\tTest Set...')
    time = datetime.now()
    test_label, test_feature, test_sex = m.transform(
        'test', opt.batch_size,
        label_rescale_mean=conf['data']['label_mean'],
        label_rescale_std=conf['data']['label_std'])
    print('\t', datetime.now() - time)

    # Inferring predictions using Nearest Neighbors
    print('Inferring Predictions...')
    print('\tValidation Set...')
    time = datetime.now()
    val_pred, nnr = feature2prediction(train_feature, train_label, val_feature, val_label)
    print('\t', datetime.now() - time)
    print('\tTest Set...')
    time = datetime.now()
    test_pred, _ = feature2prediction(train_feature, train_label, test_feature, test_label, nnr=nnr)
    print('\t', datetime.now() - time)

    print('Validation performance:')
    print('MAE:', np.abs(val_pred - val_label).mean())
    print('R2:', r2_score(val_label, val_pred))
    print('D5:', D5(val_feature, val_label))
    print('RV:', best_RV(val_feature, val_label))
    print('Test performance:')
    print('MAE:', np.abs(test_pred - test_label).mean())
    print('R2:', r2_score(test_label, test_pred))
    print('D5:', D5(test_feature, test_label))
    print('RV:', best_RV(test_feature, test_label))
