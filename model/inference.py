# -*- coding: utf-8 -*-
# @Author  : Abner Chao
# @Time    : 2022/5/26 14:48


import warnings
from datetime import datetime

import numpy as np
from sklearn.neighbors import NearestNeighbors


def RKNN_regression(ref_f, ref_l, ques, r, kn):
    warnings.filterwarnings('ignore')
    neigh = NearestNeighbors(radius=r, n_neighbors=kn)
    neigh.fit(ref_f)
    kn_graph = neigh.kneighbors_graph(ques, mode='distance')
    nz_kn = kn_graph.nonzero()
    f_graph = neigh.radius_neighbors_graph(ques, mode='distance')
    f_graph[nz_kn] = kn_graph[nz_kn]
    nz_f = f_graph.nonzero()
    f_graph[nz_f] = np.exp(-np.power((3 * f_graph[nz_f] / r), 2) / 2) + 1e-16
    f_graph = f_graph.toarray()

    pred = np.matmul(f_graph, ref_l.reshape(-1, 1)).reshape(-1) / f_graph.sum(-1)
    return pred


def search_nnr(train_feature, train_label, query_feature, query_label, lb=1, rb=500, intr=20):
    while rb - lb > 5:
        if intr > rb - lb:
            intr = rb - lb
        poi = (rb + lb) / 2
        pred_1 = RKNN_regression(train_feature, train_label, query_feature, kn=1, r=poi / 1000)
        mae_1 = np.abs(pred_1 - query_label).mean()
        pred_2 = RKNN_regression(train_feature, train_label, query_feature, kn=1, r=(poi + intr) / 1000)
        mae_2 = np.abs(pred_2 - query_label).mean()
        grad = mae_2 - mae_1
        if grad >= 0:
            rb = poi
        else:
            lb = poi
    return (rb + lb) / 2, mae_1


def feature2prediction(train_feature, train_label, inf_feature, inf_label, nnr=None):
    """
    Automatically search nnr if not provided.

    :param train_feature: features of the training data
    :param train_label: labels of the training data
    :param inf_feature: features of the inference data
    :param inf_label: labels of the inference data
    :param nnr: radius of the Nearest Neighbors
    :return: inf_pred, nnr
    """
    np.set_printoptions(precision=3, suppress=True)

    if nnr is None:
        print('\t', 'Searching NN radius...')
        time = datetime.now()
        best_r, val_mae = search_nnr(train_feature, train_label, inf_feature, inf_label)
        for i in range(max(int(best_r) - 30, 0), int(best_r) + 30, 2):
            _pred = RKNN_regression(train_feature, train_label, inf_feature, kn=1, r=i / 1000)
            _mae = np.abs(_pred - inf_label).mean()
            if val_mae > _mae:
                val_mae = _mae
                best_r = i
        print('\t', datetime.now() - time)
        nnr = best_r

    inf_pred = RKNN_regression(train_feature, train_label, inf_feature, kn=1, r=nnr / 1000)
    return inf_pred, nnr
