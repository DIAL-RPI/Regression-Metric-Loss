# -*- coding: utf-8 -*-
# @Author  : Abner Chao
# @Time    : 2022/5/25 11:39

import random
import warnings
from datetime import datetime

import numpy as np
from scipy import stats
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph


def residual_variance(feature, label, kn):
    warnings.filterwarnings('ignore')

    label_0 = np.expand_dims(label, 0)
    label_1 = np.expand_dims(label, 1)
    l_dist = np.abs(label_0 - label_1)

    neigh = NearestNeighbors(n_neighbors=kn)
    neigh.fit(feature)
    f_graph = neigh.kneighbors_graph(feature, mode='distance')
    nz_loc = f_graph.nonzero()
    f_graph[nz_loc[1], nz_loc[0]] = f_graph[nz_loc]

    _tmp = np.ones(f_graph.shape) * 1e10
    _tmp[f_graph.nonzero()] = f_graph[f_graph.nonzero()]
    f_graph = _tmp

    fg_dist = shortest_path(csgraph=f_graph)
    p, _ = stats.pearsonr(fg_dist.reshape(-1), l_dist.reshape(-1))

    return 1 - p


def best_RV(feature, label):
    _time = datetime.now()
    b_RV = 2
    for i in range(5, 201, 5):
        _RV = []
        random.seed(10000121)
        for _ in range(10):
            idx = random.sample(range(label.size), 1000)
            _RV.append(residual_variance(feature[idx], label[idx], kn=i))
        _RV = np.mean(_RV)
        if _RV < b_RV:
            b_RV = _RV
    print(datetime.now() - _time)
    return b_RV


def D5(feature, label):
    warnings.filterwarnings('ignore')

    label_0 = np.expand_dims(label, 0)
    label_1 = np.expand_dims(label, 1)
    l_dist = np.abs(label_0 - label_1)

    f_graph = kneighbors_graph(feature, n_neighbors=5, include_self=False).toarray()
    return ((f_graph * l_dist).sum(axis=-1) / 5).mean()
