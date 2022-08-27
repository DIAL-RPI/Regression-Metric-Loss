# -*- coding: utf-8 -*-
# @Author  : Abner Chao
# @Time    : 2022/5/25 13:34

import os.path as osp
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tordata

from .data import SoftmaxSampler
from .net import BoneAgeNet, RegMetricLoss


class Model:
    def __init__(
            self,
            lr,
            num_workers,
            batch_size,
            restore_iter,
            total_iter,
            save_name,
            model_name,
            sigma,
            w_leak,
            p,
            train_source,
            val_source,
            test_source,):

        self.lr = lr
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.restore_iter = restore_iter
        self.total_iter = total_iter
        self.save_name = save_name
        self.model_name = model_name
        self.train_source = train_source
        self.val_source = val_source
        self.test_source = test_source

        encoder = BoneAgeNet().cuda()
        lf = RegMetricLoss(sigma=sigma, w_leak=w_leak, p=p).cuda()

        optimizer = optim.Adam([
            {'params': encoder.parameters()},
            {'params': lf.parameters()},
        ], lr=self.lr)
        models = [encoder, lf]
        self.encoder = nn.DataParallel(models[0])
        self.lf = models[1]
        self.optimizer = optimizer

        self.loss = []
        self.nz_num = []
        self.mean_diff = []
        self.s_in_loss = []
        self.LOSS = []

    def fit(self):
        if self.restore_iter != 0:
            self.load_model()

        self.encoder.train()

        softmax_sampler = SoftmaxSampler(self.train_source, self.batch_size)
        train_loader = tordata.DataLoader(
            dataset=self.train_source,
            batch_sampler=softmax_sampler,
            num_workers=self.num_workers)

        _time1 = datetime.now()
        for volumes, sex_info, labels in train_loader:
            self.restore_iter += 1
            if self.restore_iter > self.total_iter:
                break

            self.optimizer.zero_grad()

            feature = self.encoder(volumes.cuda(), sex_info.cuda())

            total_loss, nonz_num, cur_alpha, md = self.lf(feature, labels.float().cuda())
            _total_loss = total_loss.cpu().data.numpy()
            self.loss.append(_total_loss)
            self.nz_num.append(nonz_num.cpu().data.numpy())
            self.mean_diff.append(md)
            self.s_in_loss.append(self.encoder.module.emb_loss_s.cpu().data.numpy())

            torch.cuda.empty_cache()

            if _total_loss > 1e-6:
                total_loss.backward()
                self.optimizer.step()

            if self.restore_iter % 100 == 0:
                print(datetime.now() - _time1)
                _time1 = datetime.now()
                print('iter {}:'.format(self.restore_iter), end='')
                print(', loss={0:.8f}'.format(np.mean(self.loss)), end='')
                print(', nz_num={0:.8f}'.format(np.mean(self.nz_num)), end='')
                print(', cur_alpha={0:.8f}'.format(cur_alpha), end='')
                print(', mean_diff={0:.8f}'.format(np.mean(self.mean_diff)), end='')
                print(', s={0:.8f}'.format(np.mean(self.s_in_loss)), end='')
                print(', lr=', end='')
                print([self.optimizer.param_groups[i]['lr'] for i in range(len(self.optimizer.param_groups))])
                sys.stdout.flush()

                self.LOSS.append(np.mean(self.loss))
                self.loss = []
                self.nz_num = []
                self.s_in_loss = []
                self.mean_diff = []

            if self.restore_iter % 500 == 0:
                self.save_model()

    def transform(self, subset='test', batch_size=1, label_rescale_mean=0, label_rescale_std=1):
        self.encoder.eval()
        assert subset in ['train', 'val', 'test']
        source = self.test_source
        if subset == 'train':
            source = self.train_source
        elif subset == 'val':
            source = self.val_source
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            num_workers=self.num_workers)

        feature_list = list()
        label_list = list()
        sex_list = list()

        with torch.no_grad():
            for (i, (volumes, sex, labels)) in enumerate(data_loader):
                feature = self.encoder(volumes.cuda(), sex.cuda())
                feature_list.append(feature.data.cpu().numpy())
                label_list.append(labels.data.numpy())
                sex_list.append(sex.data.numpy())

        feature_list = np.concatenate(feature_list, 0)
        label_list = np.concatenate(label_list, 0)
        sex_list = np.concatenate(sex_list, 0)

        return label_list * label_rescale_std + label_rescale_mean, feature_list, sex_list

    def save_model(self):
        torch.save(self.encoder.state_dict(), osp.join(
            'checkpoints', self.model_name,
            '{}-{:0>5}-encoder.ptm'.format(self.save_name, self.restore_iter)))
        torch.save(self.optimizer.state_dict(), osp.join(
            'checkpoints', self.model_name,
            '{}-{:0>5}-optimizer.ptm'.format(self.save_name, self.restore_iter)))

    def load_model(self, restore_iter=None):
        if restore_iter is None:
            restore_iter = self.restore_iter
        self.encoder.load_state_dict(torch.load(osp.join(
            'checkpoints', self.model_name,
            '{}-{:0>5}-encoder.ptm'.format(self.save_name, restore_iter))))
        opt_path = osp.join(
            'checkpoints', self.model_name,
            '{}-{:0>5}-optimizer.ptm'.format(self.save_name, restore_iter))
        if osp.isfile(opt_path):
            self.optimizer.load_state_dict(torch.load(opt_path))
