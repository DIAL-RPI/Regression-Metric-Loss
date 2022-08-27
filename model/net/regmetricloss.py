# -*- coding: utf-8 -*-
# @Author  : Abner Chao
# @Time    : 2022/5/25 11:42

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegMetricLoss(nn.Module):
    def __init__(self, sigma=1, w_leak=0.2, p=0.9):
        super(RegMetricLoss, self).__init__()
        self.sigma = sigma
        #         self.c = (1/(sigma * np.sqrt(2*np.pi))).tolist()
        self.w_leak = w_leak
        self.alpha = None
        self.p = p
        print('RMLoss: sigma=%f, p=%f' % (self.sigma, self.p))

    def forward(self, feature, label):
        f2 = (feature ** 2).sum(1)
        f_dist = f2.unsqueeze(0) + f2.unsqueeze(
            1) - 2 * torch.matmul(feature, feature.transpose(0, 1))
        f_dist = torch.sqrt(F.relu(f_dist))
        label_0 = label.unsqueeze(0)
        label_1 = label.unsqueeze(1)
        l_dist = torch.abs(label_0 - label_1)
        w = (torch.exp(-((l_dist / self.sigma) ** 2) / 2) + self.w_leak).detach()
        diff = torch.abs(f_dist - l_dist)
        w_diff = w * diff

        mean_diff = w_diff.mean().data.cpu().numpy().tolist()
        if self.alpha is None:
            self.alpha = mean_diff
        else:
            self.alpha = self.p * self.alpha + (1 - self.p) * mean_diff
        mask = w_diff.gt(self.alpha)
        nonz_num = mask.sum().float()
        #         nonz_diff = torch.masked_select(diff, mask)
        corr_w = torch.masked_select(w, mask)
        #         loss = corr_w*nonz_diff
        loss = torch.masked_select(w_diff, mask)
        #         print(self.alpha, mean_diff, nonz_num, corr_w.sum())
        return loss.sum() / (corr_w.sum() + 1e-9), nonz_num, self.alpha, mean_diff
#         return loss.mean()
