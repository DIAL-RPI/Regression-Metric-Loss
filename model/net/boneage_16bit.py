# -*- coding: utf-8 -*-
# @Author  : Abner Chao
# @Time    : 2022/5/25 11:41

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)


class BoneAgeNet(nn.Module):
    def __init__(self):
        super(BoneAgeNet, self).__init__()
        _net = models.inception_v3(pretrained=True)
        _net_list = list(_net.children())
        self.encoder = nn.Sequential(
            BasicConv2d(
                1, 32,
                kernel_size=7,
                stride=4,
                padding=3),
            *_net_list[1:15],
            *_net_list[16:-2],
            nn.Flatten())
        self.sex_emb = nn.Embedding(2, 32)
        self.L1 = nn.Linear(2048 + 32, 1000)
        self.L2 = nn.Linear(1000, 1000)
        self.emb_loss_s = nn.Parameter(torch.tensor(1.0).float())

    def forward(self, img, sex):
        img_feature = self.encoder(img)
        sex_feature = self.sex_emb(sex)

        feature = torch.cat((img_feature, sex_feature), dim=1)

        feature = F.relu(self.L1(feature), inplace=True)
        feature = self.L2(feature)
        feature = feature * self.emb_loss_s
        return feature
