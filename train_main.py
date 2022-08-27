# -*- coding: utf-8 -*-
# @Author  : Abner Chao
# @Time    : 2022/5/25 11:39


from configs import conf
from model import initialization

if __name__ == '__main__':
    m = initialization(conf, train_aug=True)

    print("Training START")
    m.fit()
    print("Training COMPLETE")
