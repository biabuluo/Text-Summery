# -*-coding:utf-8 -*-

"""
# File       : config
# Time       ：2024/1/31 15:19
# Author     ：chenyu
# version    ：python 3.8
# Description：配置类
"""

import torch

class Configuration(object):
    def __init__(self):
        self.saved_model_path = 'saved_model/'
        self.saved_vocab_path = 'saved_model/saved_vocab.pkl'

        self.train_sums_path = '../LCSTS_small/train.tgt.txt'
        self.train_contents_path = '../LCSTS_small/train.src.txt'
        self.processed_train_contents_path = 'processed_data/train_src.txt'
        self.processed_train_sums_path = 'processed_data/train_tgt.txt'
        self.processed_val_contents_path = 'processed_data/dev_src.txt'
        self.processed_val_sums_path = 'processed_data/dev_tgt.txt'
        self.processed_test_contents_path = 'processed_data/test_src.txt'
        self.processed_test_sums_path = 'processed_data/test_tgt.txt'
        # 模型超参
        self.emb_size = 128
        self.hidden_size = 256
        self.batch_size = 64
        self.epochs = 20
        self.max_grad_norm = 2  # 梯度最大截断值，避免出现梯度爆炸
        self.lr = 3e-4  # 学习率

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



