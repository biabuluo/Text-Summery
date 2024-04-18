# -*-coding:utf-8 -*-

"""
# File       : config
# Time       ：2024/2/4 17:17
# Author     ：chenyu
# version    ：python 3.8
# Description：配置类
"""
import torch


class Configuration(object):
    def __init__(self):
        # 设置模型参数
        self.emb_size = 128
        self.hidden_size = 256
        self.num_layers = 2
        self.dropout = 0.5
        # 设置coverage
        self.coverage = True
        self.cov_lambda = 1  # 计算总体loss时，设置coverage loss的权重。

        # 设置训练参数
        self.batch_size = 100
        self.epochs = 5
        self.lr = 1e-3
        self.max_grad_norm = 2  # 梯度最大截断值，避免出现梯度爆炸
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # path
        self.processed_train_contents_path = 'processed_data/train_src.txt'
        self.processed_train_sums_path = 'processed_data/train_tgt.txt'
        self.processed_val_contents_path = 'processed_data/dev_src.txt'
        self.processed_val_sums_path = 'processed_data/dev_tgt.txt'
        self.processed_test_contents_path = 'processed_data/test_src.txt'
        self.processed_test_sums_path = 'processed_data/test_tgt.txt'
        self.saved_model_path = 'saved_model/'
        self.saved_vocab_path = 'saved_model/saved_vocab.pkl'