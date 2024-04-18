# -*-coding:utf-8 -*-

"""
# File       : mydataset
# Time       ：2024/1/31 20:51
# Author     ：chenyu
# version    ：python 3.8
# Description：mydataset类用于构造模型对应的输入
"""
from torch.utils.data import Dataset, DataLoader
from data_utils import PAD_TOKEN, UNK_TOKEN, EOS_TOKEN, BOS_TOKEN
'''
继承Dataset类
需要实现item、len方法
'''
class MyDataset(Dataset):
    """
    传入词表、文本列表、摘要列表
    """
    def __init__(self, vocab, contents, sums=None):
        self.is_train = True if sums is not None else False   # 是否是训练模式
        # 将文本转化为ids
        self.contents_ids = [vocab.convert_tokens_to_ids(content.strip().split(' ')) for content in contents]
        # 将摘要转化为ids
        if self.is_train:
            self.sums_ids = [[vocab[BOS_TOKEN]] +
                             vocab.convert_tokens_to_ids(sum.strip().split(' '))
                             + [vocab[EOS_TOKEN]]
                             for sum in sums]


    def __getitem__(self, i):
        return (self.contents_ids[i], self.sums_ids[i]) if self.is_train else (self.contents_ids[i],)

    def __len__(self):
        return len(self.contents_ids)



