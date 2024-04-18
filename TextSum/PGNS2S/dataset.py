# -*-coding:utf-8 -*-

"""
# File       : dataset
# Time       ：2024/2/4 17:49
# Author     ：chenyu
# version    ：python 3.8
# Description：dataset类
"""

from torch.utils.data import Dataset
from data_utils import BOS_TOKEN, EOS_TOKEN


class MyDataset(Dataset):
    def __init__(self, vocab, text, title=None):
        self.is_train = True if title is not None else False  # 是否训练模式
        self.vocab = vocab   # 词表
        self.text = text     # 文本
        self.title = title   # 摘要

    def __getitem__(self, i):
        # 得到原文中的 token_id，以及 oovs
        text_ids, oovs = self.vocab.convert_text_to_ids(self.text[i].split())
        if not self.is_train:
            return {'text_ids': text_ids,
                    'oovs': oovs,
                    'len_oovs': len(oovs)}
        else:
            # 训练模式
            # title 的首尾分别加入 BOS_TOKEN 和 EOS_TOKEN
            title_ids = [self.vocab[BOS_TOKEN]] + \
                        self.vocab.convert_title_to_ids(self.title[i].split(), oovs) + \
                        [self.vocab[EOS_TOKEN]]
            return {'text_ids': text_ids,
                    'oovs': oovs,
                    'len_oovs': len(oovs),
                    'title_ids': title_ids}

    def __len__(self):
        return len(self.text)


