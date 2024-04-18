# -*-coding:utf-8 -*-

"""
# File       : vocab
# Time       ：2024/1/31 19:31
# Author     ：chenyu
# version    ：python 3.8
# Description：为训练集构造词典
"""
from collections import defaultdict
from data_utils import reserved_tokens

'''
词表类
'''


class Vocab:
    '''
    sentences:输入文本列表
    min_freq:最小词频
    '''

    def __init__(self, sentences, min_freq=10):
        self.token2idx = {}  # token向index映射
        self.idx2token = list()  # index向toke映射
        token_freqs = defaultdict(int)  # 词频记录器
        self.UNK_TOKEN = '<UNK>'  # 未登录词token
        token_list = reserved_tokens  # token表

        for sentence in sentences:  # 记录词频
            for token in sentence.split(' '):
                token_freqs[token] += 1
        # 排序,并取前50000作为字典容量
        token_freqs = sorted(token_freqs.items(), key=lambda x: x[1], reverse=True)[:50000-4]
        token_list += [token
                       for token, freq in token_freqs
                       if freq >= min_freq]  # 过滤低词频
        # 词表最后加入<UNK>
        if self.UNK_TOKEN not in token_list:
            token_list = token_list + [self.UNK_TOKEN]
        # 更新映射
        for token in token_list:
            self.idx2token.append(token)
            self.token2idx[token] = len(self.idx2token) - 1
        self.unk_idx = self.token2idx[self.UNK_TOKEN]  # 未登录词token的index

    def __len__(self):
        return len(self.idx2token)  # 返回词表长度

    def __getitem__(self, token):
        # 获取对应token的index
        return self.token2idx.get(token, self.unk_idx)

    '''
    根据idx列表或者token列表互换
    '''

    def convert_tokens_to_ids(self, tokens):
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.idx2token[idx] for idx in ids]

