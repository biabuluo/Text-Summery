# -*-coding:utf-8 -*-

"""
# File       : vocab
# Time       ：2024/2/4 17:52
# Author     ：chenyu
# version    ：python 3.8
# Description：词典类
"""
from collections import defaultdict
from data_utils import reserved_tokens, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
# 生成词典
class Vocab:
    def __init__(self, sentences, min_freq=1):
        self.idx2token = list()
        self.token2idx = {}
        token_freqs = defaultdict(int)
        for sentence in sentences:  # 记录词频
            for token in sentence.split(' '):
                token_freqs[token] += 1
        unique_tokens = reserved_tokens
        # 排序,并取前50000作为字典容量
        token_freqs = sorted(token_freqs.items(), key=lambda x: x[1], reverse=True)[:50001]
        unique_tokens += [token for token, freq in token_freqs if freq >= min_freq]
        if UNK_TOKEN not in unique_tokens:
            unique_tokens = unique_tokens + [UNK_TOKEN]
        for token in unique_tokens:
            self.idx2token.append(token)
            self.token2idx[token] = len(self.idx2token) - 1
        self.unk_idx = self.token2idx[UNK_TOKEN]

    def __len__(self):
        return len(self.idx2token)

    def __getitem__(self, token):
        return self.token2idx.get(token, self.unk_idx)

    def convert_tokens_to_ids(self, tokens):
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.idx2token[idx] for idx in ids]

    # 将 source token 转化为 ids, 其中 unk_token 加入到 oovs
    def convert_text_to_ids(self, text_tokens):
        ids = []   # ids
        oovs = []  # 未登录词表
        for token in text_tokens:
            i = self[token]
            if i == self.unk_idx:  # 未登录词
                if token not in oovs:
                    oovs.append(token)
                oov_idx = oovs.index(token)
                ids.append(oov_idx + len(self))
            else:
                ids.append(i)
        return ids, oovs

    # 将 title token 转化为 ids，考虑 source token 中出现的 oovs
    def convert_title_to_ids(self, title_tokens, oovs):
        ids = []
        for token in title_tokens:
            i = self[token]
            if i == self.unk_idx:  # 摘要未登陆词
                if token in oovs:  # 摘要未登录词出现在文本未登录词表中
                    token_idx = oovs.index(token) + len(self)
                    ids.append(token_idx)
                else:              # 是摘要独有的未登录词直接记录unk
                    ids.append(self.unk_idx)
            else:
                ids.append(i)
        return ids

