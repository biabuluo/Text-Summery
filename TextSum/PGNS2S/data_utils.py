# -*-coding:utf-8 -*-

"""
# File       : data_utils
# Time       ：2024/2/4 17:38
# Author     ：chenyu
# version    ：python 3.8
# Description：数据工具类
"""
import torch
from torch.nn.utils.rnn import pad_sequence

# 保留 token
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
reserved_tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]

'''
加载数据
'''
def load_data(contents_path, sums_path):
    contents = []
    sums = []
    with open(contents_path, 'r', encoding='utf-8') as f:
        contents = f.readlines()
    with open(sums_path, 'r', encoding='utf-8') as f:
        sums = f.readlines()
    return contents, sums


def replace_oovs(in_tensor, vocab, device):
    # 将文本张量中所有OOV单词的id, 全部替换成 UNK_TOKEN 对应的 id，以便模型可以直接处理
    oov_token = torch.full(in_tensor.shape, vocab.unk_idx, dtype=torch.long).to(device)
    out_tensor = torch.where(in_tensor > len(vocab) - 1, oov_token, in_tensor)
    return out_tensor


'''
dataloader的收集函数：
将数据转换成模型接受的格式
input:
{'text_ids': text_ids,'oovs': oovs,'len_oovs': len(oovs),'title_ids': title_ids}
'''
def collate_fn(batch):
    # 1. text 和 title 加入 padding 处理，统一每个 batch 中的句子长度
    # 2. 统计原文中的 oov 单词
    is_train = 'title_ids' in batch[0]
    text = [torch.tensor(item['text_ids']) for item in batch]
    text_len = torch.tensor([len(item['text_ids']) for item in batch])  # [batch_size, ]
    padded_text = pad_sequence(text, batch_first=True, padding_value=0)  # [batch_size, seq_len]
    oovs = [item['oovs'] for item in batch]  # batch未登录词表
    len_oovs = [example['len_oovs'] for example in batch]  # batch未登录词表长度
    if is_train:
        title_in = [torch.tensor(item['title_ids'][:-1]) for item in batch]  # 去掉EOS
        title_out = [torch.tensor(item['title_ids'][1:]) for item in batch]  # 去掉BOS
        padded_title_in = pad_sequence(title_in, batch_first=True, padding_value=0)
        padded_title_out = pad_sequence(title_out, batch_first=True, padding_value=0)
        return padded_text, text_len, padded_title_in, padded_title_out, oovs, len_oovs
    # [batch_size, seq_len]
    return padded_text, text_len, oovs, len_oovs

