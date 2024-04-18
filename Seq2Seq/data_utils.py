# -*-coding:utf-8 -*-

"""
# File       : data_analysis
# Time       ：2024/1/31 15:12
# Author     ：chenyu
# version    ：python 3.8
# Description：对训练数据进行分词处理
"""
import jieba
import torch
from torch.nn.utils.rnn import pad_sequence
import pickle as pkl
import numpy as np

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

'''
使用jieba对数据进行简单分词
去停用词以及标点符号
'''
def tokenize(text):
    # 获取所有停用词
    stopwords = set()
    with open('../utils_file/StopWords_CN.txt', 'r', encoding='utf-8') as f:
        for word in f.readlines():
            stopwords.add(word)

    result = []
    for word in jieba.lcut(text):
        if word not in stopwords:
            result.append(word)
    return " ".join(result)


'''
处理训练数据
'''
def process(contents_path, sums_path, contents_saved_dest, sums_saved_dest):
    contents, sums = load_data(contents_path, sums_path)
    contents = [tokenize(text) for text in contents]
    sums = [tokenize(text) for text in sums]
    # 写入文件
    with open('processed_data/'+contents_saved_dest, 'w', encoding='utf-8') as f:
        f.writelines(contents)
    with open('processed_data/'+sums_saved_dest, 'w', encoding='utf-8') as f:
        f.writelines(sums)


'''
dataloader的collate_fn
用于堆叠模型的batch输入
'''
def collate_fn(batch):
    # 判断是否是训练模式：即是否有摘要ids
    global padded_title_in, padded_title_out
    is_train = True if len(batch[0]) == 2 else False
    # 取出contents_ids
    contents_ids_list = [torch.tensor(item[0]) for item in batch]
    # 取出所有ids长度
    ids_len_list = torch.tensor([len(item[0]) for item in batch])
    # 对content的ids列表填充
    padded_text = pad_sequence(contents_ids_list, batch_first=True, padding_value=0)
    # 对摘要的ids填充
    if is_train:
        title_in = [torch.tensor(item[1][:-1]) for item in batch]  # 去掉EOS
        title_out = [torch.tensor(item[1][1:]) for item in batch]  # 去掉BOS
        padded_title_in = pad_sequence(title_in, batch_first=True, padding_value=0)
        padded_title_out = pad_sequence(title_out, batch_first=True, padding_value=0)
    return (padded_text, ids_len_list, padded_title_in, padded_title_out) \
        if is_train else (padded_text, ids_len_list)



if __name__ == '__main__':
    # # 处理训练集数据
    contents_path = '../LCSTS_small/train.src.txt'
    sums_path = '../LCSTS_small/train.tgt.txt'
    process(contents_path, sums_path, 'train_src.txt', 'train_tgt.txt')
    # # 处理验证集数据
    # contents_path = '../LCSTS_small/dev.src.txt'
    # sums_path = '../LCSTS_small/dev.tgt.txt'
    # process(contents_path, sums_path, 'dev_src.txt', 'dev_tgt.txt')

    # # # 处理测试集数据
    # contents_path = '../LCSTS_small/test.src.txt'
    # sums_path = '../LCSTS_small/test.tgt.txt'
    # process(contents_path, sums_path, 'test_src.txt', 'test_tgt.txt')


