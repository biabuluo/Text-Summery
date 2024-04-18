# -*-coding:utf-8 -*-

"""
# File       : main
# Time       ：2024/2/1 14:31
# Author     ：chenyu
# version    ：python 3.8
# Description：主入口函数: 添加Attention机制的Seq2Seq模型
"""
from config import Configuration
from data_utils import load_data, collate_fn
from vocab import Vocab
from mydataset import MyDataset
from torch.utils.data import DataLoader
from model import Seq2seq, Attention, Encoder, Decoder
from torch.nn import CrossEntropyLoss
import torch
from train import train
from matplotlib import pyplot as plt
import os
import pickle as pkl


if __name__ == '__main__':
    # 加载配置类
    config = Configuration()
    # 先加载数据
    train_text, train_title = load_data(config.processed_train_contents_path,
                                        config.processed_train_sums_path)
    val_text, val_title = load_data(config.processed_val_contents_path,
                                    config.processed_val_sums_path)
    # 构建训练集词表
    if os.path.exists(config.saved_vocab_path):
        vocab = pkl.load(open(config.saved_vocab_path, 'rb'))
    else:
        vocab = Vocab(train_text + train_title)
        pkl.dump(vocab, open(config.saved_vocab_path, 'wb'))
    print(len(vocab))
    # 构造数据集
    train_dataset = MyDataset(vocab, train_text, train_title)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataset = MyDataset(vocab, val_text[:100], val_title[:100])
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True)

    # 声明模型
    vocab_size = len(vocab)
    model = Seq2seq(config, vocab_size)

    # 声明损失函数
    loss_fn = CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # 训练函数
    train_loss, val_loss = train(model, train_dataloader, val_dataloader, loss_fn, optimizer, config)
    plt.plot(train_loss, label='training loss')
    plt.plot(val_loss, label='validation loss')
    plt.legend()
    plt.title('Training and validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

