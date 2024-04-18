# -*-coding:utf-8 -*-

"""
# File       : data_anaylsis
# Time       ：2024/1/31 16:00
# Author     ：chenyu
# version    ：python 3.8
# Description：对训练数据进行分析
"""
from config import Configuration
from data_utils import load_data
import matplotlib.pyplot as plt

def analysis_data(contents_path, sums_path):
    train_contents, train_sums = load_data(contents_path, sums_path)
    print(len(train_contents), ':', len(train_sums))
    text_len = [len(sentence) for sentence in train_contents]
    title_len = [len(sentence) for sentence in train_sums]

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.hist(text_len, bins=80)
    plt.title('Distribution of text length')
    plt.xlabel('text length')
    plt.ylabel('count')

    plt.subplot(1, 2, 2)
    plt.hist(title_len, bins=40)
    plt.title('Distribution of title length')
    plt.xlabel('title length')
    plt.ylabel('count')
    plt.show()

if __name__ == '__main__':
    config = Configuration()
    analysis_data(config.train_contents_path, config.train_sums_path)


