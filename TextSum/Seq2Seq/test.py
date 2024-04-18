# -*-coding:utf-8 -*-

"""
# File       : test
# Time       ：2024/2/19 17:23
# Author     ：chenyu
# version    ：python 3.8
# Description：测试一下
"""


import torch
from data_utils import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN, UNK_TOKEN
import pickle as pkl
from config import Configuration
from vocab import Vocab
from model import Seq2seq

# 模型预测过程
def predict(model, src, src_lengths, device, max_len=20):
    model.eval()
    with torch.no_grad():
        enc_input = torch.tensor(vocab.convert_tokens_to_ids(src)).unsqueeze(0)
        src_lengths = torch.tensor(src_lengths)
        enc_output, s = model.encoder(enc_input.to(device), src_lengths)
        dec_input = torch.tensor([vocab[BOS_TOKEN]]).to(device)

        dec_words = []
        for t in range(max_len):
            dec_output, s = model.decoder(dec_input, s, enc_output, src_lengths)
            dec_output = dec_output.argmax(-1)
            if dec_output.item() == vocab[EOS_TOKEN]:
                dec_words.append(EOS_TOKEN)
                break
            else:
                dec_words.append(vocab.idx2token[dec_output.item()])
            dec_input = dec_output
    return dec_words


# 获取词表
config = Configuration()
vocab = pkl.load(open(config.saved_vocab_path, 'rb'))

# 加载模型
model = Seq2seq(config, len(vocab))
model.load_state_dict(torch.load('saved_model/model_S2S_attention.pt', map_location=torch.device('cpu')))

text = '中国 铁路 总公司 消息 ， 自 2015 年 1 月 5 日起 ， 自行车 不能 进站 乘车 了 。 骑友 大呼 难以 接受 。 这部分 希望 下车 就 能 踏上 骑游 旅程 的 旅客 ， 只能 先 办理 托运 业务 ， 可 咨询 12306 客服 电话 ， 就近 提前 办理 。 运费 每公斤 价格 根据 运输 里程 不同 而 不同 。 '
text = text.strip().split()
lengths = [len(text)]
dec_words = predict(model, text, lengths, config.device)
print(''.join(dec_words))
