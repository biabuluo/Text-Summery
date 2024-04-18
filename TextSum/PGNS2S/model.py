# -*-coding:utf-8 -*-

"""
# File       : model
# Time       ：2024/2/4 15:47
# Author     ：chenyu
# version    ：python 3.8
# Description：加入PGN的网络模型
"""

import random
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from data_utils import replace_oovs
'''
Encoder结构：
双层GRU
'''
class Encoder(nn.Module):
    def __init__(self, vocab, config, embed_layer):
        super(Encoder, self).__init__()
        self.vocab_size = len(vocab)   # 词典长度
        # 词嵌入采用直接初始化词向量：三维向量[batch_size, text_max_len, emb_size]
        self.embedding = embed_layer
        # dropout层
        self.dropout = nn.Dropout(config.dropout)
        # 2层双向GRU
        self.rnn = nn.GRU(config.emb_size, config.hidden_size,
                          num_layers=config.num_layers,
                          batch_first=True,
                          dropout=config.dropout,
                          bidirectional=True)
        # 线性全连接层
        self.fn = nn.Linear(config.hidden_size * 2, config.hidden_size)
    '''
    enc_input：经过填充处理的ids[batch_size, seq_len]
    text_lens：batch中每个文本的长度记录：[batch_size,]
    '''
    def forward(self, enc_input, text_lens):
        # 经过词嵌入层：[batch_size, seq_len, emb_size]
        # 使用dropout放弃使用一些特征，使得一些本来对本模型无用的特征可以舍掉
        embedded = self.dropout(self.embedding(enc_input))
        embedded = pack_padded_sequence(embedded, text_lens, batch_first=True, enforce_sorted=False)
        # 输入GRU：output: [batch_size, seq_len, hidden_size*2]  hidden: [num_layers * 2, batch_size, hidden_size]
        enc_output, hidden = self.rnn(embedded)
        # GRU 训练完成后，再恢复 padding token 状态：[batch_size, seq_len, hidden_size]
        enc_output, _ = pad_packed_sequence(enc_output, batch_first=True)
        # 取最后时间步长的output(也就是encoder的语义变量)输入全连接层
        s = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)  # [batch_size, hidden_size * 2]
        s = torch.tanh(self.fn(s))  # [batch_size, hidden_size]
        # 返回输出以及语义变量
        # output: [batch_size, seq_len, hidden_size*2]  s：# [batch_size, hidden_size]
        return enc_output, s

'''
decoder结构：
单层单向GRU
'''
class Decoder(nn.Module):
    def __init__(self, vocab, config, attention, embed_layer):
        super(Decoder, self).__init__()
        self.vocab_size = len(vocab)
        self.embedding = embed_layer
        self.attention = attention
        self.gru = nn.GRU(config.emb_size + config.hidden_size*2, config.hidden_size, batch_first=True)
        self.linear = nn.Linear(config.emb_size + 3 * config.hidden_size, self.vocab_size)
        self.softmax = nn.Softmax(dim=-1)   # softmax?
        # PGN层：计算P_gen
        self.w_gen = nn.Linear(config.hidden_size * 3 + config.emb_size, 1)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, dec_input, s, enc_output, text_lengths, coverage_vector):
        # dec_input = [batch_size, ]
        # s = [batch_size, hidden_size]
        # enc_output = [batch_size, src_len, hidden_size*2]
        # coverage_vector = [batch_size, ]
        dec_input = dec_input.unsqueeze(1)
        embedded = self.dropout(self.embedding(dec_input))  # [batch_size, 1, emb_size]
        # 输入attention层获取该时间步的权重[batch_size, 1, src_len]
        a, coverage_vector = self.attention(embedded, enc_output, text_lengths, coverage_vector)
        a = a.unsqueeze(1)
        # 矩阵乘法计算该时间步长的语义向量[batch_size, 1, hidden_size*2]
        c = torch.bmm(a, enc_output)
        gru_input = torch.cat((embedded, c), dim=2)  # [batch_size, 1, hidden_size*2+emb_size]
        # output:[batch_size, 1, hidden_size], dec_hidden[1, batch_size, hidden_size]
        dec_output, dec_hidden = self.gru(gru_input, s.unsqueeze(0))
        # 获取P_vocab:
        dec_output = self.linear(torch.cat((dec_output.squeeze(1), c.squeeze(1), embedded.squeeze(1)), dim=1))
        dec_hidden = dec_hidden.squeeze(0)

        # 计算P_gen: sigmoid(w*[dec_hidden, c, embedded])
        # [batch_size, hidden_size*3+emb_size]
        x_gen = torch.cat([dec_hidden, c.squeeze(1), embedded.squeeze(1)], dim=1)
        p_gen = torch.sigmoid(self.w_gen(x_gen))  # [batch_size, 1]
        # [batch_size, vocab_size] [batch_size, hidden_size]

        return self.softmax(dec_output), dec_hidden, a.squeeze(1), p_gen, coverage_vector


'''
实现注意力机制：
 Bahdanau Additive Attention:
 score(dec_input, enc_output)=v * tanh(w1* dec_input + w2* enc_output)
使用PGN思想以及coverage机制
'''
class Attention_with_coverage(nn.Module):
    def __init__(self, config):
        super(Attention_with_coverage, self).__init__()
        self.device = config.device
        # 用于计算带coverage机制的attention
        self.attn = nn.Linear(config.hidden_size * 3 + config.emb_size, config.hidden_size)
        self.v = nn.Linear(config.hidden_size, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, dec_input, enc_output, text_lengths, coverage_vector):
        # enc_output = [batch_size, seq_len, hidden_size*2]
        # dec_input = [batch_size, embed_size]
        # coverage_vector = [batch_size, seq_len]
        hidden_size = enc_output.shape[2]//2
        seq_len = enc_output.shape[1]
        dec_input = dec_input.repeat(1, seq_len, 1)  # dec_input扩维度：[batch_size, seq_len, embed_size]
        # coverage升维
        coverage_vector_updim = coverage_vector.unsqueeze(2).repeat(1, 1, hidden_size)
        x = torch.tanh(self.attn(torch.cat([enc_output, dec_input, coverage_vector_updim], dim=2)))
        attention = self.v(x).squeeze(-1)  # [batch_size, seq_len, ]
        # 对每个句长做掩码：不计入填充字段mask = [batch_size, seq_len]
        max_len = enc_output.shape[1]
        mask = torch.arange(max_len).expand(text_lengths.shape[0], max_len) >= text_lengths.unsqueeze(1)
        attention.masked_fill_(mask.to(self.device), float('-inf'))
        attention_weights = self.softmax(attention)  # [batch, seq_len]
        # 更新coverage_vector：先前解码器时间步的注意力分布之和
        coverage_vector += attention_weights
        # [batch_size, seq_len] [batch_size, seq_len]
        return attention_weights, coverage_vector


'''
整合PGN模型
'''
class PGN(nn.Module):
    def __init__(self, vocab, config):
        super().__init__()
        self.device = config.device
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.device = config.device
        self.attention = Attention_with_coverage(config)
        self.embed_layer = nn.Embedding(self.vocab_size, config.emb_size, padding_idx=0)
        self.encoder = Encoder(vocab, config, self.embed_layer)
        self.decoder = Decoder(vocab, config, self.attention, self.embed_layer)

    def forward(self, src, tgt, src_lengths, len_oovs, teacher_forcing_ratio=0.5):
        # src = 送入一批文本数据[batch_size, seq_len]
        # tgt = 送入一批摘要数据[batch_size, tgt_len]
        # src_lengths = [batch_size, ]
        # len_oovs = [batch_size, ]  # 每个文本的未登录词长度

        # 将 oov 替换成 <UNK>， 以便 Encoder 可以处理
        global a
        src_copy = replace_oovs(src, self.vocab, self.device)
        batch_size = tgt.shape[0]
        tgt_len = tgt.shape[1]
        vocab_size = self.vocab_size   # 词表长度
        # [batch_size, seq_len, hidden_size*2]  [batch_size, hidden_size]
        enc_output, s = self.encoder(src_copy, src_lengths)

        dec_input = tgt[:, 0]  # 取BOS作为解码器输入[batch_size, ]
        # 最终输出全0初始化 且最终的选词在词表以及原文的未登录词表中产生
        dec_outputs = torch.zeros(batch_size, tgt_len, vocab_size+max(len_oovs))
        # 初始化coverage_vector [batch_size, seq_len]
        coverage_vector = torch.zeros_like(src, dtype=torch.float32).to(self.device)
        # 对decoder输出迭代对应摘要步长
        for t in range(tgt_len-1):
            # 将 oov 替换成 <UNK>， 以便 Dncoder 可以处理
            dec_input = replace_oovs(dec_input, self.vocab, self.device)

            dec_output, s, a, p_gen, coverage_vector = self.decoder(dec_input,
                                                                    s,
                                                                    enc_output,
                                                                    src_lengths,
                                                                    coverage_vector)

            final_distribution = self.get_final_distribution(src, p_gen, dec_output, a, max(len_oovs))
            dec_outputs[:, t, :] = final_distribution
            # teacher-Forcing机制，使得最终生成的输出结果不只依赖于 decoder 的输入
            # 文本采样方法使用：Greedy Search
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = final_distribution.argmax(1)
            dec_input = tgt[:, t] if teacher_force else top1
        return dec_outputs, a, coverage_vector

    '''
    利用PGN机制,计算最终的单词分布
    src:一批文本数据：[batch_size, seq_len]
    p_gen: [batch_size, 1]
    dec_output:[batch_size, vocab_size]
    a : 注意力权重分布：[batch_size, seq_len]
    maxlen_oovs: batch中未登录词表最长长度
    '''
    def get_final_distribution(self, src, p_gen, dec_output, a, maxlen_oovs):
        batch_size = src.shape[0]
        p_gen = torch.clamp(p_gen, 0.001, 0.999)  # p_gen截断
        p_vocab_weighted = p_gen * dec_output     # 词表中的分布[batch_size, vocab_size]
        attention_weighted = (1 - p_gen) * a      # 原文中的分布[batch_size, seq_len]
        # 加入 max_oov 维度，将原文中的 OOV 单词考虑进来
        extension = torch.zeros((batch_size, maxlen_oovs), dtype=torch.float).to(self.device)
        p_vocab_extended = torch.cat([p_vocab_weighted, extension], dim=-1)  # 扩展词表[batch_size, vocab_size+maxlen_oovs]
        # p_gen * p_vocab + (1 - p_gen) * attention_weights, 将 attention weights 中的每个位置 idx 映射成该位置的 token_id
        final_distribution = p_vocab_extended.scatter_add_(dim=1, index=src, src=attention_weighted)
        # 输出最终的 vocab distribution [batch_size, vocab_size + len(oov)]
        return final_distribution



if __name__ == '__main__':
    a = torch.tensor([[1, 2, 3],
                      [3, 2, 1]])
    b = torch.tensor([[3, 2, 1],
                      [1, 2, 3]])
    a += b
    print(a)
