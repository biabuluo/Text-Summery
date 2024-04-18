# -*-coding:utf-8 -*-

"""
# File       : model
# Time       ：2024/2/1 11:02
# Author     ：chenyu
# version    ：python 3.8
# Description：构造神经网络模型
传统S2S问题：RNN的遗忘问题，当前状态需要上一个状态
加入Attention：
decoder产生状态需要回顾一遍encoder中的状态
还能使得当前步长的decoder更关注encoder的哪个状态（权值）
问题：
计算量大：“decoder产生状态需要回顾一遍encoder中的状态”
由于 Attention 可能聚焦于某些单词，容易生成重复的的词语或短句
容易产生事实性的错误，比如姓名之间的错误
只能生成词表中的单词，无法处理 OOV（out of vocabulary）问题
"""
import random
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence




'''
Encoder结构：
双层GRU
'''
class Encoder(nn.Module):
    def __init__(self, embed_layer, vocab_size, emb_size, hidden_size, dropout=0.5, num_layers=2):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size   # 词典长度
        # 词嵌入采用直接初始化词向量：三维向量[batch_size, text_max_len, emb_size]
        # 可以优化：采用预训练的词向量（到时候使用Bert预训练模型解决）
        self.embedding = embed_layer
        # dropout层
        self.dropout = nn.Dropout(dropout)
        # 2层双向GRU
        self.rnn = nn.GRU(emb_size, hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout,
                          bidirectional=True)
        # 线性全连接层
        self.fn = nn.Linear(hidden_size * 2, hidden_size)


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
        s = torch.relu(self.fn(s))  # [batch_size, hidden_size]
        # 返回输出以及语义变量
        # output: [batch_size, seq_len, hidden_size*2]  s：# [batch_size, hidden_size]
        return enc_output, s

'''
decoder结构：
单层单向GRU
'''
class Decoder(nn.Module):
    def __init__(self, embed_layer, vocab_size, emb_size, hidden_size, attention, dropout=0.5):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = embed_layer
        self.attention = attention
        self.gru = nn.GRU(emb_size + hidden_size*2, hidden_size, batch_first=True)
        self.linear = nn.Linear(emb_size + 3 * hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        # dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input, s, enc_output, text_lengths):
        # dec_input = [batch_size, ]
        # s = [batch_size, hidden_size]
        # enc_output = [batch_size, src_len, hidden_size*2]
        dec_input = dec_input.unsqueeze(1)
        embedded = self.dropout(self.embedding(dec_input))  # [batch_size, 1, emb_size]
        # 输入attention层获取该时间步的权重[batch_size, 1, src_len]
        a = self.attention(embedded, enc_output, text_lengths).unsqueeze(1)
        # 矩阵乘法计算该时间步长的语义向量[batch_size, 1, hidden_size*2]
        c = torch.bmm(a, enc_output)
        gru_input = torch.cat((embedded, c), dim=2)  # [batch_size, 1, hidden_size*2+emb_size]
        # output:[batch_size, 1, hidden_size], dec_hidden[1, batch_size, hidden_size]
        dec_output, dec_hidden = self.gru(gru_input, s.unsqueeze(0))
        # [batch_size, vocab_size] [batch_size, hidden_size]
        dec_output = self.linear(torch.cat((dec_output.squeeze(1), c.squeeze(1), embedded.squeeze(1)), dim=1))
        return self.softmax(dec_output), dec_hidden.squeeze(0)

'''
实现注意力机制：
 Bahdanau Additive Attention:
 score(dec_input, enc_output)=v * tanh(w1* dec_input + w2* enc_output)
'''
class Attention(nn.Module):
    def __init__(self, emb_size, hidden_size, device):
        super(Attention, self).__init__()
        self.device = device
        self.attn = nn.Linear(hidden_size * 2 + emb_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, dec_input, enc_output, text_lengths):
        # enc_output = [batch_size, seq_len, hidden_size*2]
        # dec_input = [batch_size, embed_size]
        seq_len = enc_output.shape[1]
        dec_input = dec_input.repeat(1, seq_len, 1)  # dec_input扩维度：[batch_size, seq_len, embed_size]
        x = torch.tanh(self.attn(torch.cat([enc_output, dec_input], dim=2)))  # [batch_size, seq_len, hidden_size]
        attention = self.v(x)
        attention = attention.squeeze(-1)  # [batch_size, seq_len, ]
        max_len = enc_output.shape[1]
        # 对每个句长做掩码：不计入填充字段mask = [batch_size, seq_len]
        mask = torch.arange(max_len).expand(text_lengths.shape[0], max_len) >= text_lengths.unsqueeze(1)
        attention.masked_fill_(mask.to(self.device), float('-inf'))
        return self.softmax(attention)  # [batch, seq_len]


'''
整合S2S模型
'''
class Seq2seq(nn.Module):
    def __init__(self, config, vocab_size):
        super(Seq2seq, self).__init__()
        self.embed_layer = nn.Embedding(vocab_size, config.emb_size, padding_idx=0)
        self.attention = Attention(config.emb_size, config.hidden_size, config.device)  # 嵌入层
        self.encoder = Encoder(self.embed_layer, vocab_size, config.emb_size, config.hidden_size)
        self.decoder = Decoder(self.embed_layer, vocab_size, config.emb_size, config.hidden_size, self.attention)

    def forward(self, src, tgt, src_lengths, teacher_forcing_ratio=0.5):
        # src = 送入一批文本数据[batch_size, seq_len]
        # tgt = 送入一批摘要数据[batch_size, tgt_len]
        batch_size = tgt.shape[0]
        tgt_len = tgt.shape[1]
        vocab_size = self.decoder.vocab_size   # 词表长度
        # [batch_size, seq_len, hidden_size*2]  [batch_size, hidden_size]
        enc_output, s = self.encoder(src, src_lengths)

        dec_input = tgt[:, 0]  # 取BOS作为解码器输入[batch_size, ]
        dec_outputs = torch.zeros(batch_size, tgt_len, vocab_size)  # 最终输出全0初始化

        # 对decoder输出迭代对应摘要步长
        for t in range(tgt_len-1):
            #[batch_size, vocab_size], [batch_size, hidden_size]
            dec_output, s = self.decoder(dec_input, s, enc_output, src_lengths)
            dec_outputs[:, t, :] = dec_output
            # teacher-Forcing机制，使得最终生成的输出结果不只依赖于 decoder 的输入
            # 文本采样方法使用：Greedy Search
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = dec_output.argmax(1)
            dec_input = tgt[:, t] if teacher_force else top1
        return dec_outputs