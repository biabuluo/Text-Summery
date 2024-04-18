# -*-coding:utf-8 -*-

"""
# File       : predict
# Time       ：2024/2/19 11:07
# Author     ：chenyu
# version    ：python 3.8
# Description：摘要生成测试
"""
import torch
from data_utils import replace_oovs, PAD_TOKEN, EOS_TOKEN, BOS_TOKEN, UNK_TOKEN
import pickle as pkl
from config import Configuration
from vocab import Vocab
from model2 import PGN

# 模型预测过程
def predict(model, vocab, text, device, max_len=35):
    # 预测的长度大于 max_len 或者遇到 EOS_TOKEN 时停止
    model.eval()
    dec_words = []
    with torch.no_grad():
        # 处理输入
        src, oovs = vocab.convert_text_to_ids(text)
        src_lengths = torch.tensor([len(src)])
        src = torch.tensor(src).reshape(1, -1)
        src_copy = replace_oovs(src, vocab, device)
        enc_output, prev_hidden = model.encoder(src_copy, src_lengths)
        # Decoder 的第一个输入为 EOS_TOKEN
        dec_input = torch.tensor([vocab[BOS_TOKEN]]).to(device)
        # 依次处理每个时间步的 decoder 过程
        # 初始化coverage_vector [batch_size, seq_len]
        coverage_vector = torch.zeros_like(src, dtype=torch.float32).to(device)
        for t in range(max_len):
            dec_output, prev_hidden, attention_weights, p_gen, coverage_vector= model.decoder(dec_input, prev_hidden, enc_output, src_lengths, coverage_vector)
            final_distribution = model.get_final_distribution(src, p_gen, dec_output, attention_weights, len(oovs))
            dec_output = final_distribution.argmax(-1)
            token_id = dec_output.item()
            # 对 token_id 进行解码，转换成单词
            # 遇到 EOS_TOKEN 时停止
            if dec_output.item() == vocab[EOS_TOKEN]:
                dec_words.append(EOS_TOKEN)
                break
            # token_id 在 vocab 里面，直接输出
            elif token_id < len(vocab):
                dec_words.append(vocab.idx2token[token_id])
            # token_id 在 oovs 里面，输入 oovs 对应的该单词。oovs 来源于原文输入。
            elif token_id < len(vocab) + len(oovs):
                dec_words.append(oovs[token_id - len(vocab)])
            # 其他情况，输入 UNK_TOKEN
            else:
                dec_words.append(UNK_TOKEN)
            # 将 decoder output 作为下一个时刻的 decoder input，并将其中的 oovs 替换成 UNK_TOKEN
            dec_input = replace_oovs(dec_output, vocab, device)
    return dec_words


# 获取词表
config = Configuration()
vocab = pkl.load(open(config.saved_vocab_path, 'rb'))
# print(vocab.idx2token[49998])



# 加载模型
model = PGN(vocab, config)
model.load_state_dict(torch.load('saved_model/model_PGN_withoutCoverage(1).pt', map_location=torch.device('cpu')))

text = '今天 有传 在 北京 某 小区 ， 一 光头 明星 因 吸毒 被捕 的 消息 。 下午 北京警方 官方 微博 发布 声明 通报情况 ， 证实 该 明星 为 李代沫 。 李代沫 伙同 另外 6 人 ， 于 17 日晚 在 北京 朝阳区 三里屯 某 小区 的 暂住地 内 吸食毒品 ， 6 人 全部 被 警方 抓获 ， 且 当事人 对 犯案 实施 供认不讳 。'
text = text.strip().split()
dec_words = predict(model, vocab, text, config.device)
print(''.join(dec_words))

