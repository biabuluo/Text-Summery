# -*-coding:utf-8 -*-

"""
# File       : predict
# Time       ：2024/2/1 15:22
# Author     ：chenyu
# version    ：python 3.8
# Description：用来对测试集测试
"""
import torch

from Seq2Seq.config import Configuration
from data_utils import UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, load_data
import pickle as pkl
from rouge import *
import numpy as np

def predict(model, vocab, device, src, src_lengths, max_len=35):
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
    return ' '.join(dec_words)

def test(test_text, model, vocab, device):
    predict_sums = []
    for text in test_text:
        tokens = [token for token in text.split(' ')]
        ids = vocab.convert_tokens_to_ids(tokens)
        predict_sum = predict(model, vocab, device, ids, len(ids))
        predict_sums.append(predict_sum)
    # 写入文件
    with open('test_sum/test_tgt_predict.txt', 'w', encoding='utf-8') as f:
        f.writelines(predict_sums)


def test_eval(sum_real_path, sum_path):
    sums_real, sums_predict = load_data(sum_real_path, sum_path)
    rouge_1, rouge_2, rouge_l = rouge_eval_batch(sums_real, sums_predict)
    print("test_eval: ============== =============== ")
    print(
        f'rouge_1:{np.mean(rouge_1):.4f}, '
        f'rouge_2:{np.mean(rouge_2):.4f}, '
        f'rouge_l:{np.mean(rouge_l):.4f}')

if __name__ == '__main__':
    # 加载配置类
    config = Configuration()
    # 先加载数据
    test_text, test_title = load_data(config.processed_test_contents_path,
                                        config.processed_test_sums_path)
    # 加载词表
    vocab = pkl.load(open(config.saved_vocab_path, 'rb'))

    # 加载模型
    model = torch.load(config.saved_model_path + 'model.pt')

    # 测试
    test(test_text, model, vocab, config.device)

    # 评估
    test_eval(config.processed_test_sums_path, 'test_sum/test_tgt_predict.txt')


