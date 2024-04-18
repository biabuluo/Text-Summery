from rouge_chinese import Rouge
from data_utils import reserved_tokens, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
rouge = Rouge()

'''
重建文本
'''
def rebuild(vocab, text, oovs):
    decoded_text = []
    for id in text:
        # token_id 在 vocab 里面，直接输出
        if id < len(vocab):
            decoded_text.append(vocab.idx2token[id])
        # token_id 在 oovs 里面，输入 oovs 对应的该单词。oovs 来源于原文输入。
        elif id < len(vocab) + len(oovs):
            decoded_text.append(oovs[id - len(vocab)])
        # 其他情况，输入 UNK_TOKEN
        else:
            decoded_text.append(UNK_TOKEN)
    decoded_text = [i for i in decoded_text if (i != '<PAD>')]  # 去掉PAD

    return ' '.join(decoded_text)


def rebuild_batch(vocab, encoded_texts, encoded_sums, oovs_list):
    # [batch_size, ids]
    decoded_texts = []
    decoded_sums = []
    for text, oovs in zip(encoded_texts, oovs_list):
        decoded_text = rebuild(vocab, text, oovs)
        decoded_texts.append(decoded_text)
    for sum, oovs in zip(encoded_sums.argmax(-1), oovs_list):
        decoded_sum = rebuild(vocab, sum, oovs)
        decoded_sums.append(decoded_sum)
    return decoded_texts, decoded_sums


def rouge_eval(hypothesis, reference):
    scores = rouge.get_scores(hypothesis, reference)
    return {
        "rouge-1": scores[0]["rouge-1"]["f"],
        "rouge-2": scores[0]["rouge-2"]["f"],
        "rouge-l": scores[0]["rouge-l"]["f"]
    }


def rouge_eval_batch(batch_hy, batch_refer):
    rouge_1 = []
    rouge_2 = []
    rouge_l = []
    # print(len(batch_hy), len(batch_refer))
    for hy, refer in zip(batch_hy, batch_refer):
        score = rouge_eval(hy, refer)
        rouge_1.append(score["rouge-1"])
        rouge_2.append(score["rouge-2"])
        rouge_l.append(score["rouge-l"])
    return rouge_1, rouge_2, rouge_l




