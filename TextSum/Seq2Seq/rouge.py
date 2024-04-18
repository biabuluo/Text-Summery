from rouge_chinese import Rouge

rouge = Rouge()

def rebuild(vocab, text):
    decoded_text = vocab.convert_ids_to_tokens(text)
    decoded_text = [i for i in decoded_text if (i != '<PAD>')]  # 去掉PAD
    decoded_text = ' '.join(decoded_text)
    return decoded_text

def rebuild_batch(vocab, encoded_texts, encoded_sums):
    # [batch_size, ids]
    decoded_texts = []
    decoded_sums = []
    for text in encoded_texts:
        decoded_text = rebuild(vocab, text)
        decoded_texts.append(decoded_text)
    for sum in encoded_sums.argmax(-1):
        decoded_sum = rebuild(vocab, sum)
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
    for hy, refer in zip(batch_hy, batch_refer):
        score = rouge_eval(hy, refer)
        rouge_1.append(score["rouge-1"])
        rouge_2.append(score["rouge-2"])
        rouge_l.append(score["rouge-l"])
    return rouge_1, rouge_2, rouge_l
