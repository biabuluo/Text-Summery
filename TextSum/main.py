import pandas as pd

path = 'LCSTS_small/'
data_train_txt = []
with open(path+'train.src.txt', encoding='utf-8', errors='ignore') as f:
    for text in f.readlines():
        data_train_txt.append(text.strip())

data_train_sum = []
with open(path+'train.tgt.txt', encoding='utf-8', errors='ignore') as f:
    for text in f.readlines():
        data_train_sum.append(text.strip())

data_train = pd.DataFrame({'txt':data_train_txt, 'sum':data_train_sum})
data_train.to_csv('data.csv', index=0)
print(data_train)