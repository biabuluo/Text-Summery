# -*-coding:utf-8 -*-

"""
# File       : train
# Time       ：2024/2/1 13:05
# Author     ：chenyu
# version    ：python 3.8
# Description：训练模型定义函数
"""
import datetime
from torch.nn.utils import clip_grad_norm_
import numpy as np
import torch
from rouge import *


def print_logger():
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(now_time + "==========" * 8)

# 模型验证过程
def evaluate(model, val_dataloader, loss_fn, device):
    val_loss = []
    model.eval()
    with torch.no_grad():
        val_loss = []
        for i, (text, text_len, title_in, title_out) in enumerate(val_dataloader):
            text = text.to(device)
            title_in = title_in.to(device)
            title_out = title_out.to(device)
            title_pred = model(text, title_in, text_len)
            loss = loss_fn(title_pred.transpose(1, 2).to(device), title_out)
            val_loss.append(loss.item())
    return np.mean(val_loss)



def train(model, train_dataloader, val_dataloader, loss_fn, optimizer, config):
    model = model.to(config.device)
    min_val_loss = float('inf')
    model.train()
    print_logger()
    print("训练开始......")
    train_losses = []
    val_losses = []
    for epoch in range(config.epochs):
        total_loss = 0
        for i, (text, text_len, title_in, title_out) in enumerate(train_dataloader):
            model.train()
            text = text.to(config.device)
            title_in = title_in.to(config.device)
            title_out = title_out.to(config.device)
            optimizer.zero_grad()
            title_pred = model(text, title_in, text_len)
            # 计算 cross entropy loss
            loss = loss_fn(title_pred.transpose(1, 2).to(config.device), title_out)
            total_loss += loss.item()
            loss.backward()
            # 梯度截断
            clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            print('ok!')

        avg_train_loss = total_loss / len(train_dataloader)
        # 每个 epoch 结束，验证模型的精度
        avg_val_loss = evaluate(model, val_dataloader, loss_fn, config.device)
        print_logger()
        print(
            f'epoch: {epoch + 1}/{config.epochs}, '
            f'training loss: {avg_train_loss:.4f}, '
            f'validation loss: {avg_val_loss:.4f}, ')
        if epoch == 0 or avg_val_loss < min_val_loss:
            model_path = config.saved_model_path + 'model_S2S_attention.pt'
            # 保存模型
            torch.save(model.state_dict(), model_path)

            print(f'The model has been saved for epoch {epoch + 1}')
            min_val_loss = avg_val_loss

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
    return train_losses, val_losses
