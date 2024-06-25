# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/5/30


"""
LSTM系列：
    需要更加注意的放到 更新门update gate  Z
    可以遗忘的放到 遗忘门reset gate  R
Gate: 和隐藏状态一样长度的向量
视频讲得很好
"""
import torch
from torch import nn
from d2l import torch as d2l


vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = d2l.RNNModel(gru_layer, len(vocab))
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)


# TODO： 试试xlstm的效果