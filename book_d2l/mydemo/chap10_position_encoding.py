# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/6/3

import torch
from torch import nn
from d2l import torch as d2l

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len = 1000):
        """
        :param num_hiddens: 源论文的d_model
        :param dropout: 略
        :param max_len: seq最大长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 通常seq的特征都是 (n_batch,n_seq, n_features)
        self.P = torch.zeros((1, max_len, num_hiddens))  # 初始化空间, 1是batch_size_dim
        X = torch.arange(max_len).reshape(-1, 1) / torch.pow(10000,
                                                             torch.arange(0, num_hiddens, 2) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.eval()
X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(torch.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in torch.arange(6, 10)])