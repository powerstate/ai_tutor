# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/6/6

from torch import nn
import torch

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super().__init__()
        # 初始化尺度参数gamma
        self.gamma = nn.Parameter(torch.ones(d_model))
        # 初始化偏差参数beta
        self.beta = nn.Parameter(torch.zeros(d_model))
        # 设置一个小常数，防止除0
        self.eps = eps

    def forward(self, x):
        # 计算均值
        mean = x.mean(-1, keepdim=True)
        # 计算方差，unbiased=False时，方差的计算使用n而不是n-1做分母
        var = x.var(-1, unbiased=False, keepdim=True)

        # 归一化计算
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out