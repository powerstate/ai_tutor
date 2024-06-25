# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/6/20


"""
<Gradient-Based Learning Applied to Document Recognition (1998)>
输入层：32x32 灰度图像。
第一个卷积层：使用 6 个 5x5 的卷积核，生成 6 个 28x28 的特征图。
第一个池化层：平均池化，池化窗口为 2x2，生成 6 个 14x14 的特征图。
第二个卷积层：使用 16 个 5x5 的卷积核，生成 16 个 10x10 的特征图。
第二个池化层：平均池化，池化窗口为 2x2，生成 16 个 5x5 的特征图。
第三个卷积层：使用 120 个 5x5 的卷积核，生成 120 个 1x1 的特征图。
全连接层：具有 84 个神经元。
输出层：具有 10 个神经元，对应 10 个类别。
"""

import torch
from torch import nn
from thop import profile, clever_format
from utils import CscModule

class LeNet1998(CscModule):
    def __init__(self):
        super(LeNet1998, self).__init__()
        self.relu= nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5), stride=(1,1), padding = (0,0))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

x = torch.randn(64,1,32,32)
model = LeNet1998()
y = model(x)
print(model)
model.print_num_parameters()
model.compute_flops((1, 32, 32))