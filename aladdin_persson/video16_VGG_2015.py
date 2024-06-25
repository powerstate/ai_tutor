# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/6/20



"""
<Very Deep Convolutional Networks for Large-Scale Image Recognition> 2014.09
    作者提出了一系列深度卷积神经网络（VGG），其核心思想是使用非常小的（3×3）卷积核，并增加网络的深度来提高模型性能。
    这些网络在 ImageNet 大规模视觉识别挑战赛中表现出色，尤其是 VGG-16 和 VGG-19，
    这两种架构的参数量和计算复杂度都大大增加，但也带来了显著的性能提升。
VGG 的贡献和影响
    展示了深度的力量：VGG 系列模型展示了通过增加网络深度可以显著提升模型性能，这启发了后续更多深度网络的研究和应用，如 ResNet、Inception 等。
    小卷积核的优势：使用小的卷积核（3×3）堆叠起来，能够更有效地捕获复杂的特征，并且相对较小的卷积核参数量较少，有助于模型的训练和泛化。
    简单而有效的设计：VGG 的设计虽然简单，但其堆叠卷积层的策略证明了在复杂图像识别任务中的有效性，为后续的网络设计提供了灵感。
"""

from utils import CscModule
import torch
from torch import nn, optim
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

VGG16_struct = [64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M']

class VGG16(CscModule):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG16, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_layers = self.create_conv_layers(VGG16_struct)

        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architechture):
        layers = []
        in_channels = self.in_channels
        for x in architechture:
            if  type(x)==int:
                out_channels = x
                layers +=[nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=(3,3), stride=(1,1), padding = (1,1)),
                          nn.BatchNorm2d(x),
                          nn.ReLU()]
                in_channels = x
            elif x=='M':
                layers+=[nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
        return nn.Sequential(*layers)

# 输入[batch_size, n_channels,H,W]
x = torch.randn(1,3,224,224)
model = VGG16()
y = model(x)
print(model)
model.print_num_parameters()
model.compute_flops((3, 224, 224))
"""
Total number of parameters: 138365992
FLOPS: 15.55 GMac, Parameters: 138.37 M
"""