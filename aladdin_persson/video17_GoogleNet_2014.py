# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/6/20



"""
GoogleNet 是第一个提出并使用 Inception 模块的卷积神经网络架构, 使用多种kernel_size同时提取特征
<go deeper with convolutions> 2014
Inception 模块示意图：
                   输入
                  / | \
    1x1 卷积 <-> 3x3 卷积 <-> 5x5 卷积 <-> 3x3 最大池化
                  \ | /
                   合并
                   输出

GoogleNet/Inception 论文: Going Deeper with Convolutions (arXiv, 2014)
Inception v2/v3 论文: Rethinking the Inception Architecture for Computer Vision (arXiv, 2015)
Inception v4/Inception-ResNet 论文: Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning (arXiv,

引入新概念： pointwise convolution(1x1 convolutions)  channel单个像素的mlp,  一般用于调整channel数量
注意： 在训练的时候，每隔几层都会接一个分类任务，提高鲁棒性
特征融合技巧: channels间concate
"""
import torch
import torch.nn as nn
from utils import CscModule

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batch_norm(self.conv(x)))

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, b1_conv11, b2_conv11, b2_conv33, b3_conv11, b3_conv55, b4_conv11):
        super(InceptionBlock, self).__init__()
        self.branch1 = ConvBlock(in_channels, b1_conv11, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, b2_conv11, kernel_size=1),
            ConvBlock(b2_conv11, b2_conv33, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, b3_conv11, kernel_size=1),
            ConvBlock(b3_conv11, b3_conv55, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, b4_conv11, kernel_size=1),
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)

class GoogLeNet(CscModule):
    def __init__(self, in_channels=3, num_classes=1000):
        super(GoogLeNet, self).__init__()
        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=64, kernel_size=(7, 7),
                               stride=(2, 2), padding=(3, 3))
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = ConvBlock(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    model = GoogLeNet()
    y = model(x)
    print(model)
    print("Output shape:", y.shape)
    model.print_num_parameters()
    model.compute_flops((3, 224, 224))
    """
    Total number of parameters: 7008824
    FLOPS: 1.59 GMac, Parameters: 7.01 M
    """