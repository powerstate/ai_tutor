# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/6/22


"""
    <EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks>

    Google使用AutoML搭配Neural Architechture Search（NAS）技术 搜索Compound Scaling的结构得出的
    核心的技术点： 通过调整（宽度width，深度depth，分辨率resolution）称为Compound Scaling的各种配比达到参数规模的最优性价比
    技术特点：
        1. Depthwise Convolution: MobileNet 等轻量级网络中常常使用用于减少参数量
            默认所有channel共享卷积核： nn.Conv2d中groups =1
            Depthwise Conv会让每个channel有独立的卷积核： n.Conv2d中groups =num_in_channels
            分组卷积: groups 的值大于1但小于 in_channels,表示将输入通道分成多个组，每组独立使用不同的卷积核。

        2. Pointwise Convolution:
        3. Inverted Residual Block（IRB）:
            Residual Block内部channel数是 宽->窄->宽 降维的逻辑。IRB则是 窄->宽->窄
        4. Linear Bottleneck
        6. LabelSmoothSoftmaxCE: 新的损失函数， 可以用更少的数据来收敛。避免overfit
    Q&A:
        1. 为什么叫Efficient： 网络小，快，且Acc高
        2. EfficientNet为什么更适合transfer learning
        3. 如何调整分辨率,transforms.Resize?
"""

import torch
from torch import nn
from math import ceil
from utils import CscModule

# 基础模型结构
base_model = [
    # expand_ratio, channels, repeats(num_blocks), stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

# EfficientNet不同版本的缩放参数
phi_values = {
    # tuple of: (phi_value, resolution, drop_rate)
    "b0": (0, 224, 0.2),
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                             padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 expand_ratio, reduction=4, survival_prob=0.8):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = survival_prob
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expanded = in_channels != hidden_dim
        reduced_dim = int(in_channels / reduction)

        # 扩展卷积，用于扩大输入通道数
        if self.expanded:
            self.expand_conv = CNNBlock(
                in_channels, hidden_dim, kernel_size=3, stride=1, padding=1,
            )

        # 深度卷积和SE模块
        self.conv = nn.Sequential(
            CNNBlock(
                hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim
            ),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # 如果输入通道数和输出通道数不匹配，或者步长不为1，需要调整输入的形状
        if in_channels != out_channels or stride != 1:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                           bias=False)
            self.residual_bn = nn.BatchNorm2d(out_channels)
        else:
            self.residual_conv = None

    def stochastic_depth(self, x):
        if not self.training:
            return x
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        binary_tensor = binary_tensor.float()
        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, x):
        identity = x  # 保存输入张量以用于残差连接

        if self.expanded:
            x = self.expand_conv(x)

        x = self.conv(x)

        if self.use_residual:
            if self.residual_conv is not None:
                identity = self.residual_bn(self.residual_conv(identity))
            return self.stochastic_depth(x) + identity
        else:
            return x

class EfficientNet(CscModule):
    def __init__(self, version, num_classes):
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = ceil(1280 * width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_feature(width_factor, depth_factor, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes)
        )

    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return width_factor, depth_factor, drop_rate

    def create_feature(self, width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)
        features = [CNNBlock(3, channels, 3, stride=2, padding=1)]
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4 * ceil(int(channels * width_factor) / 4)
            layers_repeats = ceil(repeats * depth_factor)
            for layer in range(layers_repeats):
                features.append(InvertedResidualBlock(
                    in_channels, out_channels, kernel_size,
                    stride=stride if layer == 0 else 1,
                    padding=kernel_size // 2, expand_ratio=expand_ratio
                ))
                in_channels = out_channels

        features.append(
            CNNBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0)
        )
        return nn.Sequential(*features)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x.view(x.shape[0], -1))


if __name__ == '__main__':
    version = 'b0'
    phi,res, drop_rate = phi_values[version]
    num_examples, num_classes = 4, 10
    x = torch.randn((num_examples, 3, res, res))
    model = EfficientNet(version=version, num_classes=num_classes)
    y = model(x)
    print(model)
    print("Output shape:", y.shape)

    model.print_num_parameters()
    model.compute_flops((3, res, res))