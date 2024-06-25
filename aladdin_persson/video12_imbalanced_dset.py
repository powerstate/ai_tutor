# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/6/20



"""
    不平衡数据集
    https://github.com/aladdinpersson/Machine-Learning-Collection
"""

import torch
from torchvision import datasets,transforms
from torch.utils.data import WeightedRandomSampler, DataLoader
import torch.nn as nn

# 方法一： oversampling, 增加小样本的数量
def get_loader(root_dir, batch_size):
    my_transforms =transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ]
    )
    dataset = datasets.ImageFolder(root=root_dir, transforms=my_transforms)
    class_weights = [1,50]
    sample_weights = [0] * len(dataset)

    for idx, (data, label) in enumerate(dataset):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight

    sampler = WeightedRandomSampler(sample_weights, num_samples=len*sample_weights, replacement=True)
    loader = DataLoader(dataset, batch_size=batch_size,sampler = sampler)
    return loader

# 方法二： class we:
# Idefighting, 增加小样本的loss权重
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1,50]))   # 第一个分类样本1倍，第二个50倍
