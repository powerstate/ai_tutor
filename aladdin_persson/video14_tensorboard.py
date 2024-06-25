# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/6/20


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# 定义一个简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 创建模型、损失函数和优化器
model = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 创建 TensorBoard 的 SummaryWriter
writer = SummaryWriter('runs/simple_net_experiment')

# 模拟训练过程
for epoch in range(10):
    inputs = torch.randn(5, 10)  # 随机输入
    targets = torch.randn(5, 1)  # 随机目标

    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 将损失记录到 TensorBoard
    writer.add_scalar('Training Loss', loss.item(), epoch)

# 关闭 SummaryWriter
writer.close()