# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/6/17


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms


# Step 0: Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# Step 1: Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Step 2: Create Network and Initialize
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))    # 和nn.Relu功能一样
        x = self.fc2(x)
        return x
model = NN(input_size,num_classes).to(device)

def test_NN():
    x = torch.randn(batch_size, input_size)
    print(model(x).shape)  # out: (64,10)


# Step 3: Load Data
train_dataset = datasets.MNIST(root = 'dataset/', train = True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True)   # 生成一个iterator
test_dataset = datasets.MNIST(root = 'dataset/', train = False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle=True)   # 生成一个iterator

# Step 4: Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Step 5: Train Net
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        data = data.reshape(data.shape[0], -1)
        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        # backward
        optimizer.zero_grad()
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新参数

# Step 6: Evaluate
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.reshape(x.shape[0], -1)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    model.train()  # 切换回训练模式
    accuracy = float(num_correct) / num_samples
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

check_accuracy(train_loader,model)
check_accuracy(test_loader, model)
