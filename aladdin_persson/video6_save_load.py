# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/6/18

import torch
from video4_rnn import RNN
from torch import optim

num_epochs = 10
model = RNN()
optimizer = optim.Adam(model.parameters(), lr = 0.001)


def save_checkpoint(state, filename):
    """ state 支持其他自定义的字典 """
    torch.save(state, filename)


def load_checkpoint(checkpoint):
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Train Network
for epoch in range(num_epochs):
    # 省略训练过程
    if epoch==2:
        checkpoint = {'state_dict':model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint, "my_checkpoint.path.tar")
        load_checkpoint(torch.load("my_checkpoint.path.tar"))

