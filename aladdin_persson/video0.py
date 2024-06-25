# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/6/14

"""
https://www.youtube.com/watch?v=2S1dgHpqCdk&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz
Aladdin Persson
1 / 54
"""

import torch
import torch.nn as nn

print(torch.__version__)

class NeuralNetwork(nn.Module):
    pass


x = torch.rand(64, 10)
y = torch.rand(64, 10)
z = x+y
print(z.shape)