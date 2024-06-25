# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/6/18

"""
https://www.youtube.com/watch?v=qaDe0qQZ5AQ&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=8
"""




import torchvision
from torch import nn

model = torchvision.models.vgg16(pretrained=True)
print(model)
# 里面包含3个模块，依次是features, avgpool, classifier

class Identity(nn.Module):
    """ 用于删除一层 """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

# 修改
model.avgpool = Identity()
# 如果想要冻住部分的参数
for param in model.parameters():
    param.requires_grad = False
model.classifier = nn.Linear(512, 10)   # 替换原有classifier
model.to(device)