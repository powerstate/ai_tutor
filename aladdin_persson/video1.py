# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/6/14

"""
https://www.youtube.com/watch?v=x9JiIFvlUwk&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=2
Aladdin Persson 2 / 54
"""

import torch
import torch.nn as nn
import numpy as np

def demo0():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    my_tensor = torch.tensor([[1,2,3],[4,5,6]], dtype = torch.float32, device = 'cpu', requires_grad=True)
    print(my_tensor)
    print(my_tensor.device)
    print(my_tensor.shape)
    print(my_tensor.requires_grad)


    x = torch.empty(size = (3,3))
    x = torch.zeros((3,3))
    x = torch.rand(3,3)
    x = torch.ones(3,3)
    x = torch.eye(3,3)
    x = torch.arange(start=0, end=5, step=1)
    x = torch.linspace(start=0.1, end=1, steps= 10)
    x = torch.empty(size = (1,5)).normal_(mean = 0, std = 1)
    x = torch.empty(size = (1,5)).uniform_(0,1)
    x = torch.diag(torch.ones(3))

    my_tensor = torch.arange(4)
    # 类型变换
    print(my_tensor.bool())
    print(my_tensor.short())
    print(my_tensor.long())
    print(my_tensor.half())
    print(my_tensor.float())  # float32
    print(my_tensor.double())  # float64

    np_array = np.zeros((5, 5))
    my_tensor = torch.from_numpy(np_array)
    np_array_back = my_tensor.numpy()

def demo1():
    # 简单数学
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([9, 8, 7])

    z1 = torch.empty(3)
    torch.add(x, y, out = z1)
    z = x + y
    z = torch.true_divide(x,  y)   # 兼容类型

    t = torch.zeros(3)
    t.add_(x)   # func name后面带_ 都是inplace的
    t+=x    # inplace

    z =x.pow(2)
    z = x**2
    z = x>0

def demo2():
    # 矩阵乘法
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([9, 8, 7])

    x1= torch.rand((2,5))
    x2 = torch.rand((5, 3))
    x3 = torch.mm(x1,x2)
    x3 = x1.mm(x2)

    matrix = torch.rand(5,5)
    print(matrix.matrix_power(3))   # matrix.mm(matrix).mm(matrix)

    z = x*y   # dot product
    z = torch.dot(x, y)

    # batch mm
    batch_size = 32
    n = 10
    m = 20
    p = 30
    tensor1 = torch.rand((batch_size,n, m))
    tensor2 = torch.rand((batch_size, m, p))
    out_bmm = torch.bmm(tensor1, tensor2)   # (batch_size, 10, 30)

def demo3():
    # 广播
    x1 = torch.rand((5,5))
    x2 = torch.rand((1,5))
    z = x1-x2
    z = x1 ** x2

    sum_x = torch.sum(x, dim =0)
    values, indices = torch.max(x, dim=0)
    abs_x = torch.abs(x)
    z = torch.argmax(x, dim=0)
    z = torch.eq(x1,x2)
    sorted_y, indices = torch.sort(y, dim = 0, desending = False)
    z = torch.clamp(x, min =0, max = 10)    # 类似clip
    x = torch.tensor([1,0,1,1,1], dtype = torch.bool)
    z = torch.any(x)

def demo4():
    batch_size = 10
    features = 25
    x = torch.rand((batch_size,features))
    # Fancy Indexing
    x = torch.arange(10)
    indices = [2,5,6]
    print(x[indices])
    print(x[x.remainder(2)==0])   # remainder求余数
    print(torch.where(x>5, x, x*2))
    x.ndimension()    # number of dims
    x = numel()   # number of elements

def demo5():
    # reshaping
    # pytorch matrix内存摆放是行有限
    x = torch.arange(9)
    x33 = x.view(3,3)   # 要求x在内存里面是连续的， 更快
    x33 = x.reshape(3, 3)    # 不需要x内存联系， 自动调用.contiguous()
    x33_t = x33.t()   # transpose: 产生非连续张量
    x33_t1 = x33.transpose(dim0=0,dim1= 1)  # transpose: 产生非连续张量

    x1 = torch.rand(2, 5)
    x2 = torch.rand(2, 5)
    print(torch.cat((x1, x2), dim = 0).shape)
    print(torch.cat((x1, x2), dim=1).shape)

    x_flat = x.view(-1)  # 展平张量

    x = torch.tensor([1, 2, 3])  # 原始形状为 (3,)
    print(x.shape)  # 输出：torch.Size([3])
    # 增加维度为1 的维度
    y = x.unsqueeze(0)  # 在第0维增加一个尺寸
    print(y.shape)  # 输出：torch.Size([1, 3])
    # 去掉维度为1 的维度
    z = x.squeeze(1)  # 在第1维增加一个尺寸
    print(z.shape)  # 输出：torch.Size([3, 1])

    # permute:不会改变数据的底层顺序，它只改变张量的视图（view）,permute() 后的张量通常是非连续的
    x = torch.randn(2, 3, 4)  # 创建一个形状为 (2, 3, 4) 的随机张量
    print("Original shape:", x.shape)

    # 重新排列维度，将第二维放到最前面，然后是第三维，最后是第一维
    y = x.permute(1, 2, 0)   # 里面参数，代表维度额位置
    print("New shape:", y.shape)


if __name__ == '__main__':
    demo5()