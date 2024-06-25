# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/5/14



import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from d2l import torch as d2l
d2l.use_svg_display()
root_dir = r'C:\ctemp\aidata\fashion_mnist'

to_tensor = transforms.ToTensor()

# mnist_train和mnist_test
mnist_train = torchvision.datasets.FashionMNIST(
    root = root_dir, train= True, transform=to_tensor, download=True
)
mnist_test = torchvision.datasets.FashionMNIST(
    root = root_dir, train= False, transform=to_tensor, download=True
)


def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

batch_size = 256

workers = 4
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim = True)
    return X_exp/partition

# 一层mlp模型io测试
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

d2l.train_ch6(net, train_iter, )