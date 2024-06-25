# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/6/20

"""
    图片增强
"""

import torch
from torchvision import transforms
from torchvision.utils import save_image
from video8_image_dataset import CatsAndDogsDataset

# my_transforms = transforms.ToTensor()
my_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.RandomCrop((224,224)),  # 随机剪裁
    transforms.ColorJitter(brightness=0.5), # 随机改变亮度，对比度等
    transforms.RandomHorizontalFlip(p=0.5),   # 水平翻转
    transforms.RandomVerticalFlip(p=0.5),   # 垂直翻转
    transforms.RandomRotation(degrees=0.5),   # 随机旋转45度
    transforms.ToTensor(),
    transforms.Normalize(mean=[0,0,0], std=[1,1,1]),
    transforms.RandomGrayscale(p=0.5)
])
dataset = CatsAndDogsDataset(csv_file = 'cats_dogs.csv', root_dir = 'cats_dogs_resized', transform=my_transforms)

img_num=0
for img, label in dataset:
    save_image(img, 'img'+str(img_num)+'.png')
    img_num+=1