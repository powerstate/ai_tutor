# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/6/18


"""
https://www.youtube.com/watch?v=ZoZHd0Zm3RY&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=9
可用于各种图片训练

"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io
from torchvision import transforms

# Load data efficiently
class CatsAndDogsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)   # 这里面放着y对应的id
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index,0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index,1]))
        return (image, y_label)


dataset = CatsAndDogsDataset(csv_file ='cat_dogs.csv', root_dir= 'cats_dogs_resized',
                             transform=transforms.ToTensor())
train_set, test_set = torch.utils.data.random_split(dataset, [20000,5000])
train_loader = DataLoader(dataset = train_set, batch_size=64, shuffle = True)
test_loader = DataLoader(dataset = test_set, batch_size=64, shuffle = True)