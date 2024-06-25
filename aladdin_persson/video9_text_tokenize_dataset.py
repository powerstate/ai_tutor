# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/6/18


"""
https://www.youtube.com/watch?v=9sHcLvVXsns&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=10
text dataset
视频里面是一个多模态的例子， (一张图片,一句话)
最后生成一个batch的x和y
y需要padding
"""

# text to numerical values
# 1. word2vec
# 2. create dateset for word sequence
# 3. setup padding for the sentence(all examples should be the same seq_len
# 4. setup dataloader


import os
import pandas as pd
import spacy   # for tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms

# download with: python -m spacy download en
spacy_eng = spacy.load('en')

class Vocabulary:
    """ 词典: 词频法 """
    def __init__(self, freq_threshold):
        # PAD: padding token 用于填充和补齐
        # SOS: start of the sentence
        # EOS: end of the sentence
        # UNK: unknown token
        # freq_threshold： 如果词频不到freq_threshold会被变成<UNK>
        # itos: index to string
        # stoi: string to index
        self.itos = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"}
        self.stoi = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        # text: "I love peanuts" -> ["i","love", "peanuts"]
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        """
        sentence_list: list of sentences
        sentence: 原始数据
        """
        frequencies = {}
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word]+=1
                # 达到freq，更新stoi和itos
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx +=1

    def numericalize(self, text):
        """ 把文本转换成 list of number """
        tokenized_text = self.tokenizer_eng(text)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]


class FlickerDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform = None, freq_threshold= 5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)   # [(xxxx.jpg, sentence)...]
        self.transform = None

        # get img. caption columns
        self.imgs = self.df['image']
        self.captions = self.df['caption']     # 图片注释

        # initialize vocabulary and build vocab
        self.vovab = Vocabulary(freq_threshold)
        self.vovab.build_vocabulary(self.captions.to_list())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """ 这里并没有进行padding """
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vovab.stoi["<SOS>"]]
        numericalized_caption += self.vovab.numericalize(caption)
        numericalized_caption.append(self.vovab.stoi["<EOS>"])
        return img, torch.tensor(numericalized_caption)

class MyCollate:
    """ 用于在dataloader里面对齐数据 """
    def __init__(self, pad_idx):
        # 一般pad_idx就是0
        self.pad_idx = pad_idx

    def __call__(self, batch):
        """ batch: list of images tensors"""
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim = 0)  # [batch_size, C, H, W]
        # 从 batch 中提取每个目标（标签）
        targets = [item[1] for item in batch]
        # 将目标序列填充到相同的长度，生成一个 [batch_size, max_seq_length] 的张量
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)  # number array
        return imgs, targets

def get_loader(
        root_folder,
        annotation_file,
        transform,
        num_workers = 8,
        batch_size = 32,
        shuffle = True,
        pin_memory = True,
):
    """ 这里会进行padding """
    dataset = FlickerDataset(root_folder, annotation_file, transform= transform)
    pad_idx = dataset.vovab.stoi['<PAD>']
    loader = DataLoader(
        dataset = dataset,
        batch_size=batch_size,
        num_workers = num_workers,   # 并行取出一个batch
        shuffle = shuffle,
        pin_memory = pin_memory,   # 数据传输到GPU加速用的
        collate_fn = MyCollate(pad_idx=pad_idx)
    )
    return loader

def main():
    transform = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ]
    )
    dataloader = get_loader('flickr8k/images/', annotation_file = 'flickr8k/captions.txt', transform=transform)
    for idx, (imgs, captions) in enumerate(dataloader):
        """ 取出一个batch """
        print(imgs.shape)
        print(captions.shape)
if __name__ == '__main__':
    main()
