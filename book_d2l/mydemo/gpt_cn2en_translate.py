# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/6/7




"""
1. 数据预处理：加载和准备中文和英文翻译对，包括文本清洗、分词、构建词典等。
2. 模型配置：设置Transformer模型的参数。
3. 数据加载器：构建PyTorch的DataLoader来批量处理数据。
4. 模型训练：编写训练循环，包括损失函数和优化器的定义。
5. 模型评估：评估模型的翻译性能，包括BLEU分数的计算等。
"""


import spacy
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
from gpt_transformer_net import Transformer
import torch.optim as optim
from torchtext.datasets import Multi30k
from datasets import load_dataset

# Load data
train_data, val_data, test_data = Multi30k(split=('train', 'valid', 'test'), language_pair=('zn', 'en'))


# 加载Spacy语言模型
spacy_en = spacy.load('en_core_web_sm')
spacy_zh = spacy.load('zh_core_web_sm')


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_zh(text):
    return [tok.text for tok in spacy_zh.tokenizer(text)]

class TranslationDataset(Dataset):
    def __init__(self, path, src_lang, trg_lang):
        self.data = self.load_data(path)
        self.src_vocab = self.build_vocab([example[src_lang] for example in self.data], src_lang)
        self.trg_vocab = self.build_vocab([example[trg_lang] for example in self.data], trg_lang)

    def load_data(self, path):
        # 这里需要根据实际情况调整，假设每行是一个翻译对，中英文之间用\t分隔
        with open(path, encoding='utf-8') as file:
            lines = file.readlines()
            data = [{'zh': line.split('\t')[0], 'en': line.split('\t')[1].strip()} for line in lines]
        return data

    def build_vocab(self, texts, lang):
        counter = Counter()
        tokenizer = tokenize_zh if lang == 'zh' else tokenize_en
        for text in texts:
            counter.update(tokenizer(text))
        vocab = {word: idx + 2 for idx, word in enumerate(counter)}
        vocab['<pad>'] = 0
        vocab['<sos>'] = 1
        vocab['<eos>'] = 2
        return vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_sample = [self.src_vocab[word] for word in tokenize_zh(self.data[idx]['zh'])]
        trg_sample = [self.trg_vocab[word] for word in tokenize_en(self.data[idx]['en'])]
        return torch.tensor(src_sample), torch.tensor(trg_sample)

# 创建 dataset 实例，并设定数据文件路径和语言
dataset = TranslationDataset(path='data/train.txt', src_lang='zh', trg_lang='en')

# 获取词汇表大小和PAD索引
src_vocab_size = len(dataset.src_vocab)  # 源语言词汇表大小
trg_vocab_size = len(dataset.trg_vocab)  # 目标语言词汇表大小
src_pad_idx = dataset.src_vocab['<pad>']  # 源语言PAD索引
trg_pad_idx = dataset.trg_vocab['<pad>']  # 目标语言PAD索引

# 模型配置参数
embed_size = 512  # 嵌入层大小
num_layers = 6    # Transformer层数
forward_expansion = 4
heads = 8         # 注意力头的数量
dropout = 0.1     # Dropout率
max_length = 100  # 位置编码的最大长度

# 实例化模型
model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx,
                    embed_size, num_layers, forward_expansion, heads, dropout, max_length)

# 数据加载器
loader = DataLoader(dataset, batch_size=32, shuffle=True)


# 优化器与损失函数
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for batch_idx, (src, trg) in enumerate(loader):
        src = src.to(device)
        trg = trg.to(device)

        # 前向传播
        output = model(src, trg[:, :-1])
        output_dim = output.shape[-1]

        # 调整输出和目标的维度
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # 计算损失
        loss = criterion(output, trg)
        optimizer.zero_grad()

        # 反向传播
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train_epoch(model, loader, optimizer, criterion, device)
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}')

