# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/6/6



"""
问gpt4：我想用transformer训练中文翻译英文的任务

pip install torch torchtext spacy
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm

"""


from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from torchtext.datasets import Multi30k


SRC_LANGUAGE = 'zh'
TGT_LANGUAGE = 'en'

# Tokenizers
src_tokenizer = get_tokenizer('spacy', language='zh_core_web_sm')
tgt_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

# Load data
train_data, val_data, test_data = Multi30k(split=('train', 'valid', 'test'), language_pair=('zn', 'en'))

# 假设 train_data 是你从文件加载的成对句子 [(chinese_sentence, english_sentence), ...]
def build_vocab(data, tokenizer):
    vocab = build_vocab_from_iterator(map(tokenizer, [pair[0] if tokenizer == src_tokenizer else pair[1] for pair in data]))
    vocab.set_default_index(vocab["<unk>"])
    vocab.append_token("<pad>")
    vocab.append_token("<bos>")
    vocab.append_token("<eos>")
    return vocab

src_vocab = build_vocab(train_data, src_tokenizer)
tgt_vocab = build_vocab(train_data, tgt_tokenizer)