# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/6/7


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

batch_size = 64
seq_length = 50
embed_size = 512
heads = 8
forward_expansion = 4
dropout_rate = 0.1

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask):
        # Input dimensions:
        # values, keys, queries: (batch_size, seq_len, embed_size)
        # mask: (batch_source, 1, seq_len) or (batch_size, 1, 1, seq_len)

        batch_size = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Reshape for multi-headed attention
        values = values.reshape(batch_size, value_len, self.heads, self.head_dim)
        keys = keys.reshape(batch_size, key_len, self.heads, self.head_dim)
        queries = queries.reshape(batch_size, query_len, self.heads, self.head_dim)

        # Linear transformations, maintain the same dimensions
        values = self.values(values)  # (batch_size, value_len, heads, head_dim)
        keys = self.keys(keys)        # (batch_size, key_len, heads, head_dim)
        queries = self.queries(queries)  # (batch_size, query_len, heads, head_dim)

        # Matrix multiplication between queries and keys, scale by square root of head dimension
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # (batch_size, heads, query_len, key_len)

        # Apply the mask to the attention scores
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-inf"))

        # Softmax normalization on the last dimension
        attention = torch.softmax(attention / (self.embed_size ** (1 / 2)), dim=3)  # (batch_size, heads, query_len, key_len)

        # Apply the attention scores to the values
        out = torch.einsum("nhqk,nlhd->nqhd", [attention, values])  # (batch_size, heads, query_len, head_dim)

        # Concatenate the heads
        out = out.reshape(batch_size, query_len, self.heads * self.head_dim)  # (batch_size, query_len, embed_size)

        # Final linear transformation
        out = self.fc_out(out)  # (batch_size, query_len, embed_size)
        return out

class EncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x: (batch_size, seq_length, embed_size)
        # mask: (batch_size, 1, seq_length, seq_length) 用于掩盖序列中不需要注意的部分
        attention = self.attention(x, x, x, mask)
        # attention: (batch_size, seq_length, embed_size)
        x = self.dropout(self.norm1(attention + x))  # 应用残差连接和归一化
        forward = self.feed_forward(x)
        # forward: (batch_size, seq_length, embed_size)
        out = self.dropout(self.norm2(forward + x))  # 再次应用残差连接和归一化
        return out


class DecoderLayer(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(DecoderLayer, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.encoder_decoder_attention = SelfAttention(embed_size, heads)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        # x: (batch_size, target_seq_length, embed_size)
        # enc_out: (batch_size, source_seq_length, embed_size)
        # src_mask: (batch_size, 1, 1, source_seq_length)
        # trg_mask: (batch_size, 1, target_seq_length, target_seq_length)

        attention = self.attention(x, x, x, trg_mask)
        # attention: (batch_size, target_seq_length, embed_size)
        x = self.dropout(self.norm1(attention + x))

        encoder_decoder_attention = self.encoder_decoder_attention(x, enc_out, enc_out, src_mask)
        # encoder_decoder_attention: (batch_size, target_seq_length, embed_size)
        x = self.dropout(self.norm2(encoder_decoder_attention + x))

        forward = self.feed_forward(x)
        # forward: (batch_size, target_seq_length, embed_size)
        out = self.dropout(self.norm3(forward + x))
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_length, embed_size)
        x = x + self.pe[:x.size(1), :].detach()
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size=256, num_layers=6, forward_expansion=4, heads=8, dropout=0.1, max_length=100):
        super(Transformer, self).__init__()
        self.encoder = nn.ModuleList([
            EncoderLayer(embed_size, heads, forward_expansion, dropout)
            for _ in range(num_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(embed_size, heads, forward_expansion, dropout)
            for _ in range(num_layers)
        ])
        self.src_pe = PositionalEncoding(embed_size, max_length)
        self.trg_pe = PositionalEncoding(embed_size, max_length)
        self.src_word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        no_peak_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool()
        trg_mask = trg_mask & no_peak_mask
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        src_embeddings = self.src_word_embedding(src) # (batch_size, src_len, embed_size)
        src_embeddings = self.src_pe(src_embeddings)
        trg_embeddings = self.trg_word_embedding(trg) # (batch_size, trg_len, embed_size)
        trg_embeddings = self.trg_pe(trg_embeddings)

        enc_src = src_embeddings
        for layer in self.encoder:
            enc_src = layer(enc_src, src_mask)  # (batch_size, src_len, embed_size)

        dec_trg = trg_embeddings
        for layer in self.decoder:
            dec_trg = layer(dec_trg, enc_src, src_mask, trg_mask)   # (batch_size, trg_len, embed_size)

        output = self.fc_out(dec_trg)     # (batch_size, trg_len, trg_vocab_size)
        return output