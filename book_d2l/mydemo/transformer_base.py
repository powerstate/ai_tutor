# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/6/12


"""
    用pytorch复现<Attention is all you need>, 源论文
    WMT 2014 英文翻译德语任务
    8个GPU上训练了3.5天, 并行训练非常快

    tensorflow_code:
    https://github.com/tensorflow/tensor2tensor

    部件：
    1. multi-headed self-attention(mhsa)： 参数里面可能带mask，可能不带
    2. Positional Encoding: 编码和解码的PE是独立的？
    3. add_norm: 残差链接后layernorm
    4. FFN: pointwise mlp 每个位置上的前向dense网络
    3. Embedding Layer
    4. Encoder Layer
    5. Encoder Block：6个完全一样结构不同参数的，encoder块，每块里面有两个sub_layer(multi-headed self-attention,FFN)
    6. Decoder Layer
    7. Decoder Block: 也是6个

    ps： decoder和encoder是通过cross attention来实现链接的
        N个每个decoder提供query，然后在同一个encoder提供的k，v上进行查询

    Q&A
    1. 在每个位置上做laynorm的时候， gamma和beta要学n_seq个参数对出来，还是每层只需要一个
    GPT：每个层归一化组件在模型中的每个层只学习一组gamma和beta 参数，所有位置共享。
    2. 在翻译任务里面，一个batch是多个翻译对，每个batch，encoder和decoder的最大seq_len是怎么确定的，
        一次训练用个固定seq_len，还是每个batch的max_seq_len是不一样的
    GPT：2种方式都有，一种是每个batch一个长度叫《动态padding》，还有一整个训练是一个固定长度，第一种跟有效
    3. 源论文中的Q，K，V 是乘以W之后的状态么？
    GPT：是
    4. ScaledDotProductAttention, masked部分换非常大的负数，保证softmax基本为0
    5. position-wise feed forward 每个位置上共享参数么
    GPT：是， 每层用同一个参数，并且是双层mlp，每层用RELU激活
    6. Decoder的position encoding和Encoder的position encoding 都是从0号位置开始么， 还是decoder接着encoder
    GPT：无论Encoder还是Decoder，位置编码都是为每个位置生成的，并且都是从0开始
    7. position encoding在GPT模型里面是学出来的，还是按源论文来设计的，BERT是学出来的
    GPT：BERT和GPT都是学出来的，GPT和BERT在使用学习到的位置编码基本类似，但有区别
    8. 英语翻译德语的任务种， token集是共享的么？ 英文翻译中文嗯，也能这样么？
    GPT：对，中文一般不这样，标准的GPT模型主要是以英语为中心的，但通过适当的训绑数据和词表调整，它也能被扩展到处理多语言环境。对于真正的多语言支持，通常需要特别设计的多语言版本
"""

import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_, maxlen, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0).detach()  # 持久化并避免梯度更新

    def forward(self, x):
        """
        output: [batch_size, seq_len, d_model]
        """
        return x + self.encoding[:, :x.size(1)]


if __name__ == '__main__':
    """ 测试前向传播 """
    # 参数
    d_model = 512
    N = 6
    d_ff = 2048
    n_heads = 8
    d_k = 512/8
    d_v = 512/8
    dropout = 0.1
    label_smoothing = 0.1





