# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/6/6


"""
参考资料
《attention is all you need》简称源论文
d2l课程 - 李沐
https://mp.weixin.qq.com/s?__biz=Mzg2MzkwNDM0OQ==&mid=2247487465&idx=1&sn=16d5d0c6152c1b38cf037e01d23c217a&chksm=ce7038dbf907b1cdef29f1b1a7b5245ac990b2d1e9b7d4079a8e8fcdd96af98cfdee34840108&cur_album_id=3386373852905177096&scene=189&version=4.1.22.8031&platform=win&nwr_flag=1#wechat_redirect
https://mp.weixin.qq.com/s?__biz=Mzg2MzkwNDM0OQ==&mid=2247487502&idx=1&sn=d55683013bb2a3c2716a0f81054950b3&chksm=ce70273cf907ae2a0e3ce21f3f87111876e8e2b7991f1d65fbdb44a07448c33aed2c07655e44&mpshare=1&scene=1&srcid=0404bNqwTBvokvEmsVPMd1cx&sharer_shareinfo=587c71e9f8e683659371d0396ef5fc18&sharer_shareinfo_first=587c71e9f8e683659371d0396ef5fc18&version=4.1.22.8031&platform=win&nwr_flag=1#wechat_redirect
"""

"""
编码端：经过词向量层（Input Embedding）和位置编码层（Positional Encoding），得到最终输入，流经自注意力层（Multi-Head Attention）、残差和层归一化（Add&Norm）、前馈神经网络层（Feed Forward）、残差和层归一化（Add&Norm），得到编码端的输出（后续会和解码端进行交互）。
解码端：经过词向量层（Output Embedding）和位置编码层（Positional Encoding），得到最终输入，流经掩码自注意力层（Masked Multi-Head Attention，把当前词之后的词全部mask掉）、残差和层归一化（Add&Norm）、交互注意力层（Multi-Head Attention，把编码端的输出和解码端的信息进行交互，Q矩阵来自解码端，K、V矩阵来自编码端的输出）、残差和层归一化（Add&Norm）、前馈神经网络层（Feed Forward）、残差和层归一化（Add&Norm），得到解码端的输出。
注：编码端和解码端的输入不一定等长
"""

"""
输入Inputs维度是[batch size,sequence length]，经nn.Embedding，转换为计算机可以识别的Input Embedding，论文中每个词对应一个512维度的向量，
维度是[batch_size,sequence_length,embedding_dimmension]。batch size指的是句子数，
sequence length指的是输入的句子中最长的句子的字数，embedding_dimmension是词向量长度。
"""

import torch
from torch import nn
import math

class Embeddings(nn.Module):
    """
    类的初始化
    :param d_model: 源论文 词向量维度，512
    :param vocab: 当前语言的词表大小size
    """
    def __init__(self, vocab, d_model):

        super(Embeddings, self).__init__()
        # 调用nn.Embedding预定义层，获 得实例化词嵌入对象self.lut
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model  #表示词向量维度

    def forward(self, x):
        """
        Embedding层的前向传播
        参数x：为category正整数，输入给模型的单词文本通过此表映射后的one-hot向量
        x传给self.lut，得到形状为(batch_size, sequence_length, d_model)的张量，与self.d_model相乘，
        以保持不同维度间的方差一致性，及在训练过程中稳定梯度
        """
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """实现Positional Encoding功能"""
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        """
        位置编码器的初始化函数
        :param d_model: 词向量的维度，与输入序列的特征维度相同，原论文512
        :param dropout: 置零比率
        :param max_len: 句子最大长度,5000
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化一个位置编码矩阵
        # (1000,512)矩阵，保持每个位置的位置编码，一共5000个位置，每个位置用一个512维度向量来表示其位置编码
        pe = torch.zeros(max_len, d_model)
        # 偶数和奇数在公式上有一个共同部分，使用log函数把次方拿下来，方便计算
        # position表示的是字词在句子中的索引，如max_len是128，那么索引就是从0，1，2，...,127
        # 论文中d_model是512，2i符号中i从0取到255，那么2i对应取值就是0,2,4...510
        # (1000) -> (1000,1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算用于控制正余弦的系数，确保不同频率成分在d_model维空间内均匀分布
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 根据位置和div_term计算正弦和余弦值，分别赋值给pe的偶数列和奇数列
        pe[:, 0::2] = torch.sin(position * div_term)   # 从0开始到最后面，补长为2，其实代表的就是偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)   # 从1开始到最后面，补长为2，其实代表的就是奇数位置
        # 上面代码获取之后得到的pe:[max_len * d_model]
        # 下面这个代码之后得到的pe形状是：[1 * max_len * d_model]
        # 多增加1维，是为了适应batch_size
        # (1000, 512) -> (1, 1000, 512)
        pe = pe.unsqueeze(0)
        # 将计算好的位置编码矩阵注册为模块缓冲区（buffer），这意味着它将成为模块的一部分并随模型保存与加载，但不会被视为模型参数参与反向传播
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]  经过词向量的输入
        """
        x = x + self.pe[:, :x.size(1)].clone().detach()   # 经过词向量的输入与位置编码相加
        # x = x + self.pe[:, :x.shape[1], :].to(x.device)
        # Dropout层会按照设定的比例随机“丢弃”（置零）一部分位置编码与词向量相加后的元素，
        # 以此引入正则化效果，防止模型过拟合
        return self.dropout(x)

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, scale_factor, dropout=0.0):
        """ scale_factor就是math.sqrt(d) """
        super().__init__()
        self.scale_factor = scale_factor
        #dropout用于防止过拟合，在前向传播的过程中，让某个神经元的激活值以一定的概率停止工作
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # batch_size: 批量大小
        # len_q,len_k,len_v: 序列长度 在这里他们都相等
        # n_head: 多头注意力，论文中默认为8
        # d_k,d_v: k v 的dim(维度) 默认都是64
        # 此时q的shape为(batch_size, n_head, len_q, d_k) (batch_size, 8, len_q, 64)
        # 此时k的shape为(batch_size, n_head, len_k, d_k) (batch_size, 8, len_k, 64)
        # 此时v的shape为(batch_size, n_head, len_k, d_v) (batch_size, 8, len_k, 64)
        # q先除以self.scale_factor，再乘以k的转置(交换最后两个维度(这样才可以进行矩阵相乘))。
        # attn的shape为(batch_size, n_head, len_q, len_k)

        attn_scores = torch.matmul(q / self.scale_factor, k.transpose(2, 3))

        if mask is not None:
            """
            用-1e9代替0 -1e9是一个很大的负数 经过softmax之后接近0
            # 其一：去除掉各种padding在训练过程中的影响
            # 其二，将输入进行遮盖，避免decoder看到后面要预测的东西。（只用在decoder中）
            """
            attn_scores  = attn_scores.masked_fill(mask == 0, -1e9)
        # 先在attn的最后一个维度做softmax 再dropout 得到注意力分数
        attn_scores = self.dropout(torch.softmax(attn_scores, dim=-1))
        # 最后attn与v矩阵相乘
        # output的shape为(batch_size, 8, len_q, 64)
        output = torch.matmul(attn_scores, v)
        # 返回 output和注意力分数
        return output, attn_scores

class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        # 论文中这里的n_head, d_model, d_k, d_v分别默认为8, 512, 64, 64
        '''
        # q k v先经过不同的线性层，再用ScaledDotProductAttention，最后再经过一个线性层
        '''
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(scale_factor=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)  # 默认对最后一个维度初始化

    def forward(self, q, k, v, mask=None):
        # q, k, v初次输入为含位置信息的嵌入矩阵X，由于要堆叠N次，后面的输入则是上个多头的输出
        # q, k, v：batch_size * seq_num * d_model
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        # len_q, len_k, len_v 为输入的序列长度
        # batch_size为batch_size
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        # 用作残差连接
        residual = q
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        # q k v 分别经过一个线性层再改变维度
        # 由(batch_size, len_q, n_head*d_k) => (batch_size, len_q, n_head, d_k)
        # (batch_size, len_q, 8*64) => (batch_size, len_q, 8, 64)
        q = self.layer_norm(q)
        k = self.layer_norm(k)
        v = self.layer_norm(v)

        # 与q,k,v相关矩阵相乘，得到相应的q,k,v向量，d_model=n_head * d_k
        q = self.w_qs(q).view(batch_size, len_q, n_head, d_k)
        k = self.w_ks(k).view(batch_size, len_k, n_head, d_k)
        v = self.w_vs(v).view(batch_size, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        # 交换维度做attention
        # 由(batch_size, len_q, n_head, d_k) => (batch_size, n_head, len_q, d_k)
        # (batch_size, len_q, 8, 64) => (batch_size, 8, len_q, 64)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            # 为head增加一个维度
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        # 输出的q为Softmax(QK/d + (1-S)σ)V, attn 为QK/D
        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        # (batch_size, 8, len_k, 64) => (batch_size, len_k, 8, 64) => (batch_size, len_k, 512)
        q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        # 经过fc和dropout
        q = self.dropout(self.fc(q))
        # 残差连接 论文中的Add & Norm中的Add
        q += residual
        # 论文中的Add & Norm中的Norm
        q = self.layer_norm(q)
        # q的shape为(batch_size, len_q, 512)
        # attn的shape为(batch_size, n_head, len_q, len_k)
        return q, attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))

    def forward(self, inputs):                             # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual)   # [batch_size, seq_len, d_model]

# seq_q: [batch_size, seq_len] ,seq_k: [batch_size, seq_len]
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    # 扩展成多维度
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()                                     # 多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet()                                        # 前馈神经网络

    def forward(self, enc_inputs, enc_self_attn_mask):                                # enc_inputs: [batch_size, src_len, d_model]
        #输入3个enc_inputs分别与W_q、W_k、W_v相乘得到Q、K、V                          # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,    # enc_outputs: [batch_size, src_len, d_model],
                                               enc_self_attn_mask)                    # attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs = self.pos_ffn(enc_outputs)                                       # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

"""
编码器
"""
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.src_emb = nn.Embedding(src_vocab_size, d_model)                     # 把字转换字向量
        self.pos_emb = PositionalEncoding(d_model)                               # 加入位置信息
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):                                               # enc_inputs: [batch_size, src_len]
        # 1. 中文字索引进行Embedding，转换成512维度的字向量
        enc_outputs = self.src_emb(enc_inputs)                                   # enc_outputs: [batch_size, src_len, d_model]
        # 2. 在字向量上面加上位置信息
        enc_outputs = self.pos_emb(enc_outputs)                                  # enc_outputs: [batch_size, src_len, d_model]
        # 3. Mask掉句子中的占位符号
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)           # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_self_attns = []
        # 4. 通过6层的encoder（上一层的输出作为下一层的输入）
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)  # enc_outputs :   [batch_size, src_len, d_model],
                                                                                 # enc_self_attn : [batch_size, n_heads, src_len, src_len]
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

def get_attn_subsequence_mask(seq):                               # seq: [batch_size, tgt_len]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)          # 生成上三角矩阵,[batch_size, tgt_len, tgt_len]
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()  #  [batch_size, tgt_len, tgt_len]
    return subsequence_mask

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask): # dec_inputs: [batch_size, tgt_len, d_model]
                                                                                       # enc_outputs: [batch_size, src_len, d_model]
                                                                                       # dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
                                                                                       # dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs,
                                                 dec_inputs, dec_self_attn_mask)   # dec_outputs: [batch_size, tgt_len, d_model]
                                                                                   # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs,
                                                enc_outputs, dec_enc_attn_mask)    # dec_outputs: [batch_size, tgt_len, d_model]
                                                                                   # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs = self.pos_ffn(dec_outputs)                                    # dec_outputs: [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_len+1, d_model),freeze=True)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs): # dec_inputs : [batch_size x target_len]
        # 1. 英文字索引进行Embedding，转换成512维度的字向量，并在字向量上加上位置信息
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(torch.LongTensor([[5,1,2,3,4]]))
        # 2. Mask掉句子中的占位符号
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        # 3. 通过6层的decoder（上一层的输出作为下一层的输入）
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        # 编码器
        self.encoder = Encoder()
        # 解码器
        self.decoder = Decoder()
        # 解码器最后的分类器，分类器的输入d_model是解码层每个token的输出维度大小，需要将其转为词表大小，再计算softmax；计算哪个词出现的概率最大
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        #  Transformer的两个输入，一个是编码器的输入（源序列），一个是解码器的输入（目标序列）
        # 其中，enc_inputs的大小应该是 [batch_size, src_len] ;  dec_inputs的大小应该是 [batch_size, dec_inputs]

        """
        源数据输入到encoder之后得到 enc_outputs, enc_self_attns；
        enc_outputs是需要传给decoder的矩阵，表示源数据的表示特征
        enc_self_attns表示单词之间的相关性矩阵
        """
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)

        """
        decoder的输入数据包括三部分：
        1. encoder得到的表示特征enc_outputs、
        2. 解码器的输入dec_inputs（目标序列）、
        3. 以及enc_inputs
        """
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)

        """
        将decoder的输出映射到词表大小，最后进行softmax输出即可
        """
        dec_logits = self.projection(dec_outputs) # dec_logits : [batch_size x src_vocab_size x tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


if __name__ == '__main__':
    embedding = Embeddings(10, )