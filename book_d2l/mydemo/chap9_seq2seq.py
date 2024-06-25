# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/5/30


"""
    encoder + decoder架构
    encoder提供最后的隐状态给decoder
    decoder 结合embedding（concate）

    预测句子好坏的指标BLEU：n-gram精度的求和。主要是翻译任务。
"""

"""
beam search 用与seq序列最后的生成
目标：用什么样的概率生成seq

"""