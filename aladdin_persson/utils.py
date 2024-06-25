# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/6/20


from torch import nn
from ptflops import get_model_complexity_info

class CscModule(nn.Module):
    def __init__(self):
        super(CscModule, self).__init__()

    def print_num_parameters(self):
        """
        打印模型中所有参数的个数
        """
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of parameters: {total_params}")

    def compute_flops(self, input_size):
        """
        计算模型的 FLOPS
        参数:
        - input_size: 输入张量的大小 (channels, height, width)
        """
        # 使用 ptflops 库计算 FLOPS 和参数数量
        flops, params = get_model_complexity_info(self, input_size, as_strings=True, print_per_layer_stat=False)
        print(f"FLOPS: {flops}, Parameters: {params}")
