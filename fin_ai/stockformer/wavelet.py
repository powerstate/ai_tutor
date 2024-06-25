# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/6/24



import pywt
import numpy as np
import matplotlib.pyplot as plt

# 生成一个示例信号
t = np.linspace(0, 1, 400, endpoint=False)
signal = np.sin(50 * 2 * np.pi * t) + np.sin(120 * 2 * np.pi * t)

# 执行DWT
coeffs = pywt.wavedec(signal, 'db1', level=6)

# 绘制原始信号
plt.figure(figsize=(12, 6))
plt.subplot(7, 1, 1)
plt.plot(signal)
plt.title('Original Signal')

# 绘制分解的信号
for i in range(1, 7):
    plt.subplot(7, 1, i + 1)
    plt.plot(coeffs[i])
    plt.title(f'Detail Coefficients Level {i}')

plt.tight_layout()
plt.show()