# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/4/29


# from aiquant.dataset.h5dataset import H5DataSet
from aiquant.ai_apps.signal_process.lib import wav_to_signal, dct,plot_spectrogram
import h5py
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import torchaudio
import torch
from aiquant.ai_apps.signal_process.lib import SingalProcessor

with h5py.File(r"C:\ctemp\test\crypto_1min.h5") as h5r:
    dset = h5r['data']
    close = dset[:,0,0]
log_close = np.log(close)
rets = log_close[1:]-log_close[:-1]
rets = rets[~np.isnan(rets)]
log_close = log_close[~np.isnan(log_close)]
signal = log_close



signal = SingalProcessor()

waveform, sr = wav_to_signal(r"C:\data\LibriSpeech\dev-clean\2412\153948\2412-153948-0000.flac")
spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=7200, hop_length=50)
spectrogram_torch = spectrogram_transform(torch.Tensor(log_close))
plot_spectrogram(spectrogram_torch[:500,:].numpy())
# plot_spectrogram(dct(spectrogram_torch.numpy(), type=2, axis=0, norm='ortho'))


mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=1000, n_fft=7200, hop_length=512, n_mels=128)
mel_spectrogram = mel_spectrogram_transform(torch.Tensor(log_close))  # 计算梅尔频谱图
# plot_spectrogram(mel_spectrogram.numpy())
plot_spectrogram(dct(mel_spectrogram.numpy(), type=2, axis=0, norm='ortho'))
#
# mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=13, melkwargs={'n_fft': 2048, 'hop_length': 512, 'n_mels': 128})
# mfcc = mfcc_transform(waveform)  # 计算MFCC

# spectrogram = get_spectrogram(signal, n_window=300, stride=50)
# mel_spectrogram = get_mel_spectrogram(signal, sr)
# plot_spectrogram(dct(mel_spectrogram.numpy(), type=2, axis=0, norm='ortho'))