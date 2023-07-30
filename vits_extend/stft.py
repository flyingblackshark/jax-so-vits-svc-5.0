# MIT License
#
# Copyright (c) 2020 Jungil Kong
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import os
import random
# import torch
# import torch.utils.data
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
from typing import Optional
import scipy
class TacotronSTFT(nn.Module):

    def __init__(self, filter_length=512, hop_length=160, win_length=512,
                 n_mel_channels=80, sampling_rate=16000, mel_fmin=0.0,
                 mel_fmax=None):
        #uper(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.n_fft = filter_length
        self.hop_size = hop_length
        self.win_size = win_length
        self.fmin = mel_fmin
        self.fmax = mel_fmax
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)
        self.mel_basis = mel
    def linear_spectrogram(self, y):
        spec = jax.scipy.signal.stft(y,nfft=self.n_fft, noverlap=self.win_size-self.hop_size, nperseg=self.win_size,return_onesided=True,padded=False,boundary=None)    
        hann_win = scipy.signal.get_window('hann',self.n_fft)
        scale = np.sqrt(1.0/hann_win.sum()**2)
        return jnp.abs(spec[2]/scale)

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        hann_win = scipy.signal.get_window('hann',self.n_fft)
        scale = np.sqrt(1.0/hann_win.sum()**2)
        spec = jax.scipy.signal.stft(y, nfft=self.n_fft, noverlap=self.win_size-self.hop_size, nperseg=self.win_size,return_onesided=True,padded=False,boundary=None)
        spec = spec[2]/scale
        real = jnp.real(spec)
        imag = jnp.imag(spec)
        spec = jnp.sqrt(real**2+imag**2+(1e-9))
        spec = jnp.matmul(self.mel_basis, spec)
        spec = self.spectral_normalize_torch(spec)

        return spec

    def spectral_normalize_torch(self, magnitudes):
        output = self.dynamic_range_compression_torch(magnitudes)
        return output

    def dynamic_range_compression_torch(self, x, C=1, clip_val=1e-5):
        return jnp.log(jnp.clip(a=x, a_min=clip_val) * C)
