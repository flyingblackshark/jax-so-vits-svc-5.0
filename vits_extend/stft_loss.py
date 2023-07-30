# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based Loss modules."""

# import torch
# import torch.nn.functional as F
import jax.numpy as jnp
from flax import linen as nn
from vits import commons
import jax
import optax
import numpy as np
import scipy
#from vits.losses import l1_loss
def stft(x, fft_size, hop_size, win_length):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = jax.scipy.signal.stft(x,nfft=fft_size, noverlap=win_length-hop_size, nperseg=win_length,padded=False,boundary=None)

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    hann_win = scipy.signal.get_window('hann',fft_size)
    scale = np.sqrt(1.0/hann_win.sum()**2)
    x_stft = x_stft[2]/scale
    real = jnp.real(x_stft)
    imag = jnp.imag(x_stft)
    return jnp.sqrt(jnp.clip(a=(real**2+imag**2),a_min=1e-7)).transpose(0,2,1)


class SpectralConvergengeLoss(nn.Module):
    """Spectral convergence loss module."""

    # def __init__(self):
    #     """Initilize spectral convergence loss module."""
    #     super(SpectralConvergengeLoss, self).__init__()

    def __call__(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return jnp.sqrt(jnp.sum(jnp.square(y_mag - x_mag))) / jnp.sqrt(jnp.sum(jnp.square(y_mag)))


class LogSTFTMagnitudeLoss(nn.Module):
    """Log STFT magnitude loss module."""

    # def __init__(self):
    #     """Initilize los STFT magnitude loss module."""
    #     super(LogSTFTMagnitudeLoss, self).__init__()

    def __call__(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        #return optax.l2_loss(jnp.log(y_mag), jnp.log(x_mag))
        return jnp.mean(jnp.abs(jnp.log(y_mag) - jnp.log(x_mag)))


class STFTLoss(nn.Module):
    """STFT loss module."""
    def __init__(self, fft_size=1024, shift_size=120, win_length=600):
        """Initialize STFT loss module."""
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        #self.window = getattr(torch, window)(win_length).to(device)
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def __call__(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """Multi resolution STFT loss module."""
    #resolutions:tuple
    def __init__(self,resolutions):
        """Initialize Multi resolution STFT loss module.
        Args:
            resolutions (list): List of (FFT size, hop size, window length).
            window (str): Window function type.
        """
        #super(MultiResolutionSTFTLoss, self).__init__()
        self.stft_losses = []
        for fs, ss, wl in resolutions:
            self.stft_losses.append(STFTLoss(fs, ss, wl))

    def __call__(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l

        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss
