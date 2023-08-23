import random

import jax
import jax.numpy as jnp
import os
import matplotlib.pyplot as plt
import torch
from jax import config
import math
import numpy as np
from PIL import Image


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * start
    beta_end = scale * end
    return jnp.linspace(beta_start, beta_end, timesteps)  # , dtype = jnp.float64


# def linear_beta_schedule(timesteps):
#     """
#     linear schedule, proposed in original ddpm paper
#     """
#     scale = 1000 / timesteps
#     beta_start = scale * 0.0015
#     beta_end = scale * 0.0195
#     return jnp.linspace(beta_start, beta_end, timesteps)  # , dtype = jnp.float64


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = jnp.linspace(0, timesteps, steps, ) / timesteps
    alphas_cumprod = jnp.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = jnp.linspace(0, timesteps, steps) / timesteps
    v_start = jax.nn.sigmoid(jnp.array(start / tau))
    v_end = jax.nn.sigmoid(jnp.array(end / tau))
    alphas_cumprod = (- jax.nn.sigmoid(((t * (end - start) + start) / tau)) + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0, 0.999)


if __name__ == "__main__":
    os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'
    beta = linear_beta_schedule(1000,end=0.01,start=1e-5)
    alphas = 1 - beta
    alphas_cumprod = jnp.cumprod(alphas)
    print(alphas_cumprod)
