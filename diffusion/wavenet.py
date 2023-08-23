import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence,Any

class SinusoidalPosEmb(nn.Module):
    dim: int
    @nn.compact
    def __call__(self, x):
        half_dim = self.dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim, ) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        return emb
class ResidualBlock(nn.Module):
    encoder_hidden:int
    residual_channels:int
    dilation:int
    def setup(self):
        self.dilated_conv = nn.Conv(
            2 * self.residual_channels,
            kernel_size=[3],
            kernel_dilation=[self.dilation]
        )
        self.diffusion_projection = nn.Dense(self.residual_channels)
        self.conditioner_projection = nn.Conv(2 * self.residual_channels, [1])
        self.output_projection = nn.Conv(2 * self.residual_channels, [1])

    def __call__(self, x, conditioner, diffusion_step):
        diffusion_step = jnp.expand_dims(self.diffusion_projection(diffusion_step),-1)#.unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner.transpose(0,2,1)).transpose(0,2,1)
        y = x + diffusion_step

        y = self.dilated_conv(y.transpose(0,2,1)).transpose(0,2,1) + conditioner

        # Using torch.split instead of torch.chunk to avoid using onnx::Slice
        gate, filter = jnp.split(y, 2, axis=1)
        y = nn.sigmoid(gate) * nn.tanh(filter)

        y = self.output_projection(y.transpose(0,2,1)).transpose(0,2,1)

        # Using torch.split instead of torch.chunk to avoid using onnx::Slice
        residual, skip = jnp.split(y, 2, axis=1)
        return (x + residual) / jnp.sqrt(2.0), skip
    
class WaveNet(nn.Module):
    in_dims:int=128 
    n_layers:int=20
    n_chans:int=384
    n_hidden:int=256
    def setup(self):

        self.input_projection = nn.Conv(self.n_chans, [1])
        self.diffusion_embedding = SinusoidalPosEmb(self.n_chans)
        self.mlp = nn.Sequential(
            [nn.Dense(self.n_chans * 4),
            nn.swish,
            nn.Dense(self.n_chans)]
        )
        self.residual_layers = [
            ResidualBlock(
                encoder_hidden=self.n_hidden,
                residual_channels=self.n_chans,
                dilation=1
            )
            for i in range(self.n_layers)
        ]
        self.skip_projection = nn.Conv(self.n_chans, [1])
        self.output_projection = nn.Conv(self.in_dims, [1],kernel_init=jax.nn.initializers.zeros)

    def __call__(self, spec, diffusion_step, cond):
        """
        :param spec: [B, 1, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, M, T]
        :return:
        """
        x = spec#.squeeze(1)
        x = self.input_projection(x.transpose(0,2,1)).transpose(0,2,1)  # [B, residual_channel, T]

        x = nn.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)

        x = jnp.sum(jnp.stack(skip), axis=0) / jnp.sqrt(len(self.residual_layers))
        x = self.skip_projection(x.transpose(0,2,1)).transpose(0,2,1)
        x = nn.relu(x)
        x = self.output_projection(x.transpose(0,2,1)).transpose(0,2,1)  # [B, mel_bins, T]
        return x#[:, None, :, :]
