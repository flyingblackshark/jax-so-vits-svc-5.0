import jax.numpy as jnp
from flax import linen as nn
from vits import commons
from functools import partial
import flax
import jax
from typing import Tuple
from jax.nn.initializers import normal as normal_init
from vits.snake import SnakeBeta

class AMPBlock(nn.Module):
    channels:int
    kernel_size:int=3
    dilation:tuple=(1, 3, 5)
    def setup(self):
       
        self.convs1 =[
            nn.Conv(self.channels,[ self.kernel_size], 1, kernel_dilation=self.dilation[0],kernel_init=nn.initializers.normal()),
            nn.Conv( self.channels, [self.kernel_size], 1, kernel_dilation=self.dilation[1],kernel_init=nn.initializers.normal()),
            nn.Conv( self.channels, [self.kernel_size], 1, kernel_dilation=self.dilation[2],kernel_init=nn.initializers.normal())]
        self.convs2 = [
            nn.Conv( self.channels, [self.kernel_size], 1, kernel_dilation=1,kernel_init=nn.initializers.normal()),
            nn.Conv( self.channels, [self.kernel_size], 1, kernel_dilation=1,kernel_init=nn.initializers.normal()),
            nn.Conv(self.channels, [self.kernel_size], 1, kernel_dilation=1,kernel_init=nn.initializers.normal())
        ]
        # total number of conv layers
        self.num_layers = len(self.convs1) + len(self.convs2)
        self.activations = [SnakeBeta(self.channels) for _ in range(self.num_layers) ]
    def __call__(self, x,train=True):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2,a1,a2 in zip(self.convs1, self.convs2, acts1, acts2):
            #xt = nn.leaky_relu(x,0.1)
            xt = a1(x)
            xt = c1(xt.transpose(0,2,1)).transpose(0,2,1)
            #xt = nn.leaky_relu(xt,0.1)
            xt = a2(xt)
            xt = c2(xt.transpose(0,2,1)).transpose(0,2,1)
            x = xt + x
        return x