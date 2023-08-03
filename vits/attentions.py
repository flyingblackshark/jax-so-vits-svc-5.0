# import copy
# import math
import numpy as np
# import torch
# from torch import nn
# from torch.nn import functional as F
import jax.numpy as jnp
from flax import linen as nn
from vits import commons
from functools import partial
import flax
import jax
from jax.nn.initializers import normal as normal_init

class MultiHeadAttention(nn.Module):
    channels:int
    out_channels:int
    n_heads:int
    p_dropout:float=0.0
    window_size:int =None
    heads_share:bool=True
    @nn.compact
    def __call__(self, x, c, attn_mask=None,train=True):
        k_channels = self.channels // self.n_heads
        n_heads_rel = 1 if self.heads_share else self.n_heads
        rel_stddev = 1./jnp.sqrt(k_channels)
        emb_rel_k = self.param("emb_rel_k",nn.initializers.normal(rel_stddev),[n_heads_rel, self.window_size * 2 + 1, k_channels]) 
            
        emb_rel_v = self.param("emb_rel_v",nn.initializers.normal(rel_stddev),[n_heads_rel, self.window_size * 2 + 1, k_channels])



        q = nn.Conv(self.channels, [1],kernel_init=nn.initializers.xavier_uniform(),dtype=jnp.float32)(x.transpose(0,2,1)).transpose(0,2,1)
        k = nn.Conv(self.channels, [1],kernel_init=nn.initializers.xavier_uniform(),dtype=jnp.float32)(c.transpose(0,2,1)).transpose(0,2,1)
        v = nn.Conv(self.channels, [1],kernel_init=nn.initializers.xavier_uniform(),dtype=jnp.float32)(c.transpose(0,2,1)).transpose(0,2,1)
        x, attn = self.attention(q, k, v,emb_rel_k,emb_rel_v,k_channels, mask=attn_mask,train=train)

        x = nn.Conv(self.out_channels, [1],kernel_init=nn.initializers.xavier_uniform(),dtype=jnp.float32)(x.transpose(0,2,1)).transpose(0,2,1)
        return x

    def attention(self, query, key, value, emb_rel_k,emb_rel_v,k_channels,mask=None,train=True):
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s, t_t = (*key.shape, query.shape[2])
        query = jnp.reshape(query,(b, self.n_heads, k_channels, t_t)).transpose(0,1,3, 2)
        key = jnp.reshape(key,(b, self.n_heads, k_channels, t_s)).transpose(0,1,3, 2)
        value = jnp.reshape(value,(b, self.n_heads, k_channels, t_s)).transpose(0,1,3, 2)

        scores = jnp.matmul(query / jnp.sqrt(k_channels), jnp.swapaxes(key,-2, -1))
        if self.window_size is not None:
            assert (
                t_s == t_t
            ), "Relative attention is only available for self-attention."
            key_relative_embeddings = self._get_relative_embeddings(emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(
                query / jnp.sqrt(k_channels), key_relative_embeddings
            )
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local
        if mask is not None:
            scores=jnp.where(mask,scores,(-1e4))
        p_attn = nn.softmax(scores, axis=-1)  # [b, n_h, t_t, t_s]
        p_attn = nn.Dropout(self.p_dropout,deterministic=not train)(p_attn)
        output = jnp.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(
                emb_rel_v, t_s
            )
            output = output + self._matmul_with_relative_values(
                relative_weights, value_relative_embeddings
            )
        output = (
            jnp.reshape(output.transpose(0,1,3,2),(b, d, t_t))
        )  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        """
        x: [b, h, l, m]
        y: [h or 1, m, d]
        ret: [b, h, l, d]
        """
        ret = jnp.matmul(x, jnp.expand_dims(y,0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        """
        x: [b, h, l, d]
        y: [h or 1, m, d]
        ret: [b, h, l, m]
        """
        ret = jnp.matmul(x, jnp.swapaxes(jnp.expand_dims(y,0),-2, -1))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length):
        max_relative_position = 2 * self.window_size + 1
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = jnp.pad(
                relative_embeddings,[[0, 0], [pad_length, pad_length], [0, 0]]
            )
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[
            :, slice_start_position:slice_end_position
        ]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        batch, heads, length, _ = x.shape
        # Concat columns of pad to shift from relative to absolute indexing.
        x = jnp.pad(x, [[0, 0], [0, 0], [0, 0], [0, 1]])

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = jnp.reshape(x,[batch, heads, length * 2 * length])
        x_flat = jnp.pad( x_flat, [[0, 0], [0, 0], [0, length - 1]])
        

        # Reshape and slice out the padded elements.
        x_final = jnp.reshape(x_flat,[batch, heads, length + 1, 2 * length - 1])[
            :, :, :length, length - 1 :
        ]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = x.shape
        # padd along column
        x = jnp.pad(
            x, [[0, 0], [0, 0], [0, 0], [0, length - 1]]
        )
        x_flat = jnp.reshape(x,[batch, heads, length**2 + length * (length - 1)])
        # add 0's in the beginning that will skew the elements after reshape
        x_flat = jnp.pad(x_flat, [[0, 0], [0, 0], [length, 0]])
        x_final = jnp.reshape(x_flat,[batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final


class Encoder(nn.Module):
    hidden_channels:int
    filter_channels:int
    n_heads:int
    n_layers:int
    kernel_size:int = 1
    p_dropout:float = 0.0
    window_size:int = 4

    def setup(self):
        self.drop = nn.Dropout(self.p_dropout)
        attn_layers = []
        norm_layers_1 = []
        ffn_layers = []
        norm_layers_2 = []
        for i in range(self.n_layers):
            attn_layers.append(
                nn.SelfAttention(self.n_heads,qkv_features=self.hidden_channels,out_features=self.hidden_channels,dropout_rate=self.p_dropout)
            )
            norm_layers_1.append(nn.LayerNorm())
            ffn_layers.append(
                FFN(
                    self.hidden_channels,
                    self.filter_channels,
                    self.kernel_size,
                    p_dropout=self.p_dropout
                )
            )
            norm_layers_2.append(nn.LayerNorm())
        self.attn_layers = attn_layers
        self.norm_layers_1 = norm_layers_1
        self.ffn_layers = ffn_layers
        self.norm_layers_2 = norm_layers_2

    def __call__(self, x, x_mask,train=True):
        attn_mask = jnp.expand_dims(x_mask,2) * jnp.expand_dims(x_mask,-1)
        x = jnp.where(x_mask,x,0)
        
        for i in range(self.n_layers):
            #y = self.attn_layers[i](x, x, attn_mask,train=train)
            y = self.attn_layers[i](x.transpose(0,2,1), mask=attn_mask,deterministic=not train).transpose(0,2,1)
            y = self.drop(y,deterministic=not train)
            x = self.norm_layers_1[i]((x + y).transpose(0,2,1)).transpose(0,2,1)

            y = self.ffn_layers[i](x, x_mask,train=train)
            y = self.drop(y,deterministic=not train)
            x = self.norm_layers_2[i]((x + y).transpose(0,2,1)).transpose(0,2,1)

        x = jnp.where(x_mask,x,0)
        return x


class FFN(nn.Module):
    out_channels:int
    filter_channels:int
    kernel_size:int
    p_dropout:float=0.0

    def setup(self):
        self.conv_1 = nn.Conv(self.filter_channels, [self.kernel_size],dtype=jnp.float32,kernel_init=nn.initializers.normal())
        self.conv_2 = nn.Conv(self.out_channels, [self.kernel_size],dtype=jnp.float32,kernel_init=nn.initializers.normal())
        self.drop = nn.Dropout(self.p_dropout)

    def __call__(self, x, x_mask,train=True):
        x = jnp.where(x_mask,x,0)
        x = self.conv_1(x.transpose(0,2,1)).transpose(0,2,1)
        x = nn.gelu(x)
        x = self.drop(x.transpose(0,2,1),deterministic=not train).transpose(0,2,1)
        x = jnp.where(x_mask,x,0)
        x = self.conv_2(x.transpose(0,2,1)).transpose(0,2,1)
        x = jnp.where(x_mask,x,0)
        return x
