import numpy as np
import jax.numpy as jnp
import flax.linen as nn


class GLU(nn.Module):
    dim:int
    @nn.compact
    def __call__(self, x):
        out, gate = jnp.split(x,2, axis=self.dim)
        return out * nn.sigmoid(gate)
    
class ConformerConvModule(nn.Module):
  output_dim : int = 256
  kernel_size : int = 31
  @nn.compact
  def __call__(self, x):
    x = nn.LayerNorm()(x)
    x = nn.Conv(self.output_dim * 4, [1])(x)
    x = GLU(dim=2)(x)
    x = nn.Conv(self.output_dim, kernel_size = [self.kernel_size])(x)
    x = nn.swish(x)
    x = nn.Conv(self.output_dim, kernel_size = [self.kernel_size])(x)
    return x

class EncoderBlock(nn.Module):
  emb_dim : int = 256
  n_heads : int = 8
  

  @nn.compact
  def __call__(self, input,mask=None):
    x = input
    x = nn.LayerNorm()(x)
    x = nn.SelfAttention(num_heads=self.n_heads,out_features=self.emb_dim)(x,mask=mask)
    x = x + input
    
    y = x
    y = ConformerConvModule(self.emb_dim)(y)
    y = y + x

    return y


class PCmer(nn.Module):
  dim_model : int = 256
  num_heads : int = 8
  num_layers : int = 8
  residual_dropout : float = 0.1
  attention_dropout : float = 0.1
  def setup(self):
    self.encoder_layers = [EncoderBlock(emb_dim=self.dim_model,n_heads=self.num_heads) for i in range(self.num_layers)]

  def __call__(self, phone , mask=None):

    for i in range(len(self.encoder_layers)):
      
      phone = self.encoder_layers[i](phone,mask)

    return phone

  