import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence,Union,Any
class WeightNormConv(nn.Module):
    """
    apply weight standardization  https://arxiv.org/abs/1903.10520
    """ 
    features: int
    kernel_size: Sequence[int]
    strides: Union[None, int, Sequence[int]] = 1
    kernel_dilation : int = 1
    padding: Any = "SAME"
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32


    @nn.compact
    def __call__(self, x):
        """
        Applies a weight standardized convolution to the inputs.

        Args:
          inputs: input data with dimensions (batch, spatial_dims..., features).

        Returns:
          The convolved data.
        """
        x = x.astype(self.dtype)
        
        conv = nn.Conv(
            features=self.features, 
            kernel_size=self.kernel_size, 
            strides = self.strides,
            padding=self.padding, 
            kernel_dilation = self.kernel_dilation,
            dtype=self.dtype, 
            param_dtype = self.param_dtype,
            kernel_init=jax.nn.initializers.normal(0.01),bias_init=jax.nn.initializers.normal(0.01))
        
        kernel_init = lambda  rng, x: conv.init(rng,x)['params']['kernel']
        bias_init = lambda  rng, x: conv.init(rng,x)['params']['bias']
        
        # standardize kernel
        weight_v = self.param("weight_v", kernel_init, x)
        weight_g = self.param("weight_g", lambda _: jnp.linalg.norm(weight_v, axis=(0,1)))
        weight_v_norm = jnp.linalg.norm(weight_v, axis=(0,1))
        normed_weight_v = jnp.divide(weight_v, weight_v_norm)
        normed_kernel = jnp.multiply(normed_weight_v, weight_g)

        bias = self.param('bias',bias_init, x)

        return(conv.apply({'params': {'kernel': normed_kernel, 'bias': bias}},x))

class WeightNormConvTranspose(nn.Module):
    """
    apply weight standardization  https://arxiv.org/abs/1903.10520
    """ 
    features: int
    kernel_size: Sequence[int] 
    strides: Union[None, int, Sequence[int]] 
    padding: Any = "SAME"
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32


    @nn.compact
    def __call__(self, x):
        """
        Applies a weight standardized convolution to the inputs.

        Args:
          inputs: input data with dimensions (batch, spatial_dims..., features).

        Returns:
          The convolved data.
        """
        x = x.astype(self.dtype)
        
        conv = nn.ConvTranspose(
            features=self.features, 
            kernel_size=self.kernel_size, 
            strides = self.strides,
            padding=self.padding, 
            dtype=self.dtype, 
            param_dtype = self.param_dtype,
             kernel_init=jax.nn.initializers.normal(0.01),bias_init=jax.nn.initializers.normal(0.01))
        
        kernel_init = lambda  rng, x: conv.init(rng,x)['params']['kernel']
        bias_init = lambda  rng, x: conv.init(rng,x)['params']['bias']
        
        # standardize kernel
        weight_v = self.param("weight_v", kernel_init, x)
        weight_g = self.param("weight_g", lambda _: jnp.linalg.norm(weight_v, axis=(0,1)))
        weight_v_norm = jnp.linalg.norm(weight_v, axis=(0,1))
        normed_weight_v = jnp.divide(weight_v, weight_v_norm)
        normed_kernel = jnp.multiply(normed_weight_v, weight_g)

        bias = self.param('bias',bias_init, x)

        return(conv.apply({'params': {'kernel': normed_kernel, 'bias': bias}},x))