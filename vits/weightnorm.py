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
    feature_group_count : int = 1
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
            feature_group_count=self.feature_group_count,
            dtype=self.dtype, 
            param_dtype = self.param_dtype,
            kernel_init=jax.nn.initializers.normal(0.01),
            bias_init=jax.nn.initializers.normal(0.01))
        
        #kernel_init = lambda  rng, x: conv.init(rng,x)['params']['kernel']
        weight_shape = (
            conv.features,
            x.shape[-1] // self.feature_group_count,
            conv.kernel_size[0],
        )
        weight_v = self.param("weight_v", jax.nn.initializers.he_normal(), weight_shape)    
        weight_g = self.param("weight_g", lambda _: jnp.linalg.norm(weight_v, axis=(0, 1))[None, None, :])
        bias = self.param("bias", jax.nn.initializers.zeros, (conv.features,))
        
        weight_v_norm = jnp.linalg.norm(weight_v, axis=(0, 1))[None, None, :]
        normed_weight_v = jnp.divide(weight_v, weight_v_norm)
        normed_kernel = jnp.multiply(normed_weight_v, weight_g)

        return(conv.apply({'params': {'kernel': normed_kernel.T, 'bias': bias}},x))
class WeightNormConv2D(nn.Module):
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
            kernel_init=jax.nn.initializers.normal(0.01),
            bias_init=jax.nn.initializers.normal(0.01))
        
        #kernel_init = lambda  rng, x: conv.init(rng,x)['params']['kernel']
        weight_shape = (
            conv.features,
            x.shape[-1],
            conv.kernel_size[1],
            conv.kernel_size[0],
            
        )
        def norm_2d(x):
            for i in range(x.shape[3]):
                x=x.at[:,:,:,i].set(jnp.sqrt(jnp.sum(jnp.square(x[:,:,:,i]))))
            return x                   
        weight_v = self.param("weight_v", jax.nn.initializers.he_normal(), weight_shape)    
        weight_g = self.param("weight_g", lambda _: norm_2d(weight_v))
        bias = self.param("bias", jax.nn.initializers.zeros, (conv.features,))
        
        weight_v_norm = norm_2d(weight_v)
        normed_weight_v = jnp.divide(weight_v, weight_v_norm)
        normed_kernel = jnp.multiply(normed_weight_v, weight_g)

        return(conv.apply({'params': {'kernel': normed_kernel.T, 'bias': bias}},x))
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
        
        weight_shape = (
            conv.features,
            x.shape[-1],
            conv.kernel_size[0],
        )
        weight_v = self.param("weight_v", jax.nn.initializers.he_normal(), weight_shape)
        weight_g = self.param("weight_g", lambda _: jnp.linalg.norm(weight_v, axis=(0, 1))[None, None, :])
        bias = self.param("bias", jax.nn.initializers.zeros, (conv.features,))
        
        weight_v_norm = jnp.linalg.norm(weight_v, axis=(0, 1))[None, None, :]
        normed_weight_v = jnp.divide(weight_v, weight_v_norm)
        normed_kernel = jnp.multiply(normed_weight_v, weight_g)

        return(conv.apply({'params': {'kernel': normed_kernel.T, 'bias': bias}},x))