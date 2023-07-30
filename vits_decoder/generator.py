import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

from vits.snake import SnakeBeta
from .nsf import SourceModuleHnNSF
from .bigv import AMPBlock
from jax.nn.initializers import normal as normal_init
from jax.nn.initializers import constant as constant_init
from vits.weightnorm import WeightNormConvTranspose
class SpeakerAdapter(nn.Module):
    speaker_dim : int
    adapter_dim : int
    epsilon : int = 1e-5
    def setup(self):
        self.W_scale = nn.Dense(features=self.adapter_dim,kernel_init=constant_init(0.),bias_init=constant_init(1.),dtype=jnp.float32)
        self.W_bias = nn.Dense(features=self.adapter_dim,kernel_init=constant_init(0.),bias_init=constant_init(0.),dtype=jnp.float32)


    def __call__(self, x, speaker_embedding):
        x = x.transpose(0,2,1)
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        std = jnp.sqrt(var + self.epsilon)
        y = (x - mean) / std
        scale = self.W_scale(speaker_embedding)
        bias = self.W_bias(speaker_embedding)
        y *= jnp.expand_dims(scale,1)
        y += jnp.expand_dims(bias,1)
        y = y.transpose(0,2,1)
        return y


class Generator(nn.Module):
    hp:tuple
    # this is our main BigVGAN model. Applies anti-aliased periodic activation for resblocks.
    def setup(self):
        self.num_kernels = len(self.hp.gen.resblock_kernel_sizes)
        self.num_upsamples = len(self.hp.gen.upsample_rates)
        # speaker adaper, 256 should change by what speaker encoder you use
        #self.adapter = SpeakerAdapter(self.hp.vits.spk_dim, self.hp.gen.upsample_input)
        self.adapter = SpeakerAdapter(self.hp.vits.spk_dim, self.hp.gen.upsample_input)
        # pre conv
        self.conv_pre = nn.Conv(features=self.hp.gen.upsample_initial_channel, kernel_size=[7], strides=[1],dtype=jnp.float32,bias_init=nn.initializers.normal())
        # nsf
        # self.f0_upsamp = nn.Upsample(
        #     scale_factor=np.prod(hp.gen.upsample_rates))
        self.scale_factor = np.prod(self.hp.gen.upsample_rates)
        self.m_source = SourceModuleHnNSF(sampling_rate=self.hp.data.sampling_rate)
        noise_convs = []
        # transposed conv-based upsamplers. does not apply anti-aliasing
        ups = []
        for i, (u, k) in enumerate(zip(self.hp.gen.upsample_rates, self.hp.gen.upsample_kernel_sizes)):
            # print(f'ups: {i} {k}, {u}, {(k - u) // 2}')
            # base
            ups.append(
                    WeightNormConvTranspose(
                        self.hp.gen.upsample_initial_channel // (2 ** (i + 1)),
                        (k,),
                        (u,))
                )
            # nsf
            if i + 1 < len(self.hp.gen.upsample_rates):
                stride_f0 = np.prod(self.hp.gen.upsample_rates[i + 1:])
                stride_f0 = int(stride_f0)
                noise_convs.append(
                    nn.Conv(
                        features=self.hp.gen.upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=[stride_f0 * 2],
                        strides=[stride_f0],
                        dtype=jnp.float32,bias_init=nn.initializers.normal()
                    )
                )
            else:
                noise_convs.append(
                    nn.Conv(features=self.hp.gen.upsample_initial_channel //
                           (2 ** (i + 1)), kernel_size=[1],dtype=jnp.float32,bias_init=nn.initializers.normal())
                )

        # residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        resblocks = []
        for i in range(len(ups)):
            ch = self.hp.gen.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(self.hp.gen.resblock_kernel_sizes, self.hp.gen.resblock_dilation_sizes):
                resblocks.append(AMPBlock(ch, k, d))

        # post conv
        self.conv_post = nn.Conv(features=1, kernel_size=[7], strides=1 , use_bias=False,dtype=jnp.float32)
        self.activation_post = SnakeBeta(ch)
        # weight initialization
        self.ups = ups
        self.noise_convs = noise_convs
        self.resblocks = resblocks

    def __call__(self, spk, x, f0,train=True):
        rng = self.make_rng('rnorms')
        x_key , rng = jax.random.split(rng)
        x = x + jax.random.normal(x_key,x.shape)
        # adapter
        x = self.adapter(x, spk)
        # nsf
        f0 = f0[:, None]
        B, H, W = f0.shape
        f0 = jax.image.resize(f0, shape=(B, H, W * self.scale_factor), method='nearest').transpose(0,2,1)
        har_source = self.m_source(f0,rng)
        har_source = har_source.transpose(0,2,1)
        x = self.conv_pre(x.transpose(0,2,1)).transpose(0,2,1)
        x = x * nn.tanh(nn.softplus(x))
        for i in range(self.num_upsamples):
            #x = nn.leaky_relu(x, 0.1)
            # upsampling
            x = self.ups[i](x.transpose(0,2,1)).transpose(0,2,1)
            # nsf
            x_source = self.noise_convs[i](har_source.transpose(0,2,1)).transpose(0,2,1)
            x = x + x_source
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x,train=train)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x,train=train)
            x = xs / self.num_kernels
        # post conv
        
        #x = nn.leaky_relu(x)
        x = self.activation_post(x)
        x = self.conv_post(x.transpose(0,2,1)).transpose(0,2,1)
        x = nn.tanh(x) 
        return x

    def eval(self, inference=False):
        super(Generator, self).eval()
        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def pitch2source(self, f0):
        f0 = f0[:, None]
        f0 = self.f0_upsamp(f0).transpose(1, 2)  # [1,len,1]
        har_source = self.m_source(f0)
        har_source = har_source.transpose(1, 2)  # [1,1,len]
        return har_source

    def source2wav(self, audio):
        MAX_WAV_VALUE = 32768.0
        audio = audio.squeeze()
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
        audio = audio.short()
        return audio.cpu().detach().numpy()

    def inference(self, spk, x, har_source):
        # adapter
        x = self.adapter(x, spk)
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = nn.functional.leaky_relu(x, 0.1)
            # upsampling
            x = self.ups[i](x)
            # nsf
            x_source = self.noise_convs[i](har_source)
            x = x + x_source
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # post conv
        x = nn.functional.leaky_relu(x)
        x = self.conv_post(x)
        x = nn.tanh(x)
        return x
