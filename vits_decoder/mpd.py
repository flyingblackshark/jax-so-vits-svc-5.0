
import jax.numpy as jnp
import flax.linen as nn
from vits.weightnorm import WeightNormConv2D
class DiscriminatorP(nn.Module):
    hp:tuple
    period:tuple
    def setup(self):
        self.LRELU_SLOPE = self.hp.mpd.lReLU_slope

        kernel_size = self.hp.mpd.kernel_size
        stride = self.hp.mpd.stride
      

        self.convs = [
            WeightNormConv2D(64, (kernel_size, 1), (stride, 1)),
            WeightNormConv2D( 128, (kernel_size, 1), (stride, 1)),
            WeightNormConv2D( 256, (kernel_size, 1), (stride, 1)),
            WeightNormConv2D( 512, (kernel_size, 1), (stride, 1)),
            WeightNormConv2D( 1024, (kernel_size, 1), 1),
        ]
        self.conv_post = WeightNormConv2D(1, (3, 1), 1)

    def __call__(self, x,train=True):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = jnp.pad(x, [(0,0),(0,0),(0, n_pad)], "reflect")
            t = t + n_pad
        x = jnp.reshape(x,[b, c, t // self.period, self.period])

        for l in self.convs:
            x = l(x.transpose(0,2,3,1)).transpose(0,3,1,2)
            x = nn.swish(x)
            fmap.append(x)
        x = self.conv_post(x.transpose(0,2,3,1)).transpose(0,3,1,2)
        fmap.append(x)
        x = jnp.reshape(x,[x.shape[0],-1])

        return fmap, x


class MultiPeriodDiscriminator(nn.Module):
    hp:tuple
    def setup(self):
        self.discriminators = [DiscriminatorP(self.hp, period) for period in self.hp.mpd.periods]
        
    def __call__(self, x,train=True):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x,train=train))

        return ret  # [(feat, score), (feat, score), (feat, score), (feat, score), (feat, score)]
