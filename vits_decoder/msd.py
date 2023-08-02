
import jax.numpy as jnp
import flax.linen as nn
from vits.weightnorm import WeightNormConv

class ScaleDiscriminator(nn.Module):
    def setup(self):
        self.convs = [
           WeightNormConv(16, [15], 1),
            WeightNormConv(64, [41], 4, feature_group_count =4),
            WeightNormConv( 256, [41], 4, feature_group_count =16),
            WeightNormConv( 1024, [41], 4, feature_group_count =64),
            WeightNormConv( 1024, [41], 4, feature_group_count =256),
            WeightNormConv( 1024, [5], 1),
        ]
       
        self.conv_post = WeightNormConv( 1, [3], 1)

    def __call__(self, x,train=True):
        fmap = []
        for l in self.convs:
            x = l(x.transpose(0,2,1)).transpose(0,2,1)
            x = nn.swish(x)
            fmap.append(x)
        x = self.conv_post(x.transpose(0,2,1)).transpose(0,2,1)
        fmap.append(x)
        x = jnp.reshape(x,[x.shape[0],-1])
        return [(fmap, x)]
