
import jax.numpy as jnp
import flax.linen as nn


class ScaleDiscriminator(nn.Module):
    def setup(self):
        self.convs = [
           nn.Conv(16, [15], 1,kernel_init=nn.initializers.normal()),
            nn.Conv(64, [41], 4, feature_group_count =4,kernel_init=nn.initializers.normal()),
            nn.Conv( 256, [41], 4, feature_group_count =16,kernel_init=nn.initializers.normal()),
            nn.Conv( 1024, [41], 4, feature_group_count =64,kernel_init=nn.initializers.normal()),
            nn.Conv( 1024, [41], 4, feature_group_count =256,kernel_init=nn.initializers.normal()),
            nn.Conv( 1024, [5], 1,kernel_init=nn.initializers.normal()),
        ]
       
        self.conv_post = nn.Conv( 1, [3], 1,kernel_init=nn.initializers.normal())

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
