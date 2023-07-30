
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

import scipy
class DiscriminatorR(nn.Module):
    resolution:tuple
    hp:tuple
    def setup(self):
        self.LRELU_SLOPE = self.hp.mpd.lReLU_slope
        self.convs = [
            nn.Conv( 32, (3, 9)),
            nn.Conv( 32, (3, 9), strides=(1, 2)),
            nn.Conv( 32, (3, 9), strides=(1, 2)),
            nn.Conv( 32, (3, 9), strides=(1, 2)),
            nn.Conv( 32, (3, 3)),
        ]
        self.conv_post = nn.Conv( 1, (3, 3))
    
    def __call__(self, x,train=True):
        fmap = []
        x = self.spectrogram(x)
        x = jnp.expand_dims(x,1)
        for l in self.convs:
            x = l(x.transpose(0,2,3,1)).transpose(0,3,1,2)
            x = nn.leaky_relu(x, self.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x.transpose(0,2,3,1)).transpose(0,3,1,2)
        fmap.append(x)
        x = jnp.reshape(x, [x.shape[0],-1])
        return fmap, x

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = x.squeeze(1)
        hann_win = scipy.signal.get_window('hann',n_fft)
        scale = np.sqrt(1.0/hann_win.sum()**2)
        x = jax.scipy.signal.stft(x,nfft=n_fft, noverlap=win_length-hop_length, nperseg=win_length,padded=False,boundary=None) #[B, F, TT, 2]
        mag = jnp.abs(x[2]/scale)
        return mag

class MultiResolutionDiscriminator(nn.Module):
    hp:tuple
    def setup(self):
        self.resolutions = eval(self.hp.mrd.resolutions)
        self.discriminators = [DiscriminatorR(resolution,self.hp) for resolution in self.resolutions]
    def __call__(self, x,train=True):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x,train=train))

        return ret  # [(feat, score), (feat, score), (feat, score)]
