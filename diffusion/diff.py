import jax
import jax.numpy as jnp
import flax.linen as nn
from .pcmer import PCmer
import optax
import numpy as np
from .gaussian import Gaussian
from .wavenet import WaveNet
f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)


def f0_to_coarse(f0):
    #is_torch = isinstance(f0, jax.Tensor)
    f0_mel = 1127 * jnp.log(1 + f0 / 700)
    #f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1
    f0_mel = jnp.where(f0_mel>0,(f0_mel - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1,f0_mel)

    #f0_mel[f0_mel <= 1] = 1
    f0_mel = jnp.where(f0_mel<=1,1,f0_mel)
    #f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_mel = jnp.where(f0_mel > (f0_bin - 1),f0_bin - 1,f0_mel)
    f0_coarse = jnp.rint(f0_mel).astype(np.int32)
    # assert f0_coarse.max() <= 255 and f0_coarse.min(
    # ) >= 1, (f0_coarse.max(), f0_coarse.min())
    return f0_coarse

class Unit2MelPre(nn.Module):
    input_channel:int
    n_spk:int
    use_pitch_aug:bool=False
    out_dims:int=128
    n_layers:int=3
    n_chans:int=256
    n_hidden:int=256
    use_speaker_encoder:bool=False
    speaker_encoder_out_channels:int=256
    def setup(self):
        self.f0_embed = nn.Embed(256, self.n_chans)#nn.Dense( self.n_chans)
        self.volume_embed = nn.Dense(self.n_chans)
        if self.use_pitch_aug:
            self.aug_shift_embed = nn.Dense(self.n_chans, use_bias=False)
        else:
            self.aug_shift_embed = None

        if self.use_speaker_encoder:
            self.spk_embed = nn.Dense(self.n_chans, use_bias=False)
        else:
            if self.n_spk is not None and self.n_spk > 1:
                self.spk_embed = nn.Embed(self.n_spk, self.n_chans)

        # conv in stack
        # self.ppg_stack = nn.Sequential([
        #     nn.Conv(self.n_chans, [3]),
        #     nn.GroupNorm(num_groups=4),
        #     nn.leaky_relu,
        #     nn.Conv(self.n_chans, [3])])
        self.ppg_stack=nn.Dense(self.n_chans)
        
        # self.vec_stack = nn.Sequential([
        #     nn.Conv(self.n_chans, [3]),
        #     nn.GroupNorm(num_groups=4),
        #     nn.leaky_relu,
        #     nn.Conv(self.n_chans, [3])])

        self.vec_stack=nn.Dense(self.n_chans)



    def __call__(self, ppg , vec, f0, volume=None, spk_id=None, spk_mix_dict=None, aug_shift=None,
                gt_spec=None, infer=True, infer_speedup=10, method='ddim', k_step=None, use_tqdm=True,
                spk_emb=None, spk_emb_dict=None):

        '''
        input:
            B x n_frames x n_unit
        return:
            dict of B x n_frames x feat
        '''
        x = self.ppg_stack(ppg).transpose(0,2,1)
        v = self.vec_stack(vec).transpose(0,2,1)
        x = x + v + self.f0_embed(f0_to_coarse(f0)).transpose(0,2,1) #+ self.volume_embed(volume)
        x = x.transpose(0,2,1)
        # if self.use_speaker_encoder:
        #     if spk_mix_dict is not None:
        #         assert spk_emb_dict is not None
        #         for k, v in spk_mix_dict.items():
        #             spk_id_torch = spk_emb_dict[str(k)]
        #             spk_id_torch = np.tile(spk_id_torch, (len(units), 1))
        #             spk_id_torch = torch.from_numpy(spk_id_torch).float().to(units.device)
        #             x = x + v * self.spk_embed(spk_id_torch)
        #     else:
        #         x = x + self.spk_embed(spk_emb)
        # else:
        #     if self.n_spk is not None and self.n_spk > 1:
        #         if spk_mix_dict is not None:
        #             for k, v in spk_mix_dict.items():
        #                 spk_id_torch = torch.LongTensor(np.array([[k]])).to(units.device)
        #                 x = x + v * self.spk_embed(spk_id_torch - 1)
        #         else:
        #             x = x + self.spk_embed(spk_id - 1)
        # if self.aug_shift_embed is not None and aug_shift is not None:
        #     x = x + self.aug_shift_embed(aug_shift / 5)
       

        return x

