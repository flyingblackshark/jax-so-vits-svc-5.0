
import flax
from flax import linen as nn
from jax.nn.initializers import normal as normal_init
from flax.training import train_state
import jax.numpy as jnp
import jax
from jax import random
import numpy as np
import optax


from vits import attentions
from vits import commons
from vits import modules
from vits.modules_grl import SpeakerClassifier
from vits.utils import f0_to_coarse
from vits_decoder.generator import Generator


class TextEncoder(nn.Module):
    in_channels:int
    out_channels:int
    hidden_channels:int
    filter_channels:int
    n_heads:int
    n_layers:int
    kernel_size:int
    p_dropout:float
    def setup(self):
        self.pre = nn.Conv(features=self.hidden_channels, kernel_size=[5],dtype=jnp.float32,kernel_init=nn.initializers.normal(),bias_init=nn.initializers.normal())
        #self.hub = nn.Conv(features=self.hidden_channels, kernel_size=[5],dtype=jnp.float32,kernel_init=nn.initializers.normal())
        self.pit = nn.Embed(256, self.hidden_channels,dtype=jnp.float32)
        self.enc = attentions.Encoder(
            hidden_channels=self.hidden_channels,
            filter_channels=self.filter_channels,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            kernel_size=self.kernel_size,
            p_dropout=self.p_dropout)
        self.proj = nn.Conv(features=self.out_channels * 2, kernel_size=[1],dtype=jnp.float32,kernel_init=nn.initializers.normal(),bias_init=nn.initializers.normal())
    def __call__(self, x, x_lengths, f0,train=True):
        rng = self.make_rng('rnorms')
        normal_key,rng = jax.random.split(rng,2)
        x = x.transpose(0,2,1)  # [b, h, t]
        x_mask = jnp.expand_dims(commons.sequence_mask(x_lengths, x.shape[2]), 1)
        x = self.pre(x.transpose(0,2,1)).transpose(0,2,1)
        # v = v.transpose(0,2,1)  # [b, h, t]
        # v = self.hub(v.transpose(0,2,1)).transpose(0,2,1) 
        x = jnp.where(x_mask,x,0)
        #v = jnp.where(x_mask,v,0)
        x = x  + self.pit(f0).transpose(0,2,1)
        x = self.enc(jnp.where(x_mask,x,0), x_mask,train=train)
        stats = self.proj(x.transpose(0,2,1)).transpose(0,2,1)
        stats = jnp.where(x_mask,stats,0)
        m, logs = jnp.split(stats,2, axis=1)
        z = (m + jax.random.normal(normal_key,m.shape) * jnp.exp(logs)) 
        z = jnp.where(x_mask,z,0)
        return z, m, logs, x_mask, x


class ResidualCouplingBlock(nn.Module):
    channels:int
    hidden_channels:int
    kernel_size:int
    dilation_rate:int
    n_layers:int
    n_flows:int=4
    gin_channels:int=0
    def setup(
        self
    ):
        flows = []
        for i in range(self.n_flows):
            flows.append(
                modules.ResidualCouplingLayer(
                    self.channels,
                    self.hidden_channels,
                    self.kernel_size,
                    self.dilation_rate,
                    self.n_layers,
                    gin_channels=self.gin_channels,
                    mean_only=True
                )
            )
            flows.append(modules.Flip())
        self.flows=flows

    def __call__(self, x, x_mask, g=None, reverse=False,train=True):
        if not reverse:
            total_logdet = 0
            for flow in self.flows:
                x, log_det = flow(x, x_mask, g=g, reverse=reverse,train=train)
                total_logdet += log_det
            return x, total_logdet
        else:
            total_logdet = 0
            for flow in reversed(self.flows):
                x, log_det = flow(x, x_mask, g=g, reverse=reverse,train=train)
                total_logdet += log_det
            return x, total_logdet


class PosteriorEncoder(nn.Module):
    in_channels:int
    out_channels:int
    hidden_channels:int
    kernel_size:int
    dilation_rate:int
    n_layers:int
    gin_channels:int=0
    def setup(
        self
    ):
        self.pre = nn.Conv(features=self.hidden_channels, kernel_size=[1],kernel_init=nn.initializers.normal(),bias_init=nn.initializers.normal())
        self.enc = modules.WN(
            self.hidden_channels,
            self.kernel_size,
            self.dilation_rate,
            self.n_layers,
            gin_channels=self.gin_channels
        )
        self.proj = nn.Conv(features=self.out_channels * 2,kernel_size=[1],kernel_init=nn.initializers.normal(),bias_init=nn.initializers.normal())

    def __call__(self, x, x_lengths,g=None,train=True):
        rng = self.make_rng('rnorms')
        normal_key,rng = jax.random.split(rng,2)
        x_mask = jnp.expand_dims(commons.sequence_mask(x_lengths, x.shape[2]), 1)
        x = self.pre(x.transpose(0,2,1)).transpose(0,2,1)
        x = jnp.where(x_mask,x,0)
        x = self.enc(x, x_mask, g=g,train=train)
        stats = self.proj(x.transpose(0,2,1)).transpose(0,2,1)
        stats = jnp.where(x_mask,stats,0)
        m, logs = jnp.split(stats,2, axis=1)
        z = (m + jax.random.normal(normal_key,m.shape) * jnp.exp(logs))
        z = jnp.where(x_mask,z,0)
        return z, m, logs, x_mask
def l2norm(t, axis=1, eps=1e-12):
    """Performs L2 normalization of inputs over specified axis.

    Args:
      t: jnp.ndarray of any shape
      axis: the dimension to reduce, default -1
      eps: small value to avoid division by zero. Default 1e-12
    Returns:
      normalized array of same shape as t


    """
    denom = jnp.clip(jnp.linalg.norm(t, ord=2, axis=axis, keepdims=True), eps)
    out = t/denom
    return (out)
class SynthesizerTrn(nn.Module):
    spec_channels : int
    segment_size : int
    hp:tuple
    train: bool = True
    def setup(self):
        self.emb_g = nn.Dense(self.hp.vits.gin_channels,dtype=jnp.float32)
        self.enc_p = TextEncoder(
            self.hp.vits.ppg_dim,
            self.hp.vits.inter_channels,
            self.hp.vits.hidden_channels,
            self.hp.vits.filter_channels,
            2,
            6,
            3,
            0.1
        )
        self.speaker_classifier = SpeakerClassifier(
            self.hp.vits.hidden_channels,
            self.hp.vits.spk_dim,
        )
        self.enc_q = PosteriorEncoder(
            self.spec_channels,
            self.hp.vits.inter_channels,
            self.hp.vits.hidden_channels,
            5,
            1,
            16,
            gin_channels=self.hp.vits.gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            self.hp.vits.inter_channels,
            self.hp.vits.hidden_channels,
            5,
            1,
            4,
            gin_channels=self.hp.vits.spk_dim,
        )
        self.dec = Generator(hp=self.hp)

    def __call__(self, pit, vec,spec, spk, vec_l, spec_l,train=True):
        rng = self.make_rng('rnorms')
        ppg_key,vec_key,slice_key , rng = jax.random.split(rng,4)
        #ppg = ppg + jax.random.normal(ppg_key,ppg.shape) * 1  # Perturbation
        vec = vec + jax.random.normal(vec_key,vec.shape) * 2  # Perturbation
        g = jnp.expand_dims(self.emb_g(l2norm(spk,axis=1)),-1)
        
        z_p, m_p, logs_p, ppg_mask, x = self.enc_p(
            vec, vec_l, f0=f0_to_coarse(pit),train=train)
        z_q, m_q, logs_q, spec_mask = self.enc_q(spec, spec_l, g=g,train=train)
        z_slice, pit_slice, ids_slice = commons.rand_slice_segments_with_pitch(
            z_q, pit, spec_l, self.segment_size,rng=slice_key)
        audio = self.dec(spk, z_slice, pit_slice,train=train)

        # SNAC to flow
        z_f, logdet_f = self.flow(z_q, spec_mask, g=spk,reverse=False,train=train)
        z_r, logdet_r = self.flow(z_p, spec_mask, g=spk,reverse=True,train=train)

        # def gradient_reversal(x):
        #     zeros = -x + jax.lax.stop_gradient(x)
        #     return zeros + jax.lax.stop_gradient(x)
        
        #spk_preds = self.speaker_classifier(gradient_reversal(x))
        return audio, ids_slice, spec_mask, (z_f, z_r, z_p, m_p, logs_p, z_q, m_q, logs_q, logdet_f, logdet_r)#,spk_preds

    def infer(self,  pit,vec, spk, vec_l):
        rng = self.make_rng('rnorms')
        #infer_key , rng = jax.random.split(rng)
        #ppg = ppg + jax.random.normal(infer_key,ppg.shape) * 0.0001  # Perturbation
        z_p, m_p, logs_p, ppg_mask, x = self.enc_p(
             vec,vec_l, f0=f0_to_coarse(pit),train=False)
        z, _ = self.flow(z_p, ppg_mask, g=spk, reverse=True,train=False)
        o = self.dec(spk, z * ppg_mask, f0=pit,train=False)
        return o


# class SynthesizerInfer(nn.Module):
#     def __init__(
#         self,
#         spec_channels,
#         segment_size,
#         hp
#     ):
#         super().__init__()
#         self.segment_size = segment_size
#         self.enc_p = TextEncoder(
#             hp.vits.ppg_dim,
#             hp.vits.inter_channels,
#             hp.vits.hidden_channels,
#             hp.vits.filter_channels,
#             2,
#             6,
#             3,
#             0.1,
#         )
#         self.flow = ResidualCouplingBlock(
#             hp.vits.inter_channels,
#             hp.vits.hidden_channels,
#             5,
#             1,
#             4,
#             gin_channels=hp.vits.spk_dim
#         )
#         self.dec = Generator(hp=hp)

#     def remove_weight_norm(self):
#         self.flow.remove_weight_norm()
#         self.dec.remove_weight_norm()

#     def pitch2source(self, f0):
#         return self.dec.pitch2source(f0)

#     def source2wav(self, source):
#         return self.dec.source2wav(source)

#     def inference(self, ppg, pit, spk, ppg_l, source):

#         ppg = ppg + jax.random.normal(rng,ppg.shape) * 0.0001  # Perturbation
#         z_p, m_p, logs_p, ppg_mask, x = self.enc_p(
#             ppg, ppg_l, f0=f0_to_coarse(pit))
#         z_p = m_p + jax.random.normal(rng,m_p.shape) * jnp.exp(logs_p) * 0.7  
#         z, _ = self.flow(z_p, ppg_mask, g=spk, reverse=True)
#         o = self.dec.inference(spk, z * ppg_mask, source)
#         return o
