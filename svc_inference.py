import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp
import jax
import numpy as np
import optax
import argparse

from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from scipy.io.wavfile import write
from vits.models import SynthesizerTrn
from pitch import load_csv_pitch
from flax.training import orbax_utils
import orbax
from functools import partial
from diffusion.naive import Unit2MelNaive
from diffusion.gaussian import Gaussian
from diffusion.wavenet import WaveNet
import torch
jax.config.update('jax_platform_name', 'cpu')
from jax.experimental.compilation_cache import compilation_cache as cc
cc.initialize_cache("./jax_cache")
import flax
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
def main(args):

    # if (args.ppg == None):
    args.ppg = "svc_tmp.ppg.npy"
    #     print(
    #         f"Auto run : python whisper/inference.py -w {args.wave} -p {args.ppg}")
    #     os.system(f"python whisper/inference.py -w {args.wave} -p {args.ppg}")
    # if (args.vec == None):
    args.vec = "svc_tmp.vec.npy"
    #     print(
    #         f"Auto run : python hubert/inference.py -w {args.wave} -v {args.vec}")
    #     os.system(f"python hubert/inference.py -w {args.wave} -v {args.vec}")
    # if (args.pit == None):
    args.pit = "svc_tmp.pit.csv"
    #     print(
    #         f"Auto run : python pitch/inference.py -w {args.wave} -p {args.pit}")
    #     os.system(f"python pitch/inference.py -w {args.wave} -p {args.pit}")

    
    hp = OmegaConf.load(args.config)
    def create_naive_state(): 
        r"""Create the training state given a model class. """ 
        rng = jax.random.PRNGKey(1234)
        model = Unit2MelNaive(input_channel=hp.data.encoder_out_channels, 
                    n_spk=hp.model_naive.n_spk,
                    use_pitch_aug=hp.model_naive.use_pitch_aug,
                    out_dims=128,
                    n_layers=hp.model_naive.n_layers,
                    n_chans=hp.model_naive.n_chans,
                    n_hidden=hp.model_naive.n_hidden,
                    use_speaker_encoder=hp.model_naive.use_speaker_encoder,
                    speaker_encoder_out_channels=hp.data.speaker_encoder_out_channels)
        

        tx = optax.lion(learning_rate=0.01, b1=hp.train.betas[0],b2=hp.train.betas[1])
        fake_ppg = jnp.ones((1,400,1280))
        fake_vec = jnp.ones((1,400,256))
        # fake_spec = jnp.ones((1,513,400))
        # fake_ppg_l = jnp.ones((1))
        # fake_spec_l = jnp.ones((1))
        fake_pit = jnp.ones((1,400))
        #fake_spk = jnp.ones((1,256))
        params_key,r_key,dropout_key,rng = jax.random.split(rng,4)
        init_rngs = {'params': params_key, 'dropout': dropout_key,'rnorms':r_key}
        
        variables = model.init(init_rngs, fake_ppg, fake_vec,fake_pit)

        state = TrainState.create(apply_fn=model.apply, tx=tx,params=variables['params'])
        
        return state
    def create_wavenet_state(): 
        r"""Create the training state given a model class. """ 
        rng = jax.random.PRNGKey(1234)
        model = WaveNet(in_dims=128,
                            n_layers=hp.model_diff.n_layers,
                            n_chans=hp.model_diff.n_chans,
                            n_hidden=hp.model_diff.n_hidden)

        input_shape = (1, 128, 250)
        input_shapes = (input_shape, input_shape[0], input_shape)
        inputs = list(map(lambda shape: jnp.empty(shape), input_shapes))

        exponential_decay_scheduler = optax.exponential_decay(init_value=hp.train.learning_rate, transition_steps=hp.train.total_steps, decay_rate=hp.train.lr_decay)
        tx = optax.lion(learning_rate=exponential_decay_scheduler, b1=hp.train.betas[0],b2=hp.train.betas[1])

        variables = model.init(rng, *inputs)

        state = TrainState.create(apply_fn=model.apply, tx=tx,params=variables['params'])
        
        return state
    naive_state = create_naive_state()
    wavenet_state = create_wavenet_state()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        'chkpt/combine/', orbax_checkpointer, options)
    if checkpoint_manager.latest_step() is not None:
        target = {'model_naive': naive_state, 'model_wavenet': wavenet_state}
        step = checkpoint_manager.latest_step()  # step = 4
        states=checkpoint_manager.restore(step,items=target)
        naive_state=states['model_naive']
        wavenet_state=states['model_wavenet']
    del states
   
    #spk = np.load(args.spk)
    ppg = np.load(args.ppg)
    ppg = np.repeat(ppg, 2, 0)  # 320 PPG -> 160 * 2
    vec = np.load(args.vec)
    vec = np.repeat(vec, 2, 0)  # 320 PPG -> 160 * 2
    pit = load_csv_pitch(args.pit)
    pit = np.array(pit)
    print("pitch shift: ", args.shift)
    if (args.shift == 0):
        pass
    else:
        pit = np.array(pit)
        source = pit[pit > 0]
        source_ave = source.mean()
        source_min = source.min()
        source_max = source.max()
        print(f"source pitch statics: mean={source_ave:0.1f}, \
                min={source_min:0.1f}, max={source_max:0.1f}")
        shift = args.shift
        shift = 2 ** (shift / 12)
        pit = pit * shift

    len_pit = pit.shape[0]
    len_ppg = ppg.shape[0]
    len_min = min(len_pit, len_ppg)
    pit = pit[:len_min]
    vec = vec[:len_min, :]
    ppg = ppg[:len_min, :]

    @partial(jax.jit)
    def naive_infer(pit_i,ppg_i,vec_i,len_min_i):
        rng = jax.random.PRNGKey(1234)
        params_key,r_key,dropout_key,rng = jax.random.split(rng,4)
        init_rngs = {'params': params_key, 'dropout': dropout_key,'rnorms':r_key}
        out_audio = naive_state.apply_fn({'params': naive_state.params},ppg=ppg_i,f0=pit_i,vec=vec_i,infer=True,mutable=False,rngs=init_rngs)
        return out_audio
   
    len_min = jnp.asarray([len_min])
    ppg = jnp.asarray(ppg)
    pit = jnp.asarray(pit)
    #spk = jnp.asarray(spk)
    vec = jnp.asarray(vec)
    ppg = jnp.expand_dims(ppg,0)
    pit = jnp.expand_dims(pit,0)
    #spk = jnp.expand_dims(spk,0)
    vec = jnp.expand_dims(vec,0)
    print(ppg.shape)
    print(pit.shape)
    #print(spk.shape)
    print(vec.shape)
    print(len_min.shape)
    hidden = naive_infer(pit,ppg,vec,len_min)
    gaussian_config = hp['Gaussian']
    diff = Gaussian(**gaussian_config)
    predict_key = jax.random.PRNGKey(1234)
    print(hidden.shape)
    #hidden = hidden.squeeze(0)
    #wavenet_state = flax.jax_utils.replicate(wavenet_state)
    mel_diff = diff.pred_sample(predict_key, wavenet_state,hidden,(1,128,5050))
    mel_diff = mel_diff.squeeze(1)
    
    from vocoder import Vocoder
    mel_extractor = Vocoder("nsf-hifigan", "pretrain/nsf_hifigan/model", device="cpu")
    mel_diff = mel_diff.transpose(0,2,1)
    mel_diff = torch.from_numpy(np.asarray(mel_diff)).to(torch.device("cpu"))
    pit = torch.from_numpy(np.asarray(pit)).to(torch.device("cpu"))
    frags = mel_extractor.infer(mel_diff, pit)
    out_audio = frags.flatten().cpu().numpy()
    write("svc_out.wav", 44100, out_audio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help="yaml file for config.")
    # parser.add_argument('--model', type=str, required=True,
    #                     help="path of model for evaluation")
    parser.add_argument('--wave', type=str, required=True,
                        help="Path of raw audio.")
    # parser.add_argument('--spk', type=str, required=True,
    #                     help="Path of speaker.")
    parser.add_argument('--ppg', type=str,
                        help="Path of content vector.")
    parser.add_argument('--vec', type=str,
                        help="Path of hubert vector.")
    parser.add_argument('--pit', type=str,
                        help="Path of pitch csv file.")
    parser.add_argument('--shift', type=int, default=0,
                        help="Pitch shift key.")
    args = parser.parse_args()

    main(args)
