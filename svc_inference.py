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
jax.config.update('jax_platform_name', 'cpu')

def main(args):

    if (args.ppg == None):
        args.ppg = "svc_tmp.ppg.npy"
        print(
            f"Auto run : python whisper/inference.py -w {args.wave} -p {args.ppg}")
        os.system(f"python whisper/inference.py -w {args.wave} -p {args.ppg}")
    if (args.vec == None):
        args.vec = "svc_tmp.vec.npy"
        print(
            f"Auto run : python hubert/inference.py -w {args.wave} -v {args.vec}")
        os.system(f"python hubert/inference.py -w {args.wave} -v {args.vec}")
    if (args.pit == None):
        args.pit = "svc_tmp.pit.csv"
        print(
            f"Auto run : python pitch/inference.py -w {args.wave} -p {args.pit}")
        os.system(f"python pitch/inference.py -w {args.wave} -p {args.pit}")

    
    hp = OmegaConf.load(args.config)
    def create_generator_state(): 
        r"""Create the training state given a model class. """ 
        rng = jax.random.PRNGKey(1234)
        model = SynthesizerTrn(spec_channels=hp.data.filter_length // 2 + 1,
        segment_size=hp.data.segment_size // hp.data.hop_length,
        hp=hp)
        

        tx = optax.lion(learning_rate=0.01, b1=hp.train.betas[0],b2=hp.train.betas[1])
        fake_ppg = jnp.ones((1,400,1280))
        fake_vec = jnp.ones((1,400,256))
        fake_spec = jnp.ones((1,513,400))
        fake_ppg_l = jnp.ones((1))
        fake_spec_l = jnp.ones((1))
        fake_pit = jnp.ones((1,400))
        fake_spk = jnp.ones((1,256))
        params_key,r_key,dropout_key,rng = jax.random.split(rng,4)
        init_rngs = {'params': params_key, 'dropout': dropout_key,'rnorms':r_key}
        
        variables = model.init(init_rngs, ppg=fake_ppg, pit=fake_pit,vec=fake_vec, spec=fake_spec, spk=fake_spk, ppg_l=fake_ppg_l, spec_l=fake_spec_l,train=False)

        state = TrainState.create(apply_fn=SynthesizerTrn.apply, tx=tx,params=variables['params'])
        
        return state
    generator_state = create_generator_state()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        'chkpt/sovits5.0/', orbax_checkpointer, options)
    if checkpoint_manager.latest_step() is not None:
        target = {'model_g': generator_state, 'model_d': None}
        step = checkpoint_manager.latest_step()  # step = 4
        states=checkpoint_manager.restore(step,items=target)
        generator_state=states['model_g']
    del states
    spk = np.load(args.spk)
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
    
    @partial(jax.jit,backend='cpu')
    def parallel_infer(pit_i,ppg_i,spk_i,vec_i,len_min_i):
        model = SynthesizerTrn(spec_channels=hp.data.filter_length // 2 + 1,
            segment_size=hp.data.segment_size // hp.data.hop_length,
            hp=hp,train=False)
        rng = jax.random.PRNGKey(1234)
        params_key,r_key,dropout_key,rng = jax.random.split(rng,4)
        init_rngs = {'params': params_key, 'dropout': dropout_key,'rnorms':r_key}
        out_audio = model.apply( {'params': generator_state.params},ppg_i,pit_i,vec_i,spk_i,len_min_i,method=SynthesizerTrn.infer,rngs=init_rngs)
        return out_audio
   
    len_min = jnp.asarray([len_min])
    ppg = jnp.asarray(ppg)
    pit = jnp.asarray(pit)
    spk = jnp.asarray(spk)
    vec = jnp.asarray(vec)
    ppg = jnp.expand_dims(ppg,0)
    pit = jnp.expand_dims(pit,0)
    spk = jnp.expand_dims(spk,0)
    vec = jnp.expand_dims(vec,0)
    print(ppg.shape)
    print(pit.shape)
    print(spk.shape)
    print(vec.shape)
    print(len_min.shape)
    frags = parallel_infer(pit,ppg,spk,vec,len_min)
    out_audio = jnp.reshape(frags,[frags.shape[0]*frags.shape[1]*frags.shape[2]])
    out_audio = np.asarray(out_audio)
    write("svc_out.wav", 32000, out_audio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help="yaml file for config.")
    # parser.add_argument('--model', type=str, required=True,
    #                     help="path of model for evaluation")
    parser.add_argument('--wave', type=str, required=True,
                        help="Path of raw audio.")
    parser.add_argument('--spk', type=str, required=True,
                        help="Path of speaker.")
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
