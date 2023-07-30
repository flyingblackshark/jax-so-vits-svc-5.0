
import argparse
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vits import utils
from omegaconf import OmegaConf
import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(1,2,3),backend='cpu')
def spectrogram_jax(y, n_fft:jnp.int32, hop_size:jnp.int32, win_size:jnp.int32):
    spec = jax.scipy.signal.stft(y, nfft=n_fft, noverlap=win_size-hop_size, nperseg=win_size,return_onesided=True,padded=True,boundary=None)
    spec = jnp.abs(spec[2])+(1e-9)
    return spec

def compute_spec(hps, filename, specname):
    audio, sampling_rate = utils.load_wav_to_torch(filename)
    assert sampling_rate == hps.sampling_rate, f"{sampling_rate} is not {hps.sampling_rate}"
    audio_norm = audio / hps.max_wav_value
    audio_norm = jnp.asarray(audio_norm)
    audio_norm = jnp.expand_dims(audio_norm,axis=0)
    n_fft = hps.filter_length
    sampling_rate = hps.sampling_rate
    hop_size = hps.hop_length
    win_size = hps.win_length
    spec = spectrogram_jax(
        audio_norm, n_fft, hop_size, win_size)
    spec = jnp.squeeze(spec, 0)
    jnp.save(specname,spec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = 'please enter embed parameter ...'
    parser.add_argument("-w", "--wav", help="wav", dest="wav")
    parser.add_argument("-s", "--spe", help="spe", dest="spe")
    args = parser.parse_args()
    print(args.wav)
    print(args.spe)
    os.makedirs(args.spe)
    wavPath = args.wav
    spePath = args.spe
    hps = OmegaConf.load("./configs/base.yaml")

    for spks in os.listdir(wavPath):
        if os.path.isdir(f"./{wavPath}/{spks}"):
            os.makedirs(f"./{spePath}/{spks}")
            print(f">>>>>>>>>>{spks}<<<<<<<<<<")
            for file in os.listdir(f"./{wavPath}/{spks}"):
                if file.endswith(".wav"):
                    # print(file)
                    file = file[:-4]
                    compute_spec(hps.data, f"{wavPath}/{spks}/{file}.wav", f"{spePath}/{spks}/{file}.pt")
        else:
            file = spks
            if file.endswith(".wav"):
                # print(file)
                file = file[:-4]
                compute_spec(hps.data, f"{wavPath}/{file}.wav", f"{spePath}/{file}.pt")





