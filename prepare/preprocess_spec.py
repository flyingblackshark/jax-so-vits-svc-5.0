import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import argparse
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from vits import utils
from omegaconf import OmegaConf
from librosa.filters import mel as librosa_mel_fn
def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
   
    hann_window = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    mel_basis = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=128, fmin=50, fmax=16000)
    mel_basis = torch.from_numpy(mel_basis).float().to(y.device)
    spec = torch.matmul(mel_basis, spec)
    spec = dynamic_range_compression_torch(spec, clip_val=1e-5)
    return spec

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C
from vocoder import Vocoder
mel_extractor = Vocoder("nsf-hifigan", "pretrain/nsf_hifigan/model", device="cpu")
def compute_spec(hps, filename, specname):
    audio, sampling_rate = utils.load_wav_to_torch(filename)
    # assert sampling_rate == hps.sampling_rate, f"{sampling_rate} is not {hps.sampling_rate}"
    # audio_norm = audio / hps.max_wav_value
    audio = audio.unsqueeze(0)
    # n_fft = hps.filter_length
    # sampling_rate = hps.sampling_rate
    # hop_size = hps.hop_length
    # win_size = hps.win_length
    # spec = spectrogram_torch(
    #     audio_norm, n_fft, sampling_rate, hop_size, win_size, center=False)
    # spec = torch.squeeze(spec, 0)
    mel_t = mel_extractor.extract(audio, 44100)
    mel = mel_t.squeeze().transpose(1,0)
    torch.save(mel, specname)


def process_file(file):
    if file.endswith(".wav"):
        file = file[:-4]
        compute_spec(hps.data, f"{wavPath}/{spks}/{file}.wav", f"{spePath}/{spks}/{file}.pt")

def process_files_with_thread_pool(wavPath, spks, max_workers):
    files = os.listdir(f"./{wavPath}/{spks}")
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        list(tqdm(executor.map(process_file, files), total=len(files)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = 'please enter embed parameter ...'
    parser.add_argument("-w", "--wav", help="wav", dest="wav")
    parser.add_argument("-s", "--spe", help="spe", dest="spe")
    parser.add_argument("-t", "--thread_count", help="thread count to process, set 0 to use all cpu cores", dest="thread_count", type=int, default=1)
    args = parser.parse_args()
    print(args.wav)
    print(args.spe)
    if not os.path.exists(args.spe):
        os.makedirs(args.spe)
    wavPath = args.wav
    spePath = args.spe
    hps = OmegaConf.load("./configs/base.yaml")

    for spks in os.listdir(wavPath):
        if os.path.isdir(f"./{wavPath}/{spks}"):
            if not os.path.exists(f"./{spePath}/{spks}"):
                os.makedirs(f"./{spePath}/{spks}")
            if args.thread_count == 0:
                process_num = os.cpu_count()
            else:
                process_num = args.thread_count
            process_files_with_thread_pool(wavPath, spks, process_num)