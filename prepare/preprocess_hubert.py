import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import argparse
import torch
from tqdm import tqdm
#from multiprocessing import Pool
#from whisper.audio import load_audio
#from hubert import hubert_model
from concurrent.futures import ThreadPoolExecutor, as_completed
from fairseq import checkpoint_utils
import soundfile as sf
import torch.nn.functional as F
# def load_model(path, device):
#     model = hubert_model.hubert_soft(path)
#     model.eval()
#     #model.half()
#     model.to(device)
#     return model

def readwave(wav_path, normalize=False):
    wav, sr = sf.read(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats
def pred_vec(model, wavPath, vecPath, device):
    #feats = load_audio(wavPath)
    feats = readwave(wavPath, normalize=saved_cfg.task.normalize)
    #feats = torch.from_numpy(feats).to(device)
    padding_mask = torch.BoolTensor(feats.shape).fill_(False)
    inputs = {
        "source": feats.to(device),
        "padding_mask": padding_mask.to(device),
        "output_layer": 9 ,  # layer 9
    }
    #feats = feats[None, None, :].half()
    with torch.no_grad():
        logits = model.extract_features(**inputs)
        feats = (
            model.final_proj(logits[0])
        )
        #vec = model.units(feats).squeeze().data.cpu().float().numpy()
        # print(vec.shape)   # [length, dim=256] hop=320
        vec = feats.squeeze(0).float().cpu().numpy()
        np.save(vecPath, vec, allow_pickle=False)


def process_file(file):
    if file.endswith(".wav"):
        file = file[:-4]
        pred_vec(hubert, f"{wavPath}/{spks}/{file}.wav", f"{vecPath}/{spks}/{file}.vec", device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = 'please enter embed parameter ...'
    parser.add_argument("-w", "--wav", help="wav", dest="wav")
    parser.add_argument("-v", "--vec", help="vec", dest="vec")
    parser.add_argument("-t", "--thread_count", help="thread count to process, set 0 to use all cpu cores", dest="thread_count", type=int, default=1)
    
    args = parser.parse_args()
    print(args.wav)
    print(args.vec)
    os.makedirs(args.vec, exist_ok=True)
    
    wavPath = args.wav
    vecPath = args.vec
    model_path = "hubert_base.pt"
#    assert torch.cuda.is_available()
    device = "cpu"
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    [model_path],
    suffix="",
    )
    hubert = models[0]
    # hubert = load_model(os.path.join(
    #     "hubert_pretrain", "hubert-soft-0d54a1f4.pt"), device)

    for spks in os.listdir(wavPath):
        if os.path.isdir(f"./{wavPath}/{spks}"):
            os.makedirs(f"./{vecPath}/{spks}", exist_ok=True)
            print(f">>>>>>>>>>{spks}<<<<<<<<<<")
            if args.thread_count == 1:
                for file in os.listdir(f"./{wavPath}/{spks}"):
                    if file.endswith(".wav"):
                        print(file)
                        file = file[:-4]
                        pred_vec(hubert, f"{wavPath}/{spks}/{file}.wav", f"{vecPath}/{spks}/{file}.vec", device)
            else:
                if args.thread_count == 0:
                    process_num = os.cpu_count()
                else:
                    process_num = args.thread_count
                with ThreadPoolExecutor(max_workers=process_num) as executor:
                    futures = [executor.submit(process_file, file) for file in os.listdir(f"./{wavPath}/{spks}")]
                    for future in tqdm(as_completed(futures), total=len(futures)):
                        pass
                # with Pool(processes=process_num) as pool:
                #     results = [pool.apply_async(process_file, (file,)) for file in os.listdir(f"./{wavPath}/{spks}")]
                #     for result in tqdm(results, total=len(results)):
                #         result.wait()