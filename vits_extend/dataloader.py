from torch.utils.data import DataLoader
#import jax_dataloader as jdl
import jax.numpy as jnp
import numpy as np
from vits.data_utils import DistributedBucketSampler
from vits.data_utils import TextAudioSpeakerCollate
from vits.data_utils import TextAudioSpeakerSet

def create_dataloader_train(hps):
    collate_fn = TextAudioSpeakerCollate()
    train_dataset = TextAudioSpeakerSet(hps.data.training_files, hps.data , hps.train.mode)
    train_loader = DataLoader(
        train_dataset,
        batch_size=hps.train.batch_size,
        num_workers=0,
        shuffle=True,
        pin_memory=False,
        collate_fn=collate_fn,
        persistent_workers=False,
        drop_last=True)
    return train_loader


def create_dataloader_eval(hps):
    collate_fn = TextAudioSpeakerCollate()
    eval_dataset = TextAudioSpeakerSet(hps.data.validation_files, hps.data , hps.train.mode)
    eval_loader = DataLoader(
        eval_dataset,
        num_workers=0,
        shuffle=False,
        batch_size=8,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn)
    return eval_loader
