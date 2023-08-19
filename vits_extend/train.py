import os
import time
import logging
import tqdm
import flax
import jax
import optax
import numpy as np
import orbax
from vits_extend.dataloader import create_dataloader_train
from vits_extend.dataloader import create_dataloader_eval
from vits_extend.writer import MyWriter
from vits_extend.stft import TacotronSTFT
from vits_extend.stft_loss import MultiResolutionSTFTLoss
from vits_decoder.discriminator import Discriminator
from vits_decoder.generator import Generator

import jax.numpy as jnp
import orbax.checkpoint
from functools import partial
from flax.training.train_state import TrainState
from flax.training.common_utils import shard, shard_prng_key
from flax.training import orbax_utils
import random
# from jax.experimental.compilation_cache import compilation_cache as cc
# cc.initialize_cache("./jax_cache")
PRNGKey = jnp.ndarray
def create_generator_state(rng, model_cls,hp,trainloader): 
    r"""Create the training state given a model class. """ 
    stft = TacotronSTFT(filter_length=hp.data.filter_length,
                    hop_length=hp.data.hop_length,
                    win_length=hp.data.win_length,
                    n_mel_channels=hp.data.mel_channels,
                    sampling_rate=hp.data.sampling_rate,
                    mel_fmin=hp.data.mel_fmin,
                    mel_fmax=hp.data.mel_fmax)
    model = model_cls(hp)
    
    #exponential_decay_scheduler = optax.exponential_decay(init_value=hp.train.learning_rate, transition_steps=hp.train.total_steps,decay_rate=hp.train.lr_decay)
    tx = optax.lion(learning_rate=hp.train.learning_rate, b1=hp.train.betas[0],b2=hp.train.betas[1])
        
    (fake_audio,fake_pit) = next(iter(trainloader))
    fake_audio = jnp.asarray(fake_audio)
    fake_pit = jnp.asarray(fake_pit)
    frame_start = random.randint(0, hp.data.segment_size*3)
    frame_end = frame_start + hp.data.segment_size
    fake_audio = fake_audio[:,:,frame_start:frame_end]
    fake_pit = fake_pit[:,frame_start:frame_end]
    params_key,r_key,dropout_key,rng = jax.random.split(rng,4)
    init_rngs = {'params': params_key, 'dropout': dropout_key,'rnorms':r_key}
    mel_real = stft.mel_spectrogram(fake_audio.squeeze(1))
    variables = model.init(init_rngs, mel_real, fake_pit,train=False)

    state = TrainState.create(apply_fn=model.apply, tx=tx, 
        params=variables['params'])
    
    return state
def create_discriminator_state(rng, model_cls,hp,trainloader): 
    r"""Create the training state given a model class. """ 
    model = model_cls(hp)
    (fake_audio,fake_pit) = next(iter(trainloader))
    fake_audio = jnp.asarray(fake_audio)
    #fake_audio = fake_audio[:,:,:hp.data.segment_size]
    #exponential_decay_scheduler = optax.exponential_decay(init_value=hp.train.learning_rate, transition_steps=hp.train.total_steps, decay_rate=hp.train.lr_decay)
    tx = optax.lion(learning_rate=hp.train.learning_rate, b1=hp.train.betas[0],b2=hp.train.betas[1])
    frame_start = random.randint(0, hp.data.segment_size*3)
    frame_end = frame_start + hp.data.segment_size
    fake_audio = fake_audio[:,:,frame_start:frame_end]
    variables = model.init(rng, fake_audio)

    state = TrainState.create(apply_fn=model.apply, tx=tx, 
        params=variables['params'])
    
    return state
def train(args,chkpt_path, hp):
    #num_devices = jax.device_count()
    

    @partial(jax.pmap, axis_name='num_devices')
    def combine_step(generator_state: TrainState,discriminator_state: TrainState,pit : jnp.ndarray,audio_e:jnp.ndarray,rng_e:PRNGKey):
        #ppg = jnp.asarray(ppg)
        pit = jnp.asarray(pit)
        # vec = jnp.asarray(vec)
        # spec = jnp.asarray(spec)
        # spk = jnp.asarray(spk)
        # ppg_l = jnp.asarray(ppg_l)
        # spec_l = jnp.asarray(spec_l)
        audio_e = jnp.asarray(audio_e)

        def loss_fn(params):
            stft = TacotronSTFT(filter_length=hp.data.filter_length,
                    hop_length=hp.data.hop_length,
                    win_length=hp.data.win_length,
                    n_mel_channels=hp.data.mel_channels,
                    sampling_rate=hp.data.sampling_rate,
                    mel_fmin=hp.data.mel_fmin,
                    mel_fmax=hp.data.mel_fmax)
            stft_criterion = MultiResolutionSTFTLoss(eval(hp.mrd.resolutions))
            seg_c = hp.data.segment_size//hp.data.hop_length
            dropout_key ,predict_key, rng = jax.random.split(rng_e, 3)



            frame_start = random.randint(0, hp.data.segment_size*3)
            frame_end = frame_start + hp.data.segment_size
            audio = audio_e[:,:,frame_start:frame_end]
            print(audio.shape)
            mel_real = stft.mel_spectrogram(audio.squeeze(1))[:,:,:seg_c]
            #print(mel_real.shape)
            fake_audio = generator_state.apply_fn({'params': params},  mel_real, pit,train=True, rngs={'dropout': dropout_key,'rnorms':predict_key})
            #print(fake_audio.shape)
            #spk_loss = (1-optax.cosine_similarity(spk,spk_preds)).mean()
            #audio = commons.slice_segments(audio_e, ids_slice * hp.data.hop_length, hp.data.segment_size)  # slice
            mel_fake = stft.mel_spectrogram(fake_audio.squeeze(1))[:,:,:seg_c]
            
            
            mel_loss = jnp.mean(jnp.abs(mel_fake - mel_real)) * hp.train.c_mel
            #Multi-Resolution STFT Loss
            
            sc_loss, mag_loss = stft_criterion(fake_audio.squeeze(1), audio.squeeze(1))
            stft_loss = (sc_loss + mag_loss) * hp.train.c_stft

            # Generator Loss 
            disc_fake = discriminator_state.apply_fn(
            {'params': discriminator_state.params}, fake_audio)
            score_loss = 0.0
            for (_, score_fake) in disc_fake:
                score_loss += jnp.mean(jnp.square(score_fake - 1.0))
            score_loss = score_loss / len(disc_fake)

            # Feature Loss
            disc_real= discriminator_state.apply_fn({'params': discriminator_state.params}, audio)

            feat_loss = 0.0
            for (feat_fake, _), (feat_real, _) in zip(disc_fake, disc_real):
                for fake, real in zip(feat_fake, feat_real):
                    feat_loss += jnp.mean(jnp.abs(fake - real))
            feat_loss = feat_loss / len(disc_fake)
            feat_loss = feat_loss * 2

            # Kl Loss
            #loss_kl_f = kl_loss(z_f, logs_q, m_p, logs_p, logdet_f, z_mask) * hp.train.c_kl
            #loss_kl_r = kl_loss(z_r, logs_p, m_q, logs_q, logdet_r, z_mask) * hp.train.c_kl
            # Loss
            loss_g = mel_loss + score_loss +  feat_loss + stft_loss#+ loss_kl_f + loss_kl_r * 0.5  #+ spk_loss * 2

            return loss_g, (fake_audio,mel_loss,stft_loss,score_loss)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss_g,(fake_audio_g,mel_loss,stft_loss,score_loss)), grads_g = grad_fn(generator_state.params)

        # Average across the devices.
        grads_g = jax.lax.pmean(grads_g, axis_name='num_devices')
        loss_g = jax.lax.pmean(loss_g, axis_name='num_devices')
        loss_m = jax.lax.pmean(mel_loss, axis_name='num_devices')
        loss_s = jax.lax.pmean(stft_loss, axis_name='num_devices')
        #loss_k = jax.lax.pmean(loss_kl_f, axis_name='num_devices')
        #loss_r = jax.lax.pmean(loss_kl_r, axis_name='num_devices')
        #loss_i = jax.lax.pmean(spk_loss, axis_name='num_devices')

        new_generator_state = generator_state.apply_gradients(
            grads=grads_g)
        
        def loss_fn(params):
            disc_fake  = discriminator_state.apply_fn(
                {'params': params},fake_audio_g)
            disc_real  = discriminator_state.apply_fn(
                {'params': params},audio_e)
            loss_d = 0.0
            for (_, score_fake), (_, score_real) in zip(disc_fake, disc_real):
                loss_d += jnp.mean(jnp.square(score_real - 1.0))
                loss_d += jnp.mean(jnp.square(score_fake))
            loss_d = loss_d / len(disc_fake)
          
            return loss_d
        
        # Generate data with the Generator, critique it with the Discriminator.
        grad_fn = jax.value_and_grad(loss_fn, has_aux=False)

        loss_d, grads_d = grad_fn(discriminator_state.params)

        # Average cross the devices.
        grads_d = jax.lax.pmean(grads_d, axis_name='num_devices')
        loss_d = jax.lax.pmean(loss_d, axis_name='num_devices')

        # Update the discriminator through gradient descent.
        new_discriminator_state = discriminator_state.apply_gradients(grads=grads_d)
        return new_generator_state,new_discriminator_state,loss_g,loss_d,loss_m,loss_s,score_loss
    @partial(jax.pmap, axis_name='num_devices')         
    def do_validate(generator: TrainState,pit_val:jnp.ndarray,audio:jnp.ndarray):   
        stft = TacotronSTFT(filter_length=hp.data.filter_length,
                hop_length=hp.data.hop_length,
                win_length=hp.data.win_length,
                n_mel_channels=hp.data.mel_channels,
                sampling_rate=hp.data.sampling_rate,
                mel_fmin=hp.data.mel_fmin,
                mel_fmax=hp.data.mel_fmax)      
        # model = SynthesizerTrn(spec_channels=hp.data.filter_length // 2 + 1,
        # segment_size=hp.data.segment_size // hp.data.hop_length,
        # hp=hp)
        predict_key = jax.random.PRNGKey(1234)
        seg_c = hp.data.segment_size//hp.data.hop_length
        mel_real = stft.mel_spectrogram(audio.squeeze(1))
        fake_audio = generator.apply_fn({'params': generator.params},mel_real, pit_val,mutable=False,rngs={'rnorms':predict_key})
        mel_fake = stft.mel_spectrogram(fake_audio.squeeze(1))[:,:,:-1]
        
        mel_loss_val = jnp.mean(jnp.abs(mel_fake - mel_real))

        #f idx == 0:
        spec_fake = stft.linear_spectrogram(fake_audio.squeeze(1))
        spec_real = stft.linear_spectrogram(audio.squeeze(1))
        audio = audio[0][0]
        fake_audio = fake_audio[0][0]
        spec_fake = spec_fake[0]
        spec_real = spec_real[0]
        return mel_loss_val,audio, fake_audio, spec_fake, spec_real
    def validate(generator):
        loader = tqdm.tqdm(valloader, desc='Validation loop')
       
     
        mel_loss = 0.0
        for val_audio,val_pit in loader:
            val_audio = jnp.asarray(val_audio)
            val_pit = jnp.asarray(val_pit)
            val_pit=shard(val_pit)
            val_audio=shard(val_audio)
            mel_loss_val,val_audio,val_fake_audio,spec_fake,spec_real=do_validate(generator,val_pit,val_audio)
            val_audio,val_fake_audio,spec_fake,spec_real = \
            jax.device_get([val_audio[0],val_fake_audio[0],spec_fake[0],spec_real[0]])
            mel_loss += mel_loss_val.mean()
            writer.log_fig_audio(np.asarray(val_audio), np.asarray(val_fake_audio), \
            np.asarray(spec_fake), np.asarray(spec_real), 0, step)

        mel_loss = mel_loss / len(valloader.dataset)
        mel_loss = np.asarray(mel_loss)
       
        writer.log_validation(mel_loss, step)

    key = jax.random.PRNGKey(seed=hp.train.seed)
    combine_step_key,key_generator, key_discriminator, key = jax.random.split(key, 4)
    
    init_epoch = 1
    step = 0
    #if rank == 0:
    pth_dir = os.path.join(hp.log.pth_dir, args.name)
    log_dir = os.path.join(hp.log.log_dir, args.name)
    os.makedirs(pth_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, '%s-%d.log' % (args.name, time.time()))),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    writer = MyWriter(hp, log_dir)
    valloader = create_dataloader_eval(hp)
    trainloader = create_dataloader_train(hp)

    discriminator_state = create_discriminator_state(key_discriminator, Discriminator,hp,trainloader)
    generator_state = create_generator_state(key_generator, Generator,hp,trainloader)

    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=hp.train.max_to_keep, create=True)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        'chkpt/nsf_hifigan/', orbax_checkpointer, options)
    if checkpoint_manager.latest_step() is not None:
        target = {'model_g': generator_state, 'model_d': discriminator_state}
        step = checkpoint_manager.latest_step()  # step = 4
        states=checkpoint_manager.restore(step,items=target)
        discriminator_state=states['model_d']
        generator_state=states['model_g']

    discriminator_state = flax.jax_utils.replicate(discriminator_state)
    generator_state = flax.jax_utils.replicate(generator_state)

    for epoch in range(init_epoch, hp.train.epochs):

        loader = tqdm.tqdm(trainloader, desc='Loading train data')
        for audio,pit in loader:
            audio = jnp.asarray(audio)
            pit = jnp.asarray(pit)
            step_key,combine_step_key=jax.random.split(combine_step_key)
            step_key = shard_prng_key(step_key)
            pit = shard(pit)
            audio = shard(audio)
            generator_state,discriminator_state,loss_g,loss_d,loss_m,loss_s,score_loss= combine_step(generator_state, discriminator_state,pit=pit,audio_e=audio,rng_e=step_key)

            step += 1

            loss_g,loss_d,loss_s,loss_m,score_loss = jax.device_get([loss_g[0], loss_d[0],loss_s[0],loss_m[0],score_loss[0]])
            if step % hp.log.info_interval == 0:
                writer.log_training(loss_g, loss_d, loss_m, loss_s, score_loss,step)
                logger.info("g %.04f m %.04f s %.04f d %.04f  | step %d" % (loss_g, loss_m, loss_s, loss_d, step))
                
        if epoch % hp.log.eval_interval == 0:
            validate(generator_state)
        if epoch % hp.log.save_interval == 0:
            generator_state_s = flax.jax_utils.unreplicate(generator_state)
            discriminator_state_s = flax.jax_utils.unreplicate(discriminator_state)
            ckpt = {'model_g': generator_state_s, 'model_d': discriminator_state_s}
            save_args = orbax_utils.save_args_from_target(ckpt)
            checkpoint_manager.save(step, ckpt, save_kwargs={'save_args': save_args})

