import os
import time
import logging
import tqdm
import flax
import jax
import optax
import numpy as np
import orbax
from flax import linen as nn
from vits_extend.dataloader import create_dataloader_train
from vits_extend.dataloader import create_dataloader_eval
from vits_extend.writer import MyWriter
import jax.numpy as jnp
import orbax.checkpoint
from functools import partial
from flax.training.train_state import TrainState
from flax.training.common_utils import shard, shard_prng_key
from flax.training import orbax_utils
from diffusion.naive import Unit2MelNaive
#from diffusion.diffusion import Gaussian
PRNGKey = jnp.ndarray
def create_naive_state(rng, model_cls,hp,trainloader): 
    r"""Create the training state given a model class. """ 
    model = model_cls(input_channel=hp.data.encoder_out_channels, 
                    n_spk=hp.model.n_spk,
                    use_pitch_aug=hp.model.use_pitch_aug,
                    out_dims=128,
                    n_layers=hp.model.n_layers,
                    n_chans=hp.model.n_chans,
                    n_hidden=hp.model.n_hidden,
                    use_speaker_encoder=hp.model.use_speaker_encoder,
                    speaker_encoder_out_channels=hp.data.speaker_encoder_out_channels)
    
    exponential_decay_scheduler = optax.exponential_decay(init_value=hp.train.learning_rate, transition_steps=hp.train.total_steps,decay_rate=hp.train.lr_decay)
    tx = optax.lion(learning_rate=exponential_decay_scheduler, b1=hp.train.betas[0],b2=hp.train.betas[1])
        

    
    params_key,r_key,dropout_key,rng = jax.random.split(rng,4)
    init_rngs = {'params': params_key, 'dropout': dropout_key,'rnorms':r_key}
    (fake_ppg,fake_ppg_l,fake_vec,fake_pit,fake_spk,fake_spec,fake_spec_l,fake_audio,wav_l) = next(iter(trainloader))

    inputs = (fake_ppg,fake_vec,fake_pit)
    variables = model.init(init_rngs, *inputs)

    state = TrainState.create(apply_fn=model.apply, tx=tx, params=variables['params'])
    
    return state
# def create_wavenet_state(rng, model_cls,hp,trainloader): 
#     r"""Create the training state given a model class. """ 

#     model = model_cls(hp)

#     input_shape = (1, 320, 32)
#     input_shapes = (input_shape, input_shape[0], input_shape)
#     inputs = list(map(lambda shape: jnp.empty(shape), input_shapes))

#     exponential_decay_scheduler = optax.exponential_decay(init_value=hp.train.learning_rate, transition_steps=hp.train.total_steps, decay_rate=hp.train.lr_decay)
#     tx = optax.lion(learning_rate=exponential_decay_scheduler, b1=hp.train.betas[0],b2=hp.train.betas[1])

#     input_shape = (1, 320, 32)
#     input_shapes = (input_shape, input_shape[0], input_shape)
#     inputs = list(map(lambda shape: jnp.empty(shape), input_shapes))

#     variables = model.init(rng, *inputs)

#     state = TrainState.create(apply_fn=model.apply, tx=tx, 
#         params=variables['params'])
    
#     return state
def train(args,chkpt_path, hp):
    num_devices = jax.device_count()
    

    @partial(jax.pmap, axis_name='num_devices')
    def combine_step(naive_state: TrainState,ppg : jnp.ndarray  , pit : jnp.ndarray, vec:jnp.ndarray,spec : jnp.ndarray, spk : jnp.ndarray, ppg_l : jnp.ndarray ,spec_l:jnp.ndarray ,audio_e:jnp.ndarray,rng_e:PRNGKey):
        ppg = jnp.asarray(ppg)
        pit = jnp.asarray(pit)
        vec = jnp.asarray(vec)
        spec = jnp.asarray(spec)
        spk = jnp.asarray(spk)
        ppg_l = jnp.asarray(ppg_l)
        spec_l = jnp.asarray(spec_l)
        audio_e = jnp.asarray(audio_e)
        #gaussian_config = hp['Gaussian']
        #diff = Gaussian(**gaussian_config, image_size=dataloader_configs['image_size'])
        def naive_loss_fn(params):
            loss_naive= naive_state.apply_fn({'params': params},ppg=ppg,f0=pit,vec=vec, gt_spec=spec,infer=False,rngs={'rnorms':rng_e})
            return loss_naive

        grad_fn = jax.value_and_grad(naive_loss_fn, has_aux=False)
        loss_naive, grads_naive = grad_fn(naive_state.params)
        # Average across the devices.
        grads_naive = jax.lax.pmean(grads_naive, axis_name='num_devices')
        loss_naive = jax.lax.pmean(loss_naive, axis_name='num_devices')

        new_naive_state = naive_state.apply_gradients(grads=grads_naive)

        # def diff_loss_fn(params):
        #     loss_diff = diff(rng_e, wavenet_state, params, naive_mel)
        #     return loss_diff
        
        # # Generate data with the Generator, critique it with the Discriminator.
        # grad_fn = jax.value_and_grad(diff_loss_fn, has_aux=False)
        # loss_diff, grads_diff = grad_fn(wavenet_state.params)
        # # Average cross the devices.
        # grads_diff = jax.lax.pmean(grads_diff, axis_name='num_devices')
        # loss_diff = jax.lax.pmean(loss_diff, axis_name='num_devices')

        # # Update the discriminator through gradient descent.
        # new_wavenet_state = wavenet_state.apply_gradients(grads=grads_diff)
        return new_naive_state,loss_naive
    # @partial(jax.pmap, axis_name='num_devices')         
    # def do_validate(generator: TrainState,ppg_val:jnp.ndarray,pit_val:jnp.ndarray,vec_val:jnp.ndarray,spk_val:jnp.ndarray,ppg_l_val:jnp.ndarray,audio:jnp.ndarray):   
    #     stft = TacotronSTFT(filter_length=hp.data.filter_length,
    #             hop_length=hp.data.hop_length,
    #             win_length=hp.data.win_length,
    #             n_mel_channels=hp.data.mel_channels,
    #             sampling_rate=hp.data.sampling_rate,
    #             mel_fmin=hp.data.mel_fmin,
    #             mel_fmax=hp.data.mel_fmax)      
    #     model = SynthesizerTrn(spec_channels=hp.data.filter_length // 2 + 1,
    #     segment_size=hp.data.segment_size // hp.data.hop_length,
    #     hp=hp)
    #     predict_key = jax.random.PRNGKey(1234)
    #     fake_audio = model.apply({'params': generator.params}, 
    #                              ppg_val, pit_val,vec_val, spk_val, ppg_l_val,method=SynthesizerTrn.infer, mutable=False,rngs={'rnorms':predict_key})
    #     mel_fake = stft.mel_spectrogram(fake_audio.squeeze(1))
    #     mel_real = stft.mel_spectrogram(audio.squeeze(1))
    #     mel_loss_val = jnp.mean(jnp.abs(mel_fake - mel_real))

    #     #f idx == 0:
    #     spec_fake = stft.linear_spectrogram(fake_audio.squeeze(1))
    #     spec_real = stft.linear_spectrogram(audio.squeeze(1))
    #     audio = audio[0][0]
    #     fake_audio = fake_audio[0][0]
    #     spec_fake = spec_fake[0]
    #     spec_real = spec_real[0]
    #     return mel_loss_val,audio, fake_audio, spec_fake, spec_real
    # def validate(generator):
    #     loader = tqdm.tqdm(valloader, desc='Validation loop')
       
     
    #     mel_loss = 0.0
    #     for val_ppg, val_ppg_l,val_vec, val_pit, val_spk, val_spec, val_spec_l, val_audio, val_audio_l in loader:
    #         val_ppg=shard(val_ppg)
    #         val_ppg_l=shard(val_ppg_l)
    #         val_vec=shard(val_vec)
    #         val_pit=shard(val_pit)
    #         val_spk=shard(val_spk)
    #         val_audio=shard(val_audio)
    #         mel_loss_val,val_audio,val_fake_audio,spec_fake,spec_real=do_validate(generator,val_ppg,val_pit,val_vec,val_spk,val_ppg_l,val_audio)
    #         val_audio,val_fake_audio,spec_fake,spec_real = \
    #         jax.device_get([val_audio[0],val_fake_audio[0],spec_fake[0],spec_real[0]])
    #         mel_loss += mel_loss_val.mean()
    #         writer.log_fig_audio(np.asarray(val_audio), np.asarray(val_fake_audio), \
    #         np.asarray(spec_fake), np.asarray(spec_real), 0, step)

    #     mel_loss = mel_loss / len(valloader.dataset)
    #     mel_loss = np.asarray(mel_loss)
       
    #     writer.log_validation(mel_loss, step)

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

    naive_state = create_naive_state(key_discriminator,Unit2MelNaive ,hp,trainloader)
    #wavenet_state = create_wavenet_state(key_generator, WaveNet,hp,trainloader)

    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        'chkpt/sovits5.0/', orbax_checkpointer, options)
    if checkpoint_manager.latest_step() is not None:
        target = {'model_naive': naive_state, 'model_wavenet': None}
        step = checkpoint_manager.latest_step()  # step = 4
        states=checkpoint_manager.restore(step,items=target)
        naive_state=states['model_naive']
        #generator_state=states['model_g']

    naive_state = flax.jax_utils.replicate(naive_state)
    #wavenet_state = flax.jax_utils.replicate(wavenet_state)

    for epoch in range(init_epoch, hp.train.epochs):

        loader = tqdm.tqdm(trainloader, desc='Loading train data')
        for ppg, ppg_l,vec, pit, spk, spec, spec_l, audio, audio_l in loader:
            step_key,combine_step_key=jax.random.split(combine_step_key)
            step_key = shard_prng_key(step_key)
            ppg = shard(ppg)
            ppg_l = shard(ppg_l)
            vec = shard(vec)
            pit = shard(pit)
            spk_n = shard(spk)
            spec = shard(spec)
            spec_l = shard(spec_l)
            audio = shard(audio)
            audio_l = shard(audio_l)
            naive_state,loss_naive=\
            combine_step(naive_state,ppg=ppg,pit=pit,vec=vec, spk=spk_n, spec=spec,ppg_l=ppg_l,spec_l=spec_l,audio_e=audio,rng_e=step_key)

            step += 1

            loss_naive = jax.device_get([loss_naive[0]])
            if step % hp.log.info_interval == 0:
                # writer.log_training(
                #     loss_g, loss_d, loss_m, loss_s, loss_k, loss_r, score_loss,step)
                logger.info("loss %.04f  | step %d" % (loss_naive,  step))
                
        # if epoch % hp.log.eval_interval == 0:
        #     validate(generator_state)
        if epoch % hp.log.save_interval == 0:
            naive_state_s = flax.jax_utils.unreplicate(naive_state)
            #wavenet_state_s = flax.jax_utils.unreplicate(wavenet_state)
            ckpt = {'model_naive': naive_state_s, 'model_wavenet': None}
            save_args = orbax_utils.save_args_from_target(ckpt)
            checkpoint_manager.save(step, ckpt, save_kwargs={'save_args': save_args})

