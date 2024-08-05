import functools
import os
import time
import logging
import jax
import optax
import numpy as np
import orbax
from flax import linen as nn
from vits_extend.writer import MyWriter
from vits_extend.stft import TacotronSTFT
from vits_extend.stft_loss import MultiResolutionSTFTLoss
from vits_decoder.discriminator import Discriminator
from vits.models import SynthesizerTrn
from vits import commons
from vits.losses import kl_loss
import jax.numpy as jnp
import orbax.checkpoint
from functools import partial
from flax.training.train_state import TrainState
from flax.training import orbax_utils
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from input_pipeline.dataset import get_dataset
from jax.experimental.compilation_cache import compilation_cache as cc
cc.set_cache_dir("./jax_cache")
PRNGKey = jnp.ndarray




def create_generator_state(rng,hp,mesh): 
    r"""Create the training state given a model class. """ 
    model = SynthesizerTrn(spec_channels=hp.data.filter_length // 2 + 1,
    segment_size=hp.data.segment_size // hp.data.hop_length,
    hp=hp)
    
    #exponential_decay_scheduler = optax.exponential_decay(init_value=hp.train.learning_rate, transition_steps=hp.train.total_steps,decay_rate=hp.train.lr_decay)
    optimizer = optax.adamw(learning_rate=hp.train.learning_rate, b1=hp.train.betas[0],b2=hp.train.betas[1])
    params_key,r_key,dropout_key,rng = jax.random.split(rng,4)
    init_rngs = {'params': params_key, 'dropout': dropout_key,'rnorms':r_key}
    example_inputs = {
        "ppg":jnp.zeros((1,400,1024)),
        "pit":jnp.zeros((1,400)),
        "spec":jnp.zeros((1,513,400)),
        "ppg_l":jnp.zeros((1),dtype=jnp.int32),
        "spec_l":jnp.zeros((1),dtype=jnp.int32),
        "spk":jnp.ones((1),dtype=jnp.int32),
    }
    

    def init_fn(init_rngs,example_inputs,model, optimizer):
        variables = model.init(init_rngs, **example_inputs,train=False)
        state = TrainState.create( # Create a `TrainState`.
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer)
        return state
    abstract_variables = jax.eval_shape(functools.partial(init_fn, model=model, optimizer=optimizer), init_rngs=init_rngs, example_inputs=example_inputs)
    state_sharding = nn.get_sharding(abstract_variables, mesh)
    jit_init_fn = jax.jit(init_fn, static_argnums=(2, 3),
                      in_shardings=(NamedSharding(mesh,()), NamedSharding(mesh,())),  # PRNG key and x
                      out_shardings=state_sharding)
    return jit_init_fn(init_rngs, example_inputs,model,optimizer),state_sharding
def create_discriminator_state(rng,hp,mesh): 
    r"""Create the training state given a model class. """ 
    model = Discriminator(hp)
    params_key,r_key,dropout_key,rng = jax.random.split(rng,4)
    init_rngs = {'params': params_key, 'dropout': dropout_key,'rnorms':r_key}
    fake_audio = jnp.zeros((1,1,hp.data.segment_size))
    exponential_decay_scheduler = optax.exponential_decay(init_value=hp.train.learning_rate, transition_steps=hp.train.total_steps, decay_rate=hp.train.lr_decay)
    optimizer = optax.adamw(learning_rate=hp.train.learning_rate, b1=hp.train.betas[0],b2=hp.train.betas[1])
    
    def init_fn(init_rngs,fake_audio,model, optimizer):
        variables = model.init(init_rngs, fake_audio,train=False)
        state = TrainState.create( # Create a `TrainState`.
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer)
        return state
    abstract_variables = jax.eval_shape(functools.partial(init_fn, model=model, optimizer=optimizer), init_rngs=init_rngs, fake_audio=fake_audio)
    state_sharding = nn.get_sharding(abstract_variables, mesh)
    jit_init_fn = jax.jit(init_fn, static_argnums=(2, 3),
                      in_shardings=(NamedSharding(mesh,()), NamedSharding(mesh,())),  # PRNG key and x
                      out_shardings=state_sharding)
    
    return jit_init_fn(init_rngs, fake_audio,model,optimizer),state_sharding
def train(args,hp,mesh):
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
    #logger = logging.getLogger()
    #writer = MyWriter(hp, log_dir)
    #valloader = create_dataloader_eval(hp)
    #trainloader = create_dataloader_train(hp)
    #trainloader = get_dataset()
    discriminator_state,d_state_sharding = create_discriminator_state(key_discriminator,hp,mesh)
    generator_state,g_state_sharding = create_generator_state(key_generator,hp,mesh)

    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=hp.train.max_to_keep, create=True)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        'chkpt/sovits/', orbax_checkpointer, options)
    if checkpoint_manager.latest_step() is not None:
        target = {'model_g': generator_state, 'model_d': discriminator_state}
        step = checkpoint_manager.latest_step()  # step = 4
        states=checkpoint_manager.restore(step,items=target)
        discriminator_state=states['model_d']
        generator_state=states['model_g']
    
    # discriminator_state = flax.jax_utils.replicate(discriminator_state)
    # generator_state = flax.jax_utils.replicate(generator_state)
    x_sharding = NamedSharding(mesh,PartitionSpec('data'))

    @functools.partial(jax.jit, in_shardings=(g_state_sharding,d_state_sharding, x_sharding,None),
                   out_shardings=(g_state_sharding,d_state_sharding))
    def combine_step(generator_state: TrainState,
                       discriminator_state: TrainState,
                        example_batch,
                       rng_e:PRNGKey):
        #audio_e,audio_l,pit,pit_l,ppg,ppg_l,spec,spec_l,spk = example_batch
        def loss_fn(params):
            stft = TacotronSTFT(filter_length=hp.data.filter_length,
                    hop_length=hp.data.hop_length,
                    win_length=hp.data.win_length,
                    n_mel_channels=hp.data.mel_channels,
                    sampling_rate=hp.data.sampling_rate,
                    mel_fmin=hp.data.mel_fmin,
                    mel_fmax=hp.data.mel_fmax)
            stft_criterion = MultiResolutionSTFTLoss(eval(hp.mrd.resolutions))
            
            dropout_key ,predict_key, rng = jax.random.split(rng_e, 3)
            fake_audio, ids_slice, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = generator_state.apply_fn(
                {'params': params},  example_batch["hubert_feature"],
                  example_batch["f0_feature"], 
                  example_batch["spec_feature"],
                    example_batch["speaker_id"], 
                    example_batch["hubert_length"], 
                    example_batch["spec_length"],train=True, rngs={'dropout': dropout_key,'rnorms':predict_key})
            
            #spk_loss = (1-optax.cosine_similarity(spk,spk_preds)).mean()
            
            audio = commons.slice_segments(jnp.expand_dims(example_batch["audio"],1), ids_slice * hp.data.hop_length, hp.data.segment_size)  # slice
            mel_fake = stft.mel_spectrogram(fake_audio.squeeze(1))
            mel_real = stft.mel_spectrogram(audio.squeeze(1))
            
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
            disc_real= discriminator_state.apply_fn(
            {'params': discriminator_state.params}, audio)

            feat_loss = 0.0
            for (feat_fake, _), (feat_real, _) in zip(disc_fake, disc_real):
                for fake, real in zip(feat_fake, feat_real):
                    feat_loss += jnp.mean(jnp.abs(fake - real))
            feat_loss = feat_loss / len(disc_fake)
            feat_loss = feat_loss * 2

            # Kl Loss
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hp.train.c_kl
            #loss_kl_r = kl_loss(z_r, logs_p, m_q, logs_q, logdet_r, z_mask) * hp.train.c_kl
            # Loss
            loss_g = mel_loss + score_loss +  feat_loss + stft_loss+ loss_kl  #+ spk_loss * 2

            return loss_g, (fake_audio,audio)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss_g,(fake_audio_g,audio_g)), grads_g = grad_fn(generator_state.params)

        new_generator_state = generator_state.apply_gradients(
            grads=grads_g)
        
        def loss_fn(params):
            disc_fake  = discriminator_state.apply_fn(
                {'params': params},fake_audio_g)
            disc_real  = discriminator_state.apply_fn(
                {'params': params},audio_g)
            loss_d = 0.0
            for (_, score_fake), (_, score_real) in zip(disc_fake, disc_real):
                loss_d += jnp.mean(jnp.square(score_real - 1.0))
                loss_d += jnp.mean(jnp.square(score_fake))
            loss_d = loss_d / len(disc_fake)
          
            return loss_d
        
        # Generate data with the Generator, critique it with the Discriminator.
        grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
        loss_d, grads_d = grad_fn(discriminator_state.params)

        # Update the discriminator through gradient descent.
        #loss_g,loss_d,loss_m,loss_s,loss_k,loss_r,score_loss
        new_discriminator_state = discriminator_state.apply_gradients(grads=grads_d)
        return new_generator_state,new_discriminator_state
    data_iterator = get_dataset(hp,mesh)
    example_batch = None
    for step in range(init_epoch, hp.train.steps):
        #loader = tqdm.tqdm(trainloader, desc='Loading train data')
        #for input in loader:
        step_key,combine_step_key=jax.random.split(combine_step_key)
        example_batch = next(data_iterator)
        #print("ready a step")
        generator_state,discriminator_state=combine_step(generator_state, discriminator_state,example_batch,step_key)
        #print("finish a step")

        # loss_g,loss_d,loss_s,loss_m,loss_k,loss_r,score_loss = jax.device_get([loss_g[0], loss_d[0],loss_s[0],loss_m[0],loss_k[0],loss_r[0],score_loss[0]])
        if step % hp.log.info_interval == 0:
            print(f"current step:{step}")
        #     writer.log_training(
        #         loss_g, loss_d, loss_m, loss_s, loss_k, loss_r, score_loss,step)
        #     logger.info("g %.04f m %.04f s %.04f d %.04f k %.04f r %.04f i %.04f | step %d" % (
        #         loss_g, loss_m, loss_s, loss_d, loss_k, loss_r,0., step))
            
    # if epoch % hp.log.eval_interval == 0:
    #     validate(generator_state)
    # if step % hp.log.save_interval == 0:
    #     generator_state_s = flax.jax_utils.unreplicate(generator_state)
    #     discriminator_state_s = flax.jax_utils.unreplicate(discriminator_state)
    #     ckpt = {'model_g': generator_state_s, 'model_d': discriminator_state_s}
    #     save_args = orbax_utils.save_args_from_target(ckpt)
    #     checkpoint_manager.save(step, ckpt, save_kwargs={'save_args': save_args})

