from collections import namedtuple
from functools import partial
import numpy as np
from einops import einops
from flax.training.common_utils import shard, shard_prng_key
from tqdm import tqdm
from .noise import normal_noise, truncate_noise, pyramid_nosie, resize_noise, offset_noise
from .schedules import linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule
from .loss import l1_loss, l2_loss, charbonnier_loss
import jax
import jax.numpy as jnp

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def extract(a, t, x_shape):
    b = t.shape[0]
    # b, *_ = t.shape
    out = a[t]
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def model_predict_ema(model, x, time, x_self_cond=None):
    return model.apply_fn({"params": model.ema_params}, x, time, x_self_cond)


def model_predict(model, x, time, x_self_cond=None):
    return model.apply_fn({"params": model.params}, x, time, x_self_cond)


def extract(a, t, x_shape):
    b = t.shape[0]
    # b, *_ = t.shape
    out = a[t]
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def identity(x):
    return x


class Gaussian:
    def __init__(
            self,
            sample_shape,
            loss='l2',
            timesteps=1000,
            sampling_timesteps=1000,
            objective='predict_noise',
            beta_schedule='linear',
            beta_schedule_configs=None,
            ddim_sampling_eta=0.,
            min_snr_loss_weight=False,
            scale_shift=False,
            self_condition=False,
            noise_type='normal',
            scale_factor=1,
            p_loss=False,
            mean=0,
            std=1,
            clip_x_start=True

    ):

        self.sample_shape = sample_shape
        if beta_schedule_configs is None:
            beta_schedule_configs = {}
        self.clip_x_start = clip_x_start
        #self.train_state = True
        self.noise_type = noise_type
        self.scale_factor = scale_factor
        self.model = None

        self.self_condition = self_condition
        self.mean = mean
        self.std = std

        if mean != 0 and std != 1:
            print(f'mean:{mean} std:{std}')

        assert objective in {'predict_noise', 'predict_x0', 'predict_v', 'predict_mx'}
        self.objective = objective

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule

        betas = beta_schedule_fn(timesteps, **beta_schedule_configs)

        alphas = 1 - betas
        if scale_shift:
            image_size = self.sample_shape[0]
            scale = 64 / image_size
            snr = alphas / (1 - alphas)
            alphas = 1 - 1 / (1 + (scale) ** 1 * snr)
            betas = 1 - alphas

        alphas_cumprod = jnp.cumprod(alphas)
        alphas_cumprod_prev = jnp.pad(alphas_cumprod[:-1], (1, 0), constant_values=1)

        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = sampling_timesteps

        assert self.sampling_timesteps <= timesteps

        self.is_ddim_sample = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        self.sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1 - alphas_cumprod)
        self.log_one_minus_alphas_cumprod = jnp.log(1 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = jnp.sqrt(1 / alphas_cumprod)
        # self.sqrt_recip_one_minus_alphas_cumprod = jnp.sqrt(1 / (1 - alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = jnp.sqrt(1 / alphas_cumprod - 1)

        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

        self.posterior_variance = posterior_variance

        self.posterior_log_variance_clipped = jnp.log(posterior_variance.clip(min=1e-20))
        self.posterior_mean_coef1 = betas * jnp.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.posterior_mean_coef2 = (1 - alphas_cumprod_prev) * jnp.sqrt(alphas) / (1 - alphas_cumprod)

        snr = alphas_cumprod / (1 - alphas_cumprod)

        if loss == 'l2':
            self.loss = l2_loss
        elif loss == 'l1':
            self.loss = l1_loss
        elif loss == 'charbonnier':
            self.loss = charbonnier_loss

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr = jnp.clip(maybe_clipped_snr, a_max=5)

        if objective == 'predict_noise':
            self.loss_weight = maybe_clipped_snr / snr
        elif objective == 'predict_x0':
            self.loss_weight = maybe_clipped_snr / snr  # maybe_clipped_snr
        elif objective == 'predict_v':
            self.loss_weight = maybe_clipped_snr / (snr + 1)
        elif objective == 'predict_mx':
            self.loss_weight = snr ** 0.5

        if p_loss:
            p2_loss_weight_gamma = 1
            p2_loss_weight_k = 1
            p2_loss_weight = (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma
            self.loss_weight = p2_loss_weight

        self.pmap_q_sample = jax.pmap(self.q_sample)
        self.pmap_model_predictions = jax.pmap(self.model_predictions)
        self.pmap_p_sample = jax.pmap(self.p_sample)
        self.spec_max = 2
        self.spec_min = -12

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min
    
    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean

    def set_state(self, state):
        self.state = state

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
        pass

    def model_predictions(self, x, t=None, x_self_cond=None, state=None, rederive_pred_noise=False, *args, **kwargs):
        # model_output = model_predict(state, x, t, x_self_cond)
        #if self.train_state:
        model_output = model_predict(state, x, t, x_self_cond)
        # else:
        #     model_output = model_predict_ema(state, x, t, x_self_cond)

        clip_x_start = self.clip_x_start
        maybe_clip = partial(jnp.clip, a_min=-1., a_max=1.) if clip_x_start else identity

        if self.objective == 'predict_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'predict_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'predict_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        elif self.objective == 'predict_mx':
            x_start = -model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond=None, state=None, *args, **kwargs):
        preds = self.model_predictions(x, t, x_self_cond, state)
        x_start = preds.pred_x_start

        # x_start = jnp.clip(x_start, 0, 1)
        # if clip_denoised:
        #     x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def generate_nosie(self, key, shape):
        if self.noise_type == 'normal':
            return normal_noise(key, shape)
        elif self.noise_type == 'truncate':
            return truncate_noise(key, shape)
        elif self.noise_type == 'resize':
            return resize_noise(key, shape)
        elif self.noise_type == 'offset':
            return offset_noise(key, shape)
        elif self.noise_type == 'pyramid':
            return pyramid_nosie(key, shape)

    def p_sample(self, key, x, batch_times, x_self_cond=None, state=None):
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x, batch_times, x_self_cond, state)
        noise = self.generate_nosie(key, x.shape)
        pred_image = model_mean + jnp.exp(0.5 * model_log_variance) * noise

        return pred_image, x_start
    
    def p_sample_loop(self, key, state, self_condition=None, shape=None):
        key, normal_key = jax.random.split(key, 2)
        img = self.generate_nosie(normal_key, shape)
        img = shard(img)

        x_self_cond = self_condition
        has_condition = False
        if x_self_cond is not None:
            x_self_cond = shard(x_self_cond)
            has_condition = True

        x_start = jnp.zeros_like(img)
        for t in tqdm(reversed(range(0, self.num_timesteps)), total=self.num_timesteps):
            key, normal_key = jax.random.split(key, 2)
            normal_key = shard_prng_key(normal_key)
            batch_times = jnp.full((shape[0],), t)
            batch_times = shard(batch_times)

            if has_condition:
                pass
            elif self.self_condition:
                x_self_cond = x_start
            else:
                x_self_cond = None

            img, x_start = self.pmap_p_sample(normal_key, img, batch_times, x_self_cond, state)

        img = jnp.reshape(img,(-1, *self.sample_shape))

        ret = img

        return ret
    def p_sample_loop_predict(self, key, state, self_condition=None, shape=None):
            key, normal_key = jax.random.split(key, 2)
            img = self.generate_nosie(normal_key, shape)
            #img = shard(img)

            x_self_cond = self_condition
            has_condition = False
            if x_self_cond is not None:
                #x_self_cond = shard(x_self_cond)
                has_condition = True

            x_start = jnp.zeros_like(img)
            for t in tqdm(reversed(range(0, self.num_timesteps)), total=self.num_timesteps):
                key, normal_key = jax.random.split(key, 2)
                #normal_key = shard_prng_key(normal_key)
                batch_times = jnp.full((shape[0],), t)
                #batch_times = shard(batch_times)

                if has_condition:
                    pass
                elif self.self_condition:
                    x_self_cond = x_start
                else:
                    x_self_cond = None

                img, x_start = self.pmap_p_sample(normal_key, img, batch_times, x_self_cond, state)

            img = jnp.reshape(img,(-1, *shape))

            ret = img

            return ret
    def ddim_sample(self, key, state, self_condition=None, shape=None):
        b, *_ = shape
        key, key_image = jax.random.split(key, 2)
        img = self.generate_nosie(key_image, shape=shape)

        times = np.asarray(np.linspace(-1, 999, num=self.sampling_timesteps + 1), dtype=np.int32)
        times = list(reversed(times))

        img = shard(img)

        x_self_cond = self_condition
        has_condition = False
        if x_self_cond is not None:
            x_self_cond = shard(x_self_cond)
            has_condition = True

        x_start = jnp.zeros_like(img)
        for time, time_next in tqdm(zip(times[:-1], times[1:]), total=self.sampling_timesteps):
            batch_times = jnp.full((b,), time)

            if has_condition:
                pass
            elif self.self_condition:
                x_self_cond = x_start
            else:
                x_self_cond = None

            batch_times = shard(batch_times)

            pred_noise, x_start = self.pmap_model_predictions(x=img, t=batch_times, x_self_cond=x_self_cond,
                                                              state=state)

            if time_next < 0:
                img = x_start
            else:
                key, key_noise = jax.random.split(key, 2)
                noise = pred_noise

                # if time_next > 100:
                #     noise = self.generate_nosie(key_noise, shape=shape)
                # else:
                #     noise = pred_noise

                batch_times_next = jnp.full((b,), time_next)
                batch_times_next = shard(batch_times_next)
                img = self.pmap_q_sample(x_start, batch_times_next, noise)

        # img = einops.rearrange(img, 'n b h w c->(n b ) h w c', n=img.shape[0])
        print(img.shape)

        img = jnp.reshape(img,(-1, *self.sample_shape))

        return img

    def sample(self, key, state, self_condition=None, batch_size=64):
        if self_condition is not None:
            batch_size = self_condition.shape[0]

        if self.num_timesteps > self.sampling_timesteps:
            samples = self.ddim_sample(key, state, self_condition, (batch_size, *self.sample_shape))
        else:
            samples = self.p_sample_loop(key, state, self_condition, (batch_size, *self.sample_shape))

        # output image will be denormalized by mean(default as 0) and std(default as 1) because input image was
        # normalized if mean=0 and std=1 img=denormalize(image)
        samples = samples / self.scale_factor
        #samples = self.denormalize(samples)
        samples = self.denorm_spec(samples)

        return samples
    def pred_sample(self, key, state, self_condition=None,shape=None):
        # if self_condition is not None:
        #     batch_size = self_condition.shape[0]

        if self.num_timesteps > self.sampling_timesteps:
            samples = self.ddim_sample(key, state, self_condition, shape)
        else:
            samples = self.p_sample_loop_predict(key, state, self_condition, shape)

        # output image will be denormalized by mean(default as 0) and std(default as 1) because input image was
        # normalized if mean=0 and std=1 img=denormalize(image)
        samples = samples / self.scale_factor
        #samples = self.denormalize(samples)
        samples = self.denorm_spec(samples)

        return samples
    
    def q_sample(self, x_start, t, noise):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_loss(self, key, state, params, x_start, t , x_self_cond):
        key, cond_key = jax.random.split(key, 2)
        noise = self.generate_nosie(key, shape=x_start.shape)

        x_start = x_start * self.scale_factor
        # noise sample
        x = self.q_sample(x_start, t, noise)

        # def estimate(_):
        #     return jax.lax.stop_gradient(
        #         self.model_predictions(x, t, state=state, x_self_cond=jnp.zeros_like(x))).pred_x_start

        #zeros = jnp.zeros_like(x)
        # x_self_cond = None
        # if self.self_condition:
        #     x_self_cond = jax.lax.cond(jax.random.uniform(cond_key, shape=(1,))[0] < 0.5, estimate, lambda _: zeros,None)

        model_output = state.apply_fn({"params": params}, x, t, x_self_cond)

        if self.objective == 'predict_noise':
            target = noise
        elif self.objective == 'predict_x0':
            target = x_start
        elif self.objective == 'predict_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        elif self.objective == 'predict_mx':
            target = -x_start
        else:
            target = None

        p_loss = self.loss(target, model_output).mean()

        #p_loss = (p_loss * extract(self.loss_weight, t, p_loss.shape)).mean()
        return p_loss

    def __call__(self, key, state, params, img , self_cond):
        # input image will be normalized by mean(default as 0) and std(default as 1)
        # if mean=0 and std=1 img=normalize(image)
        #img = self.normalize(img)
        img = self.norm_spec(img)

        key_times, key_noise = jax.random.split(key, 2)
        b, *_ = img.shape
        t = jax.random.randint(key_times, (b,), minval=0, maxval=self.num_timesteps)

        return self.p_loss(key_noise, state, params, img, t , self_cond)

    # def train(self):
    #     self.train_state = True

    # def eval(self):
    #     self.train_state = False
