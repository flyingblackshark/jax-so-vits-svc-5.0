import os

import einops
import jax
import jax.numpy as jnp
import numpy as np
#import torch
#import torchvision



def normal_noise(key, shape):
    return jax.random.normal(key, shape)


def offset_noise(key, shape, offset=0.1):
    b, h, w, c = shape
    key1, key2 = jax.random.split(key, 2)
    return jax.random.normal(key1, shape) + jax.random.normal(key2, (b, 1, 1, c)) * offset


def truncate_noise(key, shape, low=-2.5, high=2.5):
    return jax.random.truncated_normal(key, low, high, shape)


def resize_noise(key, shape):
    b, h, w, c = shape
    noise = jax.random.normal(key, (b, 1024, 1024, c))
    return jax.image.resize(noise, shape, 'nearest')


# def pyramid_noise_like(key, shape, discount=0.9):
#     b, c, w, h = shape  # EDIT: w and h get over-written, rename for a different variant!
#     key, key_normal = jax.random.split(key, 2)
#     noise = jax.random.normal(key, shape)
#     # u = nn.Upsample(size=(w, h), mode='bilinear')
#     r = 1
#     for i in range(10):
#         print(i)
#         key, key_normal = jax.random.split(key, 2)
#         # r = jax.random.uniform(key,(1,))[0]*2+2
#
#         r = r * 2 #+ 2
#
#
#         # r = random.random() * 2 + 2  # Rather than always going 2x,
#         w, h = max(1, int(w / (r ))), max(1, int(h / (r )))
#         key, key_normal = jax.random.split(key, 2)
#         new_noise = jax.random.normal(key, shape=(b, w, h, c))
#         noise += jax.image.resize(new_noise, shape, method='bilinear') * discount ** i
#         if w == 1 or h == 1: break  # Lowest resolution is 1x1
#     return noise  # / noise.std()  # Scaled back to roughly unit variance


def pyramid_nosie(key, shape,discount = 0.9):

    b, h, w, c = shape
    key, noise_key = jax.random.split(key, 2)

    noise = normal_noise(noise_key, shape)
    for i in range(100):
        key, noise_key = jax.random.split(key, 2)
        r = 2  # random.random() * 2 +
        w, h = max(1, int(w / (r ** i))), max(1, int(h / (r ** i)))  #
        new_shape = (b, w, h, c)
        new_noise = normal_noise(noise_key, new_shape)
        noise += jax.image.resize(new_noise, shape, method='bilinear') * discount ** i

        if w == 1 or h == 1: break
    return noise / noise.std()


if __name__ == '__main__':
    pass

    # dl = get_dataloader(8, '/home/john/data/s', cache=False, image_size=1024, repeat=2)
    #
    # os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'
    # # for _ in range(100):
    # #     for data in dl:
    # #         print(type(data))
    # #         # print(data.shape)
    # key = jax.random.PRNGKey(55)
    # alpha = 0.9
    # for data in dl:
    #     print(data.shape)
    #     data = data / 2 + 0.5
    #     data = data.numpy()
    #     data = jnp.asarray(data)
    #
    #     noise = pyramid_nosie(key, data.shape)
    #
    #     data = alpha * data + (1 - alpha) * noise
    #     # data = noise
    #
    #     data = np.asarray(data)
    #     data = torch.Tensor(data)
    #
    #     data = einops.rearrange(data, '(n b) h w c->(b n) c h w', n=2)
    #     torchvision.utils.save_image(data, './test1.png')
    #     break
