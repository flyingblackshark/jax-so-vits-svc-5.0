import optax
import jax.numpy as jnp
import flax.linen as nn


def charbonnier_loss(predictions, target, eps=1e-3):
    return jnp.sqrt((target - predictions) ** 2 + eps)


def l2_loss(predictions, target):
    return optax.l2_loss(predictions=predictions, targets=target)


def l1_loss(predictions, target):
    return jnp.abs(target - predictions)


def hinge_d_loss(logits_real, logits_fake):
    loss_real = jnp.mean(nn.relu(1. - logits_real))
    loss_fake = jnp.mean(nn.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss
