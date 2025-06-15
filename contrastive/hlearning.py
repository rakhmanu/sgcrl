"""H-CRL learner implementation."""
import time
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple, Callable

import acme
from acme.core import Learner
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
from contrastive import config as contrastive_config
from contrastive import networks as contrastive_networks
import jax
import jax.numpy as jnp
import optax
import reverb
from jax.experimental.host_callback import id_print
from jax import debug
from jax.scipy.special import logsumexp
import numpy as np
from jax import random
import os
from default import make_default_logger
from pathlib import Path

class HierarchicalTrainingState(NamedTuple):
    # High-level
    high_policy_params: any
    high_policy_opt_state: optax.OptState
    high_q_params: any
    high_q_opt_state: optax.OptState
    high_target_q_params: any

    # Low-level
    low_policy_params: any
    low_policy_opt_state: optax.OptState
    low_q_params: any
    low_q_opt_state: optax.OptState
    low_target_q_params: any

    # RNG key
    key: jax.Array

class HierarchicalContrastiveLearner(Learner):
    def __init__(self, networks, rng, high_policy_opt, high_q_opt,
                 low_policy_opt, low_q_opt, iterator, counter, logger, config):

        self._counter = counter
        self._logger = logger
        self._iterator = iterator
        self._config = config
        self._networks = networks

        def init_state(rng):
            rng, k1, k2, k3, k4, k5 = jax.random.split(rng, 6)
            high_policy_params = networks.high_policy.init(k1)
            high_q_params = networks.high_q.init(k2)
            low_policy_params = networks.low_policy.init(k3)
            low_q_params = networks.low_q.init(k4)

            return HierarchicalTrainingState(
                high_policy_params=high_policy_params,
                high_policy_opt_state=high_policy_opt.init(high_policy_params),
                high_q_params=high_q_params,
                high_q_opt_state=high_q_opt.init(high_q_params),
                high_target_q_params=high_q_params,

                low_policy_params=low_policy_params,
                low_policy_opt_state=low_policy_opt.init(low_policy_params),
                low_q_params=low_q_params,
                low_q_opt_state=low_q_opt.init(low_q_params),
                low_target_q_params=low_q_params,

                key=k5
            )

        self._state = init_state(rng)
        self._jit_update = jax.jit(self._update_step)

    def _contrastive_loss(self, phi, psi, pos, neg):
        # InfoNCE loss: -log (exp(sim(pos)) / sum exp(sim(pos) + sim(neg)))
        sim_pos = jnp.sum(phi(pos) * psi(pos), axis=-1)
        sim_neg = jnp.matmul(phi(pos), psi(neg).T)
        logits = jnp.concatenate([sim_pos[:, None], sim_neg], axis=1)
        labels = jnp.zeros(logits.shape[0], dtype=jnp.int32)
        return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    def _update_step(self, state, transitions):
        key, subkey = jax.random.split(state.key)
        
        # Extract high-level and low-level transitions
        high_trans = transitions.high  # (s_t, z, g, next_s)
        low_trans = transitions.low    # (s_t, a_t, z, next_s)

        # High-level Q loss
        def high_q_loss_fn(q_params):
            phi = lambda s_z: self._networks.high_q.apply(q_params, s_z)
            psi = lambda g: self._networks.high_goal_encoder(g)
            return self._contrastive_loss(phi, psi, high_trans.s_z, high_trans.g_neg)

        high_q_loss, high_q_grads = jax.value_and_grad(high_q_loss_fn)(state.high_q_params)
        high_q_updates, high_q_opt_state = self._config.high_q_opt.update(high_q_grads, state.high_q_opt_state)
        high_q_params = optax.apply_updates(state.high_q_params, high_q_updates)

        # Low-level Q loss
        def low_q_loss_fn(q_params):
            phi = lambda s_a: self._networks.low_q.apply(q_params, s_a)
            psi = lambda z: self._networks.low_goal_encoder(z)
            return self._contrastive_loss(phi, psi, low_trans.s_a, low_trans.z_neg)

        low_q_loss, low_q_grads = jax.value_and_grad(low_q_loss_fn)(state.low_q_params)
        low_q_updates, low_q_opt_state = self._config.low_q_opt.update(low_q_grads, state.low_q_opt_state)
        low_q_params = optax.apply_updates(state.low_q_params, low_q_updates)

        # High-level policy loss: maximize Q(s, z, g)
        def high_policy_loss_fn(policy_params):
            z = self._networks.high_policy.apply(policy_params, high_trans.s_g)
            q_val = self._networks.high_q.apply(high_q_params, jnp.concatenate([high_trans.s, z], axis=-1))
            g_encoded = self._networks.high_goal_encoder(high_trans.g)
            return -jnp.mean(jnp.sum(q_val * g_encoded, axis=-1))

        high_policy_loss, high_policy_grads = jax.value_and_grad(high_policy_loss_fn)(state.high_policy_params)
        high_policy_updates, high_policy_opt_state = self._config.high_policy_opt.update(high_policy_grads, state.high_policy_opt_state)
        high_policy_params = optax.apply_updates(state.high_policy_params, high_policy_updates)

        # Low-level policy loss: maximize Q(s, a, z)
        def low_policy_loss_fn(policy_params):
            a = self._networks.low_policy.apply(policy_params, low_trans.s_z)
            q_val = self._networks.low_q.apply(low_q_params, jnp.concatenate([low_trans.s, a], axis=-1))
            z_encoded = self._networks.low_goal_encoder(low_trans.z)
            return -jnp.mean(jnp.sum(q_val * z_encoded, axis=-1))

        low_policy_loss, low_policy_grads = jax.value_and_grad(low_policy_loss_fn)(state.low_policy_params)
        low_policy_updates, low_policy_opt_state = self._config.low_policy_opt.update(low_policy_grads, state.low_policy_opt_state)
        low_policy_params = optax.apply_updates(state.low_policy_params, low_policy_updates)

        new_state = HierarchicalTrainingState(
            high_policy_params=high_policy_params,
            high_policy_opt_state=high_policy_opt_state,
            high_q_params=high_q_params,
            high_q_opt_state=high_q_opt_state,
            high_target_q_params=high_q_params,

            low_policy_params=low_policy_params,
            low_policy_opt_state=low_policy_opt_state,
            low_q_params=low_q_params,
            low_q_opt_state=low_q_opt_state,
            low_target_q_params=low_q_params,

            key=key
        )

        metrics = {
            "high_q_loss": high_q_loss,
            "low_q_loss": low_q_loss,
            "high_policy_loss": high_policy_loss,
            "low_policy_loss": low_policy_loss
        }

        return new_state, metrics

    def step(self):
        sample = next(self._iterator)
        self._state, metrics = self._jit_update(self._state, sample)

        counts = self._counter.increment(steps=1)
        self._logger.write({**metrics, **counts})

    def get_variables(self, names):
        return [getattr(self._state, f"{name}_params") for name in names]

    def save(self):
        return self._state

    def restore(self, state):
        self._state = state

