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
from contrastive import hnetworks as contrastive_networks
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
from collections import namedtuple
# Define expected transition structures
class HighLevelTransition(NamedTuple):
    s_z: jax.Array
    g: jax.Array
    g_neg: jax.Array
    s: jax.Array
    s_g: jax.Array

class LowLevelTransition(NamedTuple):
    s_a: jax.Array
    z: jax.Array
    z_neg: jax.Array
    s: jax.Array
    s_z: jax.Array

class ReplaySample(NamedTuple):
    high: HighLevelTransition
    low: LowLevelTransition

class HierarchicalTrainingState(NamedTuple):
    high_policy_params: Any
    high_policy_opt_state: optax.OptState
    high_q_params: Any
    high_q_opt_state: optax.OptState
    high_target_q_params: Any

    low_policy_params: Any
    low_policy_opt_state: optax.OptState
    low_q_params: Any
    low_q_opt_state: optax.OptState
    low_target_q_params: Any

    log_alpha: jax.Array
    log_alpha_opt_state: optax.OptState

    key: jax.Array

def unpack_high_array(high_array: jax.Array) -> HighLevelTransition:
        # Each column is one field
        s_z = high_array[:, 0]
        g = high_array[:, 1]
        g_neg = high_array[:, 2]
        s = high_array[:, 3]
        s_g = high_array[:, 4]
        return HighLevelTransition(s_z=s_z, g=g, g_neg=g_neg, s=s, s_g=s_g)

def unpack_low_array(low_array: jax.Array) -> LowLevelTransition:
    s_a = low_array[:, 0]
    z = low_array[:, 1]
    z_neg = low_array[:, 2]
    s = low_array[:, 3]
    s_z = low_array[:, 4]
    return LowLevelTransition(s_a=s_a, z=z, z_neg=z_neg, s=s, s_z=s_z)

        
class HierarchicalContrastiveLearner(Learner):
    def __init__(self, hnetworks, rng, high_policy_opt, high_q_opt,
                 low_policy_opt, low_q_opt, alpha_opt,
                 iterator, counter, logger, config):

        self._counter = counter
        self._logger = logger
        self._iterator = iterator
        self._config = config
        self._networks = hnetworks

        self._high_policy_opt = high_policy_opt
        self._high_q_opt = high_q_opt
        self._low_policy_opt = low_policy_opt
        self._low_q_opt = low_q_opt
        self._alpha_opt = alpha_opt

        def init_state(rng):
            rng, k1, k2, k3, k4, k5, k6 = jax.random.split(rng, 7)
            high_policy_params = hnetworks.high_policy_network.init(k1)
            high_q_params = hnetworks.high_q_network.init(k2)
            low_policy_params = hnetworks.low_policy_network.init(k3)
            low_q_params = hnetworks.low_q_network.init(k4)
            log_alpha = jnp.array(0.0)
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
                log_alpha=log_alpha,
                log_alpha_opt_state=alpha_opt.init(log_alpha),
                key=k6
            )

        self._state = init_state(rng)
        self._jit_update = jax.jit(self._update_step)

    def _contrastive_loss(self, phi, psi, pos_obs, pos_subgoal, neg_subgoal):
        sim_pos = jnp.sum(phi(pos_obs, pos_subgoal) * psi(pos_subgoal), axis=-1)
        sim_neg = jnp.einsum("bd,bnd->bn", phi(pos_obs, pos_subgoal), psi(neg_subgoal))
        logits = jnp.concatenate([sim_pos[:, None], sim_neg], axis=1)
        labels = jnp.zeros(logits.shape[0], dtype=jnp.int32)
        return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()


    def _alpha_loss(self, log_alpha, policy_params, transitions, key):
        dist_params = self._networks.low_policy.apply(policy_params, transitions.s_z)
        action = self._networks.low_policy.sample(dist_params, key)
        log_prob = self._networks.low_policy.log_prob(dist_params, action)
        alpha = jnp.exp(log_alpha)
        entropy_diff = -log_prob - self._config.target_entropy
        return jnp.mean(alpha * jax.lax.stop_gradient(entropy_diff))

    def _update_step(self, state: HierarchicalTrainingState, transitions: ReplaySample):
        key, subkey = jax.random.split(state.key)

        # Unpack transition fields BEFORE any JIT-traced functions to avoid tracer attribute errors
        high_s_z = transitions.high.s_z
        high_g_neg = transitions.high.g_neg
        high_s_g = transitions.high.s_g
        high_s = transitions.high.s
        high_g = transitions.high.g

        low_s_a = transitions.low.s_a
        low_z_neg = transitions.low.z_neg
        low_s_z = transitions.low.s_z
        low_s = transitions.low.s
        low_z = transitions.low.z

        # High-level Q loss
        def high_q_loss_fn(q_params):
            phi = lambda obs, g: self._networks.high_q_network.apply(q_params, obs, g)
            psi = lambda g: self._networks.high_goal_encoder(g)
            return self._contrastive_loss(phi, psi, high_s_z, high_g, high_g_neg)

        high_q_loss, high_q_grads = jax.value_and_grad(high_q_loss_fn)(state.high_q_params)
        high_q_updates, high_q_opt_state = self._high_q_opt.update(high_q_grads, state.high_q_opt_state)
        high_q_params = optax.apply_updates(state.high_q_params, high_q_updates)

        # Low-level Q loss
        def low_q_loss_fn(q_params):
            phi = lambda s: self._networks.high_q_network.apply(q_params, s)
            #phi = lambda obs, subgoal: self._networks.low_q_network.apply(q_params, None, obs, subgoal)
            #phi = lambda s, z: self._networks.high_q_network.apply(q_params, None, s, z)
            psi = lambda z: self._networks.low_goal_encoder(z)
            return self._contrastive_loss(phi, psi, low_s_a, low_z_neg)

        low_q_loss, low_q_grads = jax.value_and_grad(low_q_loss_fn)(state.low_q_params)
        low_q_updates, low_q_opt_state = self._low_q_opt.update(low_q_grads, state.low_q_opt_state)
        low_q_params = optax.apply_updates(state.low_q_params, low_q_updates)

        # High-level policy loss
        def high_policy_loss_fn(policy_params):
            z = self._networks.high_policy.apply(policy_params, high_s_g)
            q_val = self._networks.high_q.apply(high_q_params, jnp.concatenate([high_s, z], axis=-1))
            g_encoded = self._networks.high_goal_encoder(high_g)
            return -jnp.mean(jnp.sum(q_val * g_encoded, axis=-1))

        high_policy_loss, high_policy_grads = jax.value_and_grad(high_policy_loss_fn)(state.high_policy_params)
        high_policy_updates, high_policy_opt_state = self._high_policy_opt.update(high_policy_grads, state.high_policy_opt_state)
        high_policy_params = optax.apply_updates(state.high_policy_params, high_policy_updates)

        # Low-level policy loss
        def low_policy_loss_fn(policy_params):
            a = self._networks.low_policy.apply(policy_params, low_s_z)
            q_val = self._networks.low_q.apply(low_q_params, jnp.concatenate([low_s, a], axis=-1))
            z_encoded = self._networks.low_goal_encoder(low_z)
            return -jnp.mean(jnp.sum(q_val * z_encoded, axis=-1))

        low_policy_loss, low_policy_grads = jax.value_and_grad(low_policy_loss_fn)(state.low_policy_params)
        low_policy_updates, low_policy_opt_state = self._low_policy_opt.update(low_policy_grads, state.low_policy_opt_state)
        low_policy_params = optax.apply_updates(state.low_policy_params, low_policy_updates)

        # Alpha loss and update
        alpha_loss_val, alpha_grads = jax.value_and_grad(self._alpha_loss)(
            state.log_alpha, low_policy_params, transitions.low, subkey)
        alpha_updates, log_alpha_opt_state = self._alpha_opt.update(alpha_grads, state.log_alpha_opt_state)
        log_alpha = optax.apply_updates(state.log_alpha, alpha_updates)

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
            log_alpha=log_alpha,
            log_alpha_opt_state=log_alpha_opt_state,
            key=key
        )

        metrics = {
            "high_q_loss": high_q_loss,
            "low_q_loss": low_q_loss,
            "high_policy_loss": high_policy_loss,
            "low_policy_loss": low_policy_loss,
            "alpha_loss": alpha_loss_val,
            "alpha_value": jnp.exp(log_alpha),
        }

        return new_state, metrics
    
    
    def step(self):
        sample = next(self._iterator)
        sample_data = sample.data

        try:
            high = sample_data.extras['high']
            low = sample_data.extras['low']
            

        except (AttributeError, KeyError, TypeError) as e:
            raise ValueError(
                f"Expected `ReplaySample.data.extras` to contain `high` and `low`, "
                f"but got: {type(sample_data.extras)} with keys: "
                f"{list(sample_data.extras.keys()) if hasattr(sample_data.extras, 'keys') else sample_data.extras}"
            ) from e

       
        high_trans = unpack_high_array(high)
        low_trans = unpack_low_array(low)
        sample_namedtuple = ReplaySample(high=high_trans, low=low_trans)
        self._state, metrics = self._jit_update(self._state, sample_namedtuple)

        # Logging and counting
        if self._counter:
            counts = self._counter.increment(steps=1)
        else:
            counts = {}

        if self._logger:
            self._logger.write({**metrics, **counts})

        return metrics


    def get_variables(self, names):
        mapping = {
            'policy': ['high_policy_params', 'low_policy_params'],
            'high_policy': 'high_policy_params',
            'low_policy': 'low_policy_params',
            'high_q': 'high_q_params',
            'low_q': 'low_q_params',
        }
        variables = []
        for name in names:
            if name == 'policy':
                variables.extend([
                    getattr(self._state, 'high_policy_params'),
                    getattr(self._state, 'low_policy_params'),
                ])
            else:
                attr = mapping.get(name, f"{name}_params")
                if isinstance(attr, list):
                    variables.extend([getattr(self._state, a) for a in attr])
                else:
                    variables.append(getattr(self._state, attr))
        return variables


    def save(self):
        return self._state

    def restore(self, state):
        if not isinstance(state, HierarchicalTrainingState):
            raise TypeError(f"Expected HierarchicalTrainingState, got {type(state)}")
        self._state = state

