"""H-CRL networks definition."""
import dataclasses
from typing import Optional, Tuple, Callable

from acme import specs
from acme.agents.jax import actor_core as actor_core_lib
from acme.jax import networks as networks_lib
from acme.jax import utils
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from itertools import product


# modified Tanh mean to be mapped to tanh(mean) to keep within [-1, 1]
from distributional import NormalTanhDistribution

@dataclasses.dataclass
class HierarchicalContrastiveNetworks:
  """Hierarchical networks for Contrastive RL."""
  high_policy_network: networks_lib.FeedForwardNetwork   # π_high(z | s, g)
  low_policy_network: networks_lib.FeedForwardNetwork    # π_low(a | s, z)
  high_q_network: networks_lib.FeedForwardNetwork        # Q_high(s, z, g)
  low_q_network: networks_lib.FeedForwardNetwork         # Q_low(s, a, z)
  high_repr_fn: Callable
  low_repr_fn: Callable
  high_sample: networks_lib.SampleFn
  low_sample: networks_lib.SampleFn
  high_sample_eval: Optional[networks_lib.SampleFn]
  low_sample_eval: Optional[networks_lib.SampleFn]
  log_prob: networks_lib.LogProbFn
  
def apply_policy_and_sample(
    networks,
    eval_mode = False):
  """Returns a function that computes actions."""
  sample_fn = networks.sample if not eval_mode else networks.sample_eval
  if not sample_fn:
    raise ValueError('sample function is not provided')

  def apply_and_sample(params, key, obs):
    return sample_fn(networks.policy_network.apply(params, obs), key)
  return apply_and_sample

def make_hcrl_networks(
    spec,
    obs_dim,
    subgoal_dim,
    repr_dim=64,
    repr_norm=True,
    hidden_layer_sizes=(256, 256),
    actor_min_std=1e-6,
):
  """Creates hierarchical contrastive networks."""

  def mlp(output_dim):
    return hk.nets.MLP(
        list(hidden_layer_sizes) + [output_dim],
        w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
        activation=jax.nn.relu)

  # Low-level Q: Q_low(s, a, z) = φ(s, a)^T ψ(z)
  def low_repr_fn(obs, action):
    s = obs[:, :obs_dim]
    z = obs[:, obs_dim:]
    sa_repr = mlp(repr_dim)(jnp.concatenate([s, action], axis=-1))
    z_repr = mlp(repr_dim)(z)

    if repr_norm:
      sa_repr /= jnp.linalg.norm(sa_repr, axis=-1, keepdims=True)
      z_repr /= jnp.linalg.norm(z_repr, axis=-1, keepdims=True)

    return sa_repr, z_repr

  def low_q_fn(obs, action):
    sa_repr, z_repr = low_repr_fn(obs, action)
    q_val = jnp.sum(sa_repr * z_repr, axis=-1)
    return q_val, sa_repr, z_repr

  # High-level Q: Q_high(s, z, g) = φ(s, z)^T ψ(g)
  def high_repr_fn(obs, subgoal):
    s = obs[:obs_dim]
    g = obs[obs_dim:]
    sz_repr = mlp(repr_dim)(jnp.concatenate([s, subgoal], axis=-1))
    g_repr = mlp(repr_dim)(g)

    if repr_norm:
        sz_repr /= jnp.linalg.norm(sz_repr, axis=-1, keepdims=True)
        g_repr /= jnp.linalg.norm(g_repr, axis=-1, keepdims=True)
    jax.debug.print("obs shape: {}", obs.shape)
    
    return sz_repr, g_repr

  def high_q_fn(obs, subgoal):
    sz_repr, g_repr = high_repr_fn(obs, subgoal)
    q_val = jnp.sum(sz_repr * g_repr, axis=-1)
    return q_val, sz_repr, g_repr


  # Actor Networks
  def high_actor_fn(obs):
    s = obs[:, :obs_dim]
    g = obs[:, obs_dim:]
    inp = jnp.concatenate([s, g], axis=-1)
    net = hk.Sequential([
        hk.nets.MLP(list(hidden_layer_sizes), activation=jax.nn.relu, activate_final=True),
        NormalTanhDistribution(subgoal_dim, min_scale=actor_min_std),
    ])
    return net(inp)

  def low_actor_fn(obs):
    s = obs[:, :obs_dim]
    z = obs[:, obs_dim:]
    inp = jnp.concatenate([s, z], axis=-1)
    net = hk.Sequential([
        hk.nets.MLP(list(hidden_layer_sizes), activation=jax.nn.relu, activate_final=True),
        NormalTanhDistribution(np.prod(spec.actions.shape), min_scale=actor_min_std),
    ])
    return net(inp)

  # Haiku transforms
  high_policy = hk.without_apply_rng(hk.transform(high_actor_fn))
  low_policy = hk.without_apply_rng(hk.transform(low_actor_fn))
  high_q = hk.without_apply_rng(hk.transform(high_q_fn))
  low_q = hk.without_apply_rng(hk.transform(low_q_fn))
  high_repr = hk.without_apply_rng(hk.transform(high_repr_fn))
  low_repr = hk.without_apply_rng(hk.transform(low_repr_fn))

  # Dummy inputs
  dummy_obs = utils.add_batch_dim(utils.zeros_like(spec.observations))
  dummy_subgoal = utils.add_batch_dim(jnp.zeros((subgoal_dim,), dtype=jnp.float32))
  dummy_action = utils.add_batch_dim(utils.zeros_like(spec.actions))

  return HierarchicalContrastiveNetworks(
      high_policy_network=networks_lib.FeedForwardNetwork(
          lambda key: high_policy.init(key, dummy_obs), high_policy.apply),
      low_policy_network=networks_lib.FeedForwardNetwork(
          lambda key: low_policy.init(key, dummy_obs), low_policy.apply),
      high_q_network=networks_lib.FeedForwardNetwork(
          lambda key: high_q.init(key, dummy_obs, dummy_subgoal), high_q.apply),
      low_q_network=networks_lib.FeedForwardNetwork(
          lambda key: low_q.init(key, dummy_obs, dummy_action), low_q.apply),
      high_repr_fn=high_repr.apply,
      low_repr_fn=low_repr.apply,
      high_sample=lambda params, key: params.sample(seed=key),
      low_sample=lambda params, key: params.sample(seed=key),
      high_sample_eval=lambda params, key: params.mode(),
      low_sample_eval=lambda params, key: params.mode(),
      log_prob=lambda params, actions: params.log_prob(actions),
  )
