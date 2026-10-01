"""Guarded MAPPO bridge for the new physical observation/action protocol.

Weights with the legacy 36-D layout must NOT be silently reused just because
tensor shapes agree. Fresh physical policies carry an explicit schema tag.
"""
from __future__ import annotations

import numpy as np
import torch

from marl.geometric_control import direct_local_action
from marl.mappo_advanced import MAPPOAdvanced


def make_physical_agent(env, *, seed=42, hidden_dim=32, device='cpu', ppo=None):
    # Explicit experimental optimizer settings; no silent unknown-key fallback.
    ppo = dict(ppo or {})
    allowed = {'lr_actor', 'lr_critic', 'gae_lambda', 'clip_epsilon', 'value_clip',
               'entropy_coef', 'value_loss_coef', 'max_grad_norm'}
    if set(ppo) - allowed:
        raise ValueError(f'Unknown physical PPO settings: {sorted(set(ppo)-allowed)}')
    for key, value in ppo.items():
        if isinstance(value, bool) or not np.isfinite(value) or value < 0:
            raise ValueError(f'Invalid physical PPO setting: {key}')
        if key != 'entropy_coef' and value == 0:
            raise ValueError(f'Physical PPO setting must be positive: {key}')
        if key in ('gae_lambda', 'clip_epsilon') and value > 1:
            raise ValueError(f'Physical PPO setting exceeds one: {key}')
    torch.manual_seed(seed)
    agent = MAPPOAdvanced(n_agents=env.num_robots, obs_dim=env.obs_dim, action_dim=3,
                         state_dim=env.num_clots*4, hidden_dim=hidden_dim,
                         num_layers=1, architecture='adaptive_edge_gat',
                         control_mode='local', critic_value_mode='v', dropout=0,
                         max_agents=env.num_robots, device=device, gamma=env.config.reward_discount, **ppo)
    agent.meta.update(observation_schema=env.observation_schema,
                      environment_protocol='EXP0022B_engineering',
                      physical_action_semantics='Frenet_unit_to_world_zero_order_hold',
                      hardware_validated=False,
                      ppo_loss_normalization='active_minibatch_v2',
                      ppo_hyperparameters=dict(lr_actor=agent.actor_optimizer.param_groups[0]['lr'],
                          lr_critic=agent.critic_optimizer.param_groups[0]['lr'],
                          **{k:getattr(agent,k) for k in allowed-{'lr_actor','lr_critic'}}))
    return agent


def initialize_expanded_obstacle_policy(agent, checkpoint):
    """Explicit schema expansion; preserve old inputs, zero new input weights."""
    meta=checkpoint['meta']
    schemas = {
        ('mca_point_36_v3', 'mca_point_obstacles_76_v4'): (36, 76),
        ('mca_point_obstacles_76_v4', 'mca_point_trajectories_172_v5'): (76, 172),
        ('mca_point_obstacles_76_v4', 'mca_point_bounded_172_v6'): (76, 172),
        ('mca_point_obstacles_76_v4', 'mca_point_anchored_172_v7'): (76, 172),
        ('mca_point_obstacles_76_v4', 'mca_point_routed_112_v8'): (76, 112),
        ('mca_point_bounded_172_v6', 'mca_point_routed_208_v8'): (172, 208),
    }
    pair = (meta.get('observation_schema'), agent.meta.get('observation_schema'))
    if pair not in schemas:
        raise ValueError('Only explicit point v3->v4, obstacle v4->trajectory v5/v6/v7 '
                         'or v4/v6->routed v8 expansion is supported')
    old_dim, new_dim = schemas[pair]
    for key in ('architecture','n_agents','action_dim','state_dim','hidden_dim','num_layers',
                'critic_value_mode','dropout','action_semantics','physical_action_semantics'):
        if meta.get(key)!=agent.meta.get(key):raise ValueError(f'Expanded initialization {key} mismatch')
    if meta.get('obs_dim')!=old_dim or agent.meta.get('obs_dim')!=new_dim:
        raise ValueError('Expanded observation dimensions mismatch')
    expanded=[];states={}
    for name,net,input_key in (('actor',agent.actor,'encoder.input_proj.weight'),
                              ('critic',agent.critic,'inner.obs_encoder.input_proj.weight')):
        current=net.state_dict();old=checkpoint[name]
        if current.keys()!=old.keys():raise ValueError('Expanded initialization network keys mismatch')
        for key,value in old.items():
            if key==input_key:
                if value.shape!=(current[key].shape[0],old_dim):raise ValueError('Unexpected input projection')
                weight=torch.zeros_like(current[key]);weight[:,:old_dim]=value.to(weight.device)
                current[key]=weight;expanded.append(f'{name}.{key}')
            elif value.shape==current[key].shape:
                current[key]=value
            else:raise ValueError(f'Unsupported weight shape change: {name}.{key}')
        states[name]=current
    agent.actor.load_state_dict(states['actor']);agent.critic.load_state_dict(states['critic'])
    return expanded


def physical_context(env, observation):
    return dict(positions=env.robot_positions.copy(), velocities=env.robot_velocities.copy(),
                adjacency=observation['adjacency'].copy(),
                agent_mask=observation['agent_mask'].astype(bool))


def physical_policy_action(agent, env, observation, *, deterministic=False):
    if (agent.meta.get('observation_schema') != env.observation_schema or
            agent.meta.get('environment_protocol') != 'EXP0022B_engineering' or
            agent.meta.get('physical_action_semantics') != 'Frenet_unit_to_world_zero_order_hold' or
            agent.meta.get('action_semantics') != 'direct_local_frenet'):
        raise RuntimeError('Policy observation/action schema is not the physical MCA protocol')
    context = physical_context(env, observation)
    state = observation['clot_state'].reshape(-1)
    proposal, log_prob, value = agent.act(observation['nodes'], context, state,
                                        deterministic=deterministic)
    execution = direct_local_action(proposal, env)
    execution[~context['agent_mask']] = 0
    return proposal, log_prob, value, execution, context, state


def store_physical_transition(agent, observation, proposal, log_prob, value,
                              context, state, env, next_observation, info,
                              terminated, truncated):
    next_context = physical_context(env, next_observation)
    lost = ~next_context['agent_mask']
    # Time limits bootstrap; physical loss and cleared/all-lost episodes do not.
    done = np.logical_or(lost, terminated or truncated).astype(np.float32)
    terminal = np.logical_or(lost, terminated).astype(np.float32)
    rewards = info['agent_rewards'].copy()
    rewards += info['team_reward'] * context['agent_mask'] / max(context['agent_mask'].sum(), 1)
    agent.buffer.store(observation['nodes'], proposal, rewards, done, log_prob, value,
                       context, state, next_obs=next_observation['nodes'],
                       next_ctx=next_context, next_state=next_observation['clot_state'].reshape(-1),
                       terminals=terminal)
