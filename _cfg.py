
import gym
import math
import os, sys

# config and local imports
from envs.ContCartpoalEnv import ContinuousCartPoleEnv
from envs.ReachingDotEnv import ReachingDotEnv
from _nets import _mlp_actor_net_orth, _mlp_critic_net_orth
from _utils import RolloutInverseTimeDecay, StepLambda


IMPORT_SIM_FRAMEWORK = True
if IMPORT_SIM_FRAMEWORK:
  sys.path.append('../SimulationFramework/simulation/src/')
  sys.path.append('../SimulationFramework/simulation/src/gym_envs/mujoco/')
  from gym_envs.mujoco.reach_env import ReachEnv
  from envs.throw_env import ThrowEnv, RandomThrowEnv, FourTrayThrowEnv, FourTrayReachThrowEnv, TossEnv





""" 
  base config, environment has to set in subconfigs
"""
base_cfg = {
  # ---- NET/TF CONFIG ----
  'actor_model' : _mlp_actor_net_orth(),
  'adam_actor_alpha' : StepLambda(lambda step: 3e-4),
  'adam_actor_epsilon' : 1e-5,

  'critic_model' : _mlp_critic_net_orth(),
  'adam_critic_alpha' : StepLambda(lambda step: 3e-4),
  'adam_critic_epsilon' : 1e-5,
  
  'model_save_interval' : 10000,
  'model_load_path_prefix' : None,
  'tb_log_graph' : True,
  
  # ---- LOSS CALCULATION ----
  'ppo_clip' : (lambda step: 0.2),
  'entropy_factor' : (lambda step: 1e-3),
  
  'value_loss_factor' : 1.0,
  'vest_clip' : (lambda step: 0.2),
  
  'actor_regloss_factor' : 1e-4,
  'critic_regloss_factor' : 1e-4,
  
  'clip_policy_gradient_norm' : 0.5,
  
  # ---- TRAINING ----
  'epochs' : 3,
  'batchsize' : 64,
  'shuffle' : False,
  'permutate' : True,
  'total_steps' : 1000000,
  'rollout' : 1024,
  
  'gae_gamma' : 0.99,               # reward discount factor
  'gae_lambda' : 0.95,              # smoothing for advantage, reducing variance in training
  
  # ---- ENVIRONMENT ----
  'environment' : None,
  'normalize_advantages' : True,     # minibatch advantage normalization
  'normalize_observations' : True,   # running mean + variance normalization
  'normalize_rewards' : True,        # running variance normalization
  'scale_actions' : True,

  'clip_observations' : 10.0,
  'clip_rewards' : 10.0,
  'clip_eplength' : None,
  
  'gamma_env_normalization' : 0.99,
  
  # ---- NUMERICAL ----
  'num_stab_envnorm' : 1e-8,      # numerical stab. normalization wrapper environment
  'num_stab_advnorm' : 1e-10,     # numerical stab. batch-wise advantage normalization
}

mujoco_base_cfg = {
  **base_cfg,
  
  # ---- NET/TF CONFIG ----
  'actor_model' : _mlp_actor_net_orth(hidden_layers=[128, 128]),
  'adam_actor_alpha' : StepLambda(lambda step: 3e-4 * (1.0 - step / 1e6)),
  'critic_model' : _mlp_actor_net_orth(hidden_layers=[128, 128]),
  'adam_critic_alpha' : StepLambda(lambda step: 3e-4 * (1.0 - step / 1e6)),

  # ---- LOSS CALCULATION ----
  'entropy_factor' : (lambda step: 0e-3),
  
  'actor_regloss_factor' : 0e-4,
  'critic_regloss_factor' : 0e-4,
  
  # ---- TRAINING ----
  'epochs' : 10,
  'batchsize' : 32,
  'total_steps' : 2000000,
  'rollout' : 2048 * 8, # about n_steps * n_env
}

atari_base_cfg = {
  **base_cfg,

  # ---- NET/TF CONFIG ----
  'adam_actor_alpha' : StepLambda(lambda step: 2.5e-4),
  'adam_critic_alpha' : StepLambda(lambda step: 2.5e-4),
  
  # ---- LOSS CALCULATION ----
  'ppo_clip' : (lambda step: 0.1),
  'entropy_factor' : (lambda step: 1e-2),
  
  #'actor_regloss_factor' : 1e-4,
  #'critic_regloss_factor' : 1e-4,
  
  # ---- TRAINING ----
  'epochs' : 4,
  'batchsize' : 4,
  'total_steps' : 1000000,
  'rollout' : 128,
}


# ------------------------------------------------ ENVIRONMENT EXAMPLES ------------------------------------------------
cont_cartpoal_cfg = {
  **base_cfg,

  'environment' : (lambda : ContinuousCartPoleEnv()),
  
  # ---- NET/TF CONFIG ----
  'adam_actor_alpha' : RolloutInverseTimeDecay(3e-4, 50000, 1.0, staircase=False),
  'adam_critic_alpha' : RolloutInverseTimeDecay(3e-4, 50000, 1.0, staircase=False),
  
  # ---- TRAINING ----
  'epochs' : 5,
  'batchsize' : 32,
  'total_steps' : 200000,
  'rollout' : 2048,
}

reaching_dot_cfg = {
  **base_cfg,

  'environment' : (lambda : ReachingDotEnv()),

  'adam_actor_alpha' : RolloutInverseTimeDecay(3e-4, 50000, 1.0, staircase=False),
  'adam_critic_alpha' : RolloutInverseTimeDecay(3e-4, 50000, 1.0, staircase=False),
  
  # ---- TRAINING ----
  'total_steps' : 200000,
}

cartpole_v1_cfg = {
  **base_cfg,

  'environment' : (lambda : gym.make('CartPole-v1')),
  
  # ---- NET/TF CONFIG ----
  'adam_actor_alpha' : RolloutInverseTimeDecay(3e-4, 50000, 1.0, staircase=False),
  'adam_critic_alpha' : RolloutInverseTimeDecay(3e-4, 50000, 1.0, staircase=False),
  
  # ---- TRAINING ----
  'epochs' : 5,
  'batchsize' : 32,
  'total_steps' : 200000,
  'rollout' : 2048
}

pendulum_v0_cfg = {
  **base_cfg,

  'environment' : (lambda : gym.make('Pendulum-v0')),

  # ---- NET/TF CONFIG ----
  'adam_actor_alpha' : RolloutInverseTimeDecay(1e-3, 100000, 0.5, staircase=False),
  'adam_critic_alpha' : RolloutInverseTimeDecay(1e-3, 100000, 0.5, staircase=False),
  'actor_model' : _mlp_actor_net_orth([64]),
  'critic_model' : _mlp_critic_net_orth([64]),

  # ---- LOSS CALCULATION ----
  'entropy_factor' : (lambda step: 0e-3),

  'actor_regloss_factor' : 0e-4,
  'critic_regloss_factor' : 0e-4,

  # ---- TRAINING ----
  'epochs' : 4,
  'batchsize' : 32,
  'total_steps' : 5000000,
  'rollout' : 256,

  'gae_gamma' : 0.95,               # reward discount factor
  'gae_lambda' : 0.95,               # smoothing for advantage, reducing variance in training
}

reach_env_nonrandom_cfg = {
  **base_cfg,
  
  'environment' : (lambda : ReachEnv(max_steps=50, render=False, randomize_objects=False)),
  
  # ---- NET/TF CONFIG ----
  'adam_actor_alpha' : RolloutInverseTimeDecay(3e-4, 100000, 1.0, staircase=False),
  'adam_critic_alpha' : RolloutInverseTimeDecay(3e-4, 100000, 1.0, staircase=False),

  # ---- LOSS CALCULATION ----
  'entropy_factor' : (lambda step: 1e-3),
  
  'actor_regloss_factor' : 0e-4,
  'critic_regloss_factor' : 0e-4,
  
  # ---- TRAINING ----
  'epochs' : 10,
  'batchsize' : 32,
  'total_steps' : 500000,
  'rollout' : 2048,
}

reach_env_random_cfg = {
  **base_cfg,
  
  'environment' : (lambda : ReachEnv(max_steps=75, render=True, randomize_objects=True)),
  
  # ---- NET/TF CONFIG ----
  'adam_actor_alpha' : RolloutInverseTimeDecay(3e-4, 100000, 1.0, staircase=False),
  'adam_critic_alpha' : RolloutInverseTimeDecay(3e-4, 100000, 1.0, staircase=False),

  # ---- LOSS CALCULATION ----
  'entropy_factor' : (lambda step: 1e-3),
  
  'actor_regloss_factor' : 0e-4,
  'critic_regloss_factor' : 0e-4,
  
  # ---- TRAINING ----
  'epochs' : 10,
  'batchsize' : 32,
  'total_steps' : 5000000,
  'rollout' : 2048,
}
# ------------------------------------------------ TODO ENVIRONMENT EXAMPLES ------------------------------------------------

mountaincar_v0_cfg = {
  **base_cfg,
  
  'environment' : (lambda : gym.make('MountainCar-v0')),

  # ---- LOSS CALCULATION ----
  'entropy_factor' : (lambda step: 0e-3),
  
  'actor_regloss_factor' : 0e-4,
  'critic_regloss_factor' : 0e-4,
  
  # ---- TRAINING ----
  'epochs' : 4,
  'batchsize' : 16,
  'total_steps' : 1000000,
  'rollout' : 16,

  'gae_gamma' : 0.99,               # reward discount factor
  'gae_lambda' : 0.98,               # smoothing for advantage, reducing variance in training
}





throw_env_cfg = {
  **base_cfg,
  
  'environment' : (lambda : ThrowEnv(max_steps=50, render=True, randomize_objects=False, trajectory_length=35, target_min_dist=0.15)),
  
  # ---- NET/TF CONFIG ----
  'adam_actor_alpha' : RolloutInverseTimeDecay(3e-4, 100000, 1.0, staircase=False),
  'adam_critic_alpha' : RolloutInverseTimeDecay(3e-4, 100000, 1.0, staircase=False),

  'actor_model' : _mlp_actor_net_orth([64, 64]),
  'critic_model' : _mlp_critic_net_orth([64, 64]),
  
  # ---- LOSS CALCULATION ----
  'entropy_factor' : (lambda step: 1e-3),
  
  'actor_regloss_factor' : 0e-4,
  'critic_regloss_factor' : 0e-4,
  
  'normalize_advantages' : True,     # minibatch advantage normalization
  'normalize_observations' : True,   # running mean + variance normalization
  'normalize_rewards' : True,        # running variance normalization
  'scale_actions' : True,
  
  'gae_gamma' : 0.99,               # reward discount factor
  'gae_lambda' : 0.95,              # smoothing for advantage, reducing variance in training  
  'gamma_env_normalization' : 0.99,
  
  
  # ---- TRAINING ----
  'epochs' : 10,
  'batchsize' : 64,
  'total_steps' : 3000000,
  'rollout' : 2048,
}

random_throw_env_cfg = {
  **base_cfg,
  
  'environment' : (lambda : RandomThrowEnv(max_steps=50, render=True, randomize_objects=False, trajectory_length=35, target_min_dist=0.15)),
  
  # ---- NET/TF CONFIG ----
  'adam_actor_alpha' : RolloutInverseTimeDecay(3e-4, 100000, 1.0, staircase=False),
  'adam_critic_alpha' : RolloutInverseTimeDecay(3e-4, 100000, 1.0, staircase=False),

  'actor_model' : _mlp_actor_net_orth([128, 128]),
  'critic_model' : _mlp_critic_net_orth([128, 128]),
  
  # ---- LOSS CALCULATION ----
  'entropy_factor' : (lambda step: 1e-3),
  
  'actor_regloss_factor' : 0e-4,
  'critic_regloss_factor' : 0e-4,
  
  'normalize_advantages' : True,     # minibatch advantage normalization
  'normalize_observations' : True,   # running mean + variance normalization
  'normalize_rewards' : True,        # running variance normalization
  'scale_actions' : True,
  
  'gae_gamma' : 0.999,               # reward discount factor
  'gae_lambda' : 0.95,              # smoothing for advantage, reducing variance in training  
  'gamma_env_normalization' : 0.99,
  
  
  # ---- TRAINING ----
  'epochs' : 10,
  'batchsize' : 64,
  'total_steps' : 5000000,
  'rollout' : 2048,
}

four_tray_throw_env_cfg = {
  **base_cfg,
  
  'environment' : (lambda : FourTrayThrowEnv(max_steps=60, render=False, randomize_objects=False, trajectory_length=35, target_min_dist=0.1, trays_center=[0.4, 0.0], trays_stride=[0.15, 0.15])),
  
  # ---- NET/TF CONFIG ----
  'adam_actor_alpha' : StepLambda(lambda step: 3e-4 * (1.0 - step / 1e6)),
  'adam_critic_alpha' : StepLambda(lambda step: 3e-4 * (1.0 - step / 1e6)),

  'actor_model' : _mlp_actor_net_orth([128, 128]),
  'critic_model' : _mlp_critic_net_orth([128, 128]),
  
  # ---- LOSS CALCULATION ----
  'entropy_factor' : (lambda step: 1e-3),
  
  'actor_regloss_factor' : 0e-4,
  'critic_regloss_factor' : 0e-4,
  
  'normalize_advantages' : True,     # minibatch advantage normalization
  'normalize_observations' : True,   # running mean + variance normalization
  'normalize_rewards' : True,        # running variance normalization
  'scale_actions' : True,
  
  'gae_gamma' : 0.999,               # reward discount factor
  'gae_lambda' : 0.95,              # smoothing for advantage, reducing variance in training  
  'gamma_env_normalization' : 0.99,
  
  
  # ---- TRAINING ----
  'epochs' : 10,
  'batchsize' : 64,
  'total_steps' : 1000000,
  'rollout' : 2048,
}


four_tray_reach_throw_env_cfg = {
  **base_cfg,
  
  'environment' : (lambda : FourTrayReachThrowEnv(max_steps=130, render=False, randomize_objects=False, trajectory_length=35, target_min_dist=0.1, table_position=[0.4, 0.2], trays_center=[0.4, 0.0], trays_stride=[0.15, 0.15])),
  
  # ---- NET/TF CONFIG ----
  'adam_actor_alpha' : StepLambda(lambda step: 3e-4 * (1.0 - step / 2e6)),
  'adam_critic_alpha' : StepLambda(lambda step: 3e-4 * (1.0 - step / 2e6)),

  'actor_model' : _mlp_actor_net_orth([192, 192]),
  'critic_model' : _mlp_critic_net_orth([192, 192]),
  
  # ---- LOSS CALCULATION ----
  'entropy_factor' : (lambda step: 1e-3),
  
  'actor_regloss_factor' : 0e-4,
  'critic_regloss_factor' : 0e-4,
  
  'normalize_advantages' : True,     # minibatch advantage normalization
  'normalize_observations' : True,   # running mean + variance normalization
  'normalize_rewards' : True,        # running variance normalization
  'scale_actions' : True,
  
  'gae_gamma' : 0.999,               # reward discount factor
  'gae_lambda' : 0.95,              # smoothing for advantage, reducing variance in training  
  'gamma_env_normalization' : 0.99,
  
  
  # ---- TRAINING ----
  'epochs' : 10,
  'batchsize' : 64,
  'total_steps' : 1000000,
  'rollout' : 2048,
}


toss_env_cfg = {
  **base_cfg,
  
  'environment' : (lambda : TossEnv(max_steps=100, render=True, randomize_objects=False, trajectory_length=35, target_min_dist=0.1, tray_position=[1.5, 0.0, 0.0])),
  
  # ---- NET/TF CONFIG ----
  'adam_actor_alpha' : StepLambda(lambda step: 3e-4 * (1.0 - step / 2e6)),
  'adam_critic_alpha' : StepLambda(lambda step: 3e-4 * (1.0 - step / 2e6)),

  'actor_model' : _mlp_actor_net_orth([192, 192]),
  'critic_model' : _mlp_critic_net_orth([192, 192]),
  
  # ---- LOSS CALCULATION ----
  'entropy_factor' : (lambda step: 1e-3),
  
  'actor_regloss_factor' : 0e-4,
  'critic_regloss_factor' : 0e-4,
  
  'normalize_advantages' : True,     # minibatch advantage normalization
  'normalize_observations' : True,   # running mean + variance normalization
  'normalize_rewards' : True,        # running variance normalization
  'scale_actions' : True,
  
  'gae_gamma' : 0.999,               # reward discount factor
  'gae_lambda' : 0.95,              # smoothing for advantage, reducing variance in training  
  'gamma_env_normalization' : 0.99,
  
  
  # ---- TRAINING ----
  'epochs' : 10,
  'batchsize' : 64,
  'total_steps' : 1000000,
  'rollout' : 2048,
}