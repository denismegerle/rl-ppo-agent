
import gym
import os, sys

# config and local imports
from envs.ContCartpoalEnv import ContinuousCartPoleEnv
from envs.ReachingDotEnv import ReachingDotEnv
from _nets import _twolayer_mlp_actor_net_orth, _twolayer_mlp_critic_net_orth
from _utils import RolloutInverseTimeDecay, StepLambda

IMPORT_SIM_FRAMEWORK = False
if IMPORT_SIM_FRAMEWORK:
  sys.path.append('../SimulationFramework/simulation/src/gym_envs//mujoco/')
  from gym_envs.mujoco.reach_env import ReachEnv






base_cfg = {
  # ---- NUMERICAL ----
  'num_stab_envnorm' : 1e-8,      # numerical stab. normalization wrapper environment
  'num_stab_advnorm' : 1e-10,     # numerical stab. batch-wise advantage normalization
}

mujoco_base_cfg = {}
atari_base_cfg = {}


cont_ppo_test_split_cfg = {
  **base_cfg,
  
  # ---- NET/TF CONFIG ----
  'actor_model' : _twolayer_mlp_actor_net_orth,
  'adam_actor_alpha' : StepLambda(lambda step: 3e-4 * (1.0 - (step / 3e6))),
  'adam_actor_epsilon' : 1e-5,

  'critic_model' : _twolayer_mlp_critic_net_orth,
  'adam_critic_alpha' : StepLambda(lambda step: 3e-4 * (1.0 - (step / 3e6))),
  'adam_critic_epsilon' : 1e-5,
  
  'model_save_interval' : 10000,
  'model_load_path_prefix' : None,
  'tb_log_graph' : True,
  
  # ---- LOSS CALCULATION ----
  'ppo_clip' : (lambda step: 0.2),
  'entropy_factor' : (lambda step: 0e-2 * (1.0 - (step / 3e6))),
  
  'value_loss_factor' : 1.0,
  'vest_clip' : (lambda step: 0.2),
  
  'actor_regloss_factor' : 0e-4,
  'critic_regloss_factor' : 0e-4,
  
  'clip_policy_gradient_norm' : 0.5,
  
  # ---- TRAINING ----
  'epochs' : 10,
  'batchsize' : 32,
  'shuffle' : False,
  'permutate' : True,
  'total_steps' : 3000000,
  'rollout' : 2048,
  
  'gae_gamma' : 0.99,               # reward discount factor
  'gae_lambda' : 0.95,              # smoothing for advantage, reducing variance in training
  
  # ---- ENVIRONMENT ----
  'environment' : (lambda : gym.make('HalfCheetah-v2')),
  'normalize_advantages' : True,     # minibatch advantage normalization
  'normalize_observations' : True,   # running mean + variance normalization
  'normalize_rewards' : True,        # running variance normalization
  'scale_actions' : True,

  'clip_observations' : 10.0,
  'clip_rewards' : 10.0,
  
  'gamma_env_normalization' : 0.99,
}


# cont_ppo_test_split_cfg = {
#   # ---- NET/TF CONFIG ----
#   'actor_model' : _twolayer_mlp_actor_net_orth,
#   'adam_actor_alpha' : RolloutInverseTimeDecay(3e-4, 100000, 1.0, staircase=False),
#   'adam_actor_epsilon' : 1e-5,

#   'critic_model' : _twolayer_mlp_critic_net_orth,
#   'adam_critic_alpha' : RolloutInverseTimeDecay(3e-4, 100000, 1.0, staircase=False),
#   'adam_critic_epsilon' : 1e-5,
  
#   'model_save_interval' : 1000,
#   'model_load_path_prefix' : None,
#   'tb_log_graph' : True,
  
#   # ---- LOSS CALCULATION ----
#   'ppo_clip' : 0.2,
#   'entropy_factor' : 0.001,
  
#   'value_loss_factor' : 1.0,
#   'vest_clip' : 0.2,
  
#   'actor_regloss_factor' : 1e-4,
#   'critic_regloss_factor' : 1e-4,
  
#   'clip_policy_gradient_norm' : 0.5,
  
#   # ---- TRAINING ----
#   'epochs' : 3,
#   'batchsize' : 64,
#   'shuffle' : False,
#   'permutate' : False,
#   'total_steps' : 1000000,
#   'rollout' : 2048,
  
#   'gae_gamma' : 0.99,               # reward discount factor
#   'gae_lambda' : 0.95,              # smoothing for advantage, reducing variance in training
  
#   # ---- ENVIRONMENT ----
#   'environment' : (lambda : ContinuousCartPoleEnv(seed=1)),
#   'normalize_advantages' : True,     # minibatch advantage normalization
#   'normalize_observations' : True,   # running mean + variance normalization
#   'normalize_rewards' : True,        # running variance normalization
#   'scale_actions' : True,

#   'clip_observations' : 10.0,
#   'clip_rewards' : 10.0,
  
#   'gamma_env_normalization' : 0.99,
  
#   # ---- NUMERICAL ----
#   'num_stab_ppo' : 1e-10,   # value for numeric stabilization of div/log
#   'num_stab_envnorm' : 1e-8,
#   'num_stab_advnorm' : 1e-10,
#   'num_stab_pdf' : 1e-10,
# }
  # ENVIRONMENT / TRAINING
  #'environment' : ReachEnv(control='ik', render=True, randomize_objects=False),
  
  #'environment' : (lambda : gym.make('HalfCheetah-v2')),
  #'environment' : (lambda : ReachEnv(max_steps=25, render=True, randomize_objects=True)),
  #'environment' : (lambda : ReachingDotEnv(seed=777)),
  #'environment' : VecNormalize(DummyVecEnv([lambda: ContinuousCartPoleEnv(seed=1)]), norm_obs=True, norm_reward=True,
  #                 clip_obs=10.),
  
  
# info: for continuous rewards normalize advantages, for non continuous (i.e pos good, neg bad, sparse) do not normalize advantages


# cont_ppo_test_split_cfg = {
#   # ACTOR
#   'actor_model' : _twolayer_mlp_actor_net_orth,
#   'adam_actor_alpha' : RolloutInverseTimeDecay(3e-4, 100000, 1.0, staircase=False),             # learning rate actor
#   'actor_epochs' : 10,               # 5
#   'actor_batchsize' : 32,
#   'actor_shuffle' : False,
#   'actor_permutate' : True,
#   'adam_actor_epsilon' : 1e-5,
  
#   # CRITIC
#   'critic_model' : _twolayer_mlp_critic_net_orth,
#   'adam_critic_alpha' : RolloutInverseTimeDecay(3e-4, 100000, 1.0, staircase=False),            # learning rate critic
#   'critic_epochs' : 10,
#   'critic_shuffle' : False,
#   'critic_batchsize' : 64,
#   'adam_critic_epsilon' : 1e-5,

#   'tb_log_graph' : True,
#   'clip_policy_gradient_norm' : 0.5,

#   # CRITIC
#   'vest_clip' : 0.2,
  
#   'gae_gamma' : 0.99,               # reward discount factor
#   'gae_lambda' : 0.95,              # smoothing for advantage, reducing variance in training
  
#   'ppo_clip' : 0.2,                 # clipping value of ppo
#   'entropy_factor' : 0.0,           # entropy factor according to ppo paper
#   'value_loss_factor' : 1.0,        # factor for value loss
  
#   # ACTOR_TRAINING
#   'normalize_advantages' : True,     # minibatch advantage normalization
#   'normalize_observations' : True,   # running mean + variance normalization
#   'normalize_rewards' : True,        # running variance normalization
#   'scale_actions' : True,
#   'clip_observations' : 10.0,
#   'clip_rewards' : 10.0,
#   'gamma_env_normalization' : 0.99,

#   'actor_regloss_factor' : 0e-4,
#   'critic_regloss_factor' : 0e-4,
  
#   # ENVIRONMENT / TRAINING
#   #'environment' : ReachEnv(control='ik', render=True, randomize_objects=False),
#   #'environment' : (lambda : ContinuousCartPoleEnv(seed=777)),
#   'environment' : (lambda : gym.make('HalfCheetah-v2')),
#   #'environment' : (lambda : ReachingDotEnv(seed=777)),
#   #'environment' : VecNormalize(DummyVecEnv([lambda: ContinuousCartPoleEnv(seed=1)]), norm_obs=True, norm_reward=True,
#   #                 clip_obs=10.),
#   'num_stab_ppo' : 1e-10,   # value for numeric stabilization of div/log
#   'num_stab_envnorm' : 1e-8,
#   'num_stab_advnorm' : 1e-10,
#   'num_stab_pdf' : 1e-10,
  
#   'rollout_episodes' : 5,
#   'total_episodes' : 5000,
#   'print_interval' : 20,
#   'rollout_steps' : 64,
  
#   'total_steps' : 1000000,
#   'rollout' : 2048
# }

# cont_ppo_test_split_cfg = {
#   # ACTOR
#   'actor_model' : _twolayer_mlp_actor_net_orth,
#   'adam_actor_alpha' : RolloutInverseTimeDecay(3e-4, 100000, 1.0, staircase=False),             # learning rate actor
#   'actor_epochs' : 10,               # 5
#   'actor_batchsize' : 32,
#   'actor_shuffle' : False,
#   'actor_permutate' : True,
#   'adam_actor_epsilon' : 1e-5,
  
#   # CRITIC
#   'critic_model' : _twolayer_mlp_critic_net_orth,
#   'adam_critic_alpha' : RolloutInverseTimeDecay(3e-4, 100000, 1.0, staircase=False),            # learning rate critic
#   'critic_epochs' : 10,
#   'critic_shuffle' : False,
#   'critic_batchsize' : 64,
#   'adam_critic_epsilon' : 1e-5,

#   'tb_log_graph' : True,
#   'clip_policy_gradient_norm' : 0.5,

#   # CRITIC
#   'vest_clip' : 0.2,
  
#   'gae_gamma' : 0.99,               # reward discount factor
#   'gae_lambda' : 0.95,              # smoothing for advantage, reducing variance in training
  
#   'ppo_clip' : 0.2,                 # clipping value of ppo
#   'entropy_factor' : 0.0,           # entropy factor according to ppo paper
#   'value_loss_factor' : 1.0,        # factor for value loss
  
#   # ACTOR_TRAINING
#   'normalize_advantages' : True,     # minibatch advantage normalization
#   'normalize_observations' : True,   # running mean + variance normalization
#   'normalize_rewards' : True,        # running variance normalization
#   'scale_actions' : True,
#   'clip_observations' : 10.0,
#   'clip_rewards' : 10.0,
#   'gamma_env_normalization' : 0.99,

#   'actor_regloss_factor' : 0e-4,
#   'critic_regloss_factor' : 0e-4,
  
#   # ENVIRONMENT / TRAINING
#   #'environment' : ReachEnv(control='ik', render=True, randomize_objects=False),
#   #'environment' : (lambda : ContinuousCartPoleEnv(seed=777)),
#   #'environment' : (lambda : gym.make('HalfCheetah-v2')),
#   'environment' : (lambda : ReachEnv(max_steps=25, render=True, randomize_objects=True)),
#   #'environment' : (lambda : ReachingDotEnv(seed=777)),
#   #'environment' : VecNormalize(DummyVecEnv([lambda: ContinuousCartPoleEnv(seed=1)]), norm_obs=True, norm_reward=True,
#   #                 clip_obs=10.),
#   'num_stab_ppo' : 1e-10,   # value for numeric stabilization of div/log
#   'num_stab_envnorm' : 1e-8,
#   'num_stab_advnorm' : 1e-10,
#   'num_stab_pdf' : 1e-10,
  
#   'rollout_episodes' : 5,
#   'total_episodes' : 5000,
#   'print_interval' : 20,
#   'rollout_steps' : 64,
  
#   'total_steps' : 1000000,
#   'rollout' : 2048
# }