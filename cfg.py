
from networks import _simple_actor_net, _simple_critic_net, _super_simple_actor_net, _super_simple_critic_net, _twolayer_mlp_actor_net, _twolayer_mlp_critic_net, _twolayer_mlp_policy_net, _twolayer_complex_mlp_policy_net
from ContCartpoalEnv import ContinuousCartPoleEnv
from ReachingDotEnv import ReachingDotEnv
import gym
import os, sys
#from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

#sys.path.append('../SimulationFramework/simulation/src/gym_envs//mujoco/')
#from gym_envs.mujoco.reach_env import ReachEnv

"""
[summary]

"""

cont_ppo_cartpoal_cfg = {
  # ACTOR
  'actor_model' : _simple_actor_net,
  'alpha_actor' : 1e-4,             # learning rate actor
  'actor_epochs' : 8,               #
  'actor_batchsize' : 32,
  'actor_shuffle' : False,
  
  # CRITIC
  'critic_model' : _simple_critic_net,
  'alpha_critic' : 1e-4,            # learning rate critic
  'critic_epochs' : 8,
  'critic_shuffle' : True,
  'critic_batchsize' : 32,
  'qval_clip' : 0.2,
  
  'gae_gamma' : 0.99,               # reward discount factor
  'gae_lambda' : 0.95,              # smoothing for advantage, reducing variance in training
  
  'ppo_clip' : 0.2,                 # clipping value of ppo
  'ppo_entropy_factor' : 0.001,     # entropy factor according to ppo paper
  
  # ACTOR_TRAINING
  
  
  # ENVIRONMENT / TRAINING
  'environment' : ContinuousCartPoleEnv(),
  'num_stab' : 1e-10,   # value for numeric stabilization of div/log

  'rollout_episodes' : 5,
  'total_episodes' : 20000,
  'print_intervall' : 20,
  'rollout_steps' : 64
}

cont_ppo_testnormal_cfg = {
  # ACTOR
  'actor_model' : _twolayer_mlp_actor_net,
  'alpha_actor' : 1e-4,             # learning rate actor
  'actor_epochs' : 8,               #
  'actor_batchsize' : 32,
  'actor_shuffle' : False,
  
  # CRITIC
  'critic_model' : _twolayer_mlp_critic_net,
  'alpha_critic' : 1e-4,            # learning rate critic
  'critic_epochs' : 8,
  'critic_shuffle' : False,
  'critic_batchsize' : 32,
  'qval_clip' : 0.2,
  
  'gae_gamma' : 0.99,               # reward discount factor
  'gae_lambda' : 0.95,              # smoothing for advantage, reducing variance in training
  
  'ppo_clip' : 0.2,                 # clipping value of ppo
  'ppo_entropy_factor' : 0.001,     # entropy factor according to ppo paper
  
  # ACTOR_TRAINING
  
  
  # ENVIRONMENT / TRAINING
  'environment' : ContinuousCartPoleEnv(),
  #'environment' : VecNormalize(DummyVecEnv([lambda: ContinuousCartPoleEnv()]), norm_obs=True, norm_reward=True,
  #                 clip_obs=10.),
  #'environment' : ReachEnv(max_steps=500, control='ik', render=False, randomize_objects=False),
  'num_stab' : 1e-10,   # value for numeric stabilization of div/log

  'rollout_episodes' : 5,
  'total_episodes' : 5000,
  'print_interval' : 20,
}

cont_ppo_test_cartpole_cfg = {
  # ACTOR
  'actor_model' : _twolayer_mlp_policy_net,
  'alpha_actor' : 1e-4,             # learning rate actor
  'actor_epochs' : 10,               #
  'actor_batchsize' : 64,
  'actor_shuffle' : False,
  
  # CRITIC
  'alpha_critic' : 3e-4,            # learning rate critic
  'critic_epochs' : 5,
  'critic_shuffle' : False,
  'critic_batchsize' : 64,
  'vest_clip' : 0.2,
  
  'gae_gamma' : 0.99,               # reward discount factor
  'gae_lambda' : 0.95,              # smoothing for advantage, reducing variance in training
  
  'ppo_clip' : 0.2,                 # clipping value of ppo
  'ppo_entropy_factor' : 0.001,     # entropy factor according to ppo paper
  
  # ACTOR_TRAINING
  
  
  # ENVIRONMENT / TRAINING
  #'environment' : ReachEnv(max_steps=2048, control='ik', render=True, randomize_objects=False),
  'environment' : ContinuousCartPoleEnv(seed=1),
  #'environment' : VecNormalize(DummyVecEnv([lambda: ContinuousCartPoleEnv(seed=1)]), norm_obs=True, norm_reward=True,
  #                 clip_obs=10.),
  'num_stab' : 1e-10,   # value for numeric stabilization of div/log

  'rollout_episodes' : 5,
  'total_episodes' : 5000,
  'print_interval' : 20,
  'rollout_steps' : 64,
  
  'total_steps' : 1000000,
  'rollout' : 2048
}

cont_ppo_test_shared_cfg = {
  # ACTOR
  'actor_model' : _twolayer_mlp_policy_net,
  'alpha_actor' : 3e-4,             # learning rate actor
  'actor_epochs' : 10,               # 5 for other envs
  'actor_batchsize' : 64,
  'actor_shuffle' : False,
  
  # CRITIC
  'vest_clip' : 0.2,
  
  'gae_gamma' : 0.99,               # reward discount factor
  'gae_lambda' : 0.95,              # smoothing for advantage, reducing variance in training
  
  'ppo_clip' : 0.2,                 # clipping value of ppo
  'ppo_entropy_factor' : 0.000,     # entropy factor according to ppo paper
  'value_loss_factor' : 0.5,        # factor for value loss
  # ACTOR_TRAINING
  'normalize_advantages' : False,
  
  # ENVIRONMENT / TRAINING
  #'environment' : ReachEnv(control='ik', render=True, randomize_objects=False),
  'environment' : ContinuousCartPoleEnv(seed=1),
  #'environment' : gym.make('HalfCheetah-v2'),
  #'environment' : ReachingDotEnv(seed=1),
  #'environment' : VecNormalize(DummyVecEnv([lambda: ContinuousCartPoleEnv(seed=1)]), norm_obs=True, norm_reward=True,
  #                 clip_obs=10.),
  'num_stab' : 1e-10,   # value for numeric stabilization of div/log

  'rollout_episodes' : 5,
  'total_episodes' : 5000,
  'print_interval' : 20,
  'rollout_steps' : 64,
  
  'total_steps' : 1000000,
  'rollout' : 2048
}
cont_ppo_test_reachenv_cfg = {
  # ACTOR
  'actor_model' : _twolayer_complex_mlp_policy_net,
  'alpha_actor' : 3e-4,             # learning rate actor
  'actor_epochs' : 10,               #
  'actor_batchsize' : 64,
  'actor_shuffle' : False,
  
  # CRITIC
  'alpha_critic' : 3e-4,            # learning rate critic
  'critic_epochs' : 5,
  'critic_shuffle' : False,
  'critic_batchsize' : 64,
  'vest_clip' : 0.2,
  
  'gae_gamma' : 0.99,               # reward discount factor
  'gae_lambda' : 0.95,              # smoothing for advantage, reducing variance in training
  
  'ppo_clip' : 0.2,                 # clipping value of ppo
  'ppo_entropy_factor' : 0.001,     # entropy factor according to ppo paper
  
  # ACTOR_TRAINING
  
  
  # ENVIRONMENT / TRAINING
  #'environment' : ReachEnv(render=True),
  #'environment' : ContinuousCartPoleEnv(seed=1),
  #'environment' : VecNormalize(DummyVecEnv([lambda: ReachEnv(render=True)]), norm_obs=True, norm_reward=True,
  #                 clip_obs=10.),
  'num_stab' : 1e-10,   # value for numeric stabilization of div/log

  'rollout_episodes' : 5,
  'total_episodes' : 5000,
  'print_interval' : 20,
  'rollout_steps' : 64,
  
  'total_steps' : 1000000,
  'rollout' : 2048
}

cont_ppo_test_split_cfg = {
  # ACTOR
  'actor_model' : _twolayer_mlp_actor_net,
  'alpha_actor' : 3e-4,             # learning rate actor
  'actor_epochs' : 3,               # 5
  'actor_batchsize' : 64,
  'actor_shuffle' : False,
  
  # CRITIC
  'critic_model' : _twolayer_mlp_critic_net,
  'alpha_critic' : 3e-4,            # learning rate critic
  'critic_epochs' : 10,
  'critic_shuffle' : False,
  'critic_batchsize' : 64,
  
  # CRITIC
  'vest_clip' : 0.2,
  
  'gae_gamma' : 0.99,               # reward discount factor
  'gae_lambda' : 0.95,              # smoothing for advantage, reducing variance in training
  
  'ppo_clip' : 0.2,                 # clipping value of ppo
  'ppo_entropy_factor' : 0.001,     # entropy factor according to ppo paper
  'value_loss_factor' : 1.0,        # factor for value loss
  
  # ACTOR_TRAINING
  'normalize_advantages' : True,     # minibatch advantage normalization
  'normalize_observations' : True,   # running mean + variance normalization
  'normalize_rewards' : True,        # running variance normalization
  'scale_actions' : True,
  'clip_observations' : 10.0,
  'clip_rewards' : 10.0,
  'gamma_env_normalization' : 0.99,
  
  # ENVIRONMENT / TRAINING
  #'environment' : ReachEnv(control='ik', render=True, randomize_objects=False),
  'environment' : (lambda : ReachingDotEnv(seed=999)),
  #'environment' : gym.make('HalfCheetah-v2'),
  #'environment' : ReachingDotEnv(seed=1),
  #'environment' : VecNormalize(DummyVecEnv([lambda: ContinuousCartPoleEnv(seed=1)]), norm_obs=True, norm_reward=True,
  #                 clip_obs=10.),
  'num_stab_ppo' : 1e-10,   # value for numeric stabilization of div/log
  'num_stab_envnorm' : 1e-8,
  'num_stab_advnorm' : 1e-10,
  'num_stab_pdf' : 1e-10,
  
  'rollout_episodes' : 5,
  'total_episodes' : 5000,
  'print_interval' : 20,
  'rollout_steps' : 64,
  
  'total_steps' : 1000000,
  'rollout' : 2048
}

# info: for continuous rewards normalize advantages, for non continuous (i.e pos good, neg bad, sparse) do not normalize advantages