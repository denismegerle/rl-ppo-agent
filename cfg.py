
from networks import _twolayer_mlp_actor_net, _twolayer_mlp_critic_net, _twolayer_mlp_actor_net_orth, _twolayer_mlp_critic_net_orth
from ContCartpoalEnv import ContinuousCartPoleEnv
from ReachingDotEnv import ReachingDotEnv
import gym
import os, sys
from _utils import RolloutInverseTimeDecay

from tensorflow.keras.optimizers.schedules import InverseTimeDecay

#sys.path.append('../SimulationFramework/simulation/src/gym_envs//mujoco/')
#from gym_envs.mujoco.reach_env import ReachEnv

cont_ppo_test_split_cfg = {
  # ACTOR
  'actor_model' : _twolayer_mlp_actor_net_orth,
  'adam_actor_alpha' : RolloutInverseTimeDecay(3e-4, 100000, 0.0, staircase=True),             # learning rate actor
  'actor_epochs' : 3,               # 5
  'actor_batchsize' : 64,
  'actor_shuffle' : False,
  'actor_permutate' : False,
  'adam_actor_epsilon' : 1e-5,
  
  # CRITIC
  'critic_model' : _twolayer_mlp_critic_net_orth,
  'adam_critic_alpha' : RolloutInverseTimeDecay(3e-4, 100000, 0.0, staircase=True),            # learning rate critic
  'critic_epochs' : 10,
  'critic_shuffle' : False,
  'critic_batchsize' : 64,
  'adam_critic_epsilon' : 1e-5,

  'tb_log_graph' : True,
  'clip_policy_gradient_norm' : 0.5,

  # CRITIC
  'vest_clip' : 0.2,
  
  'gae_gamma' : 0.99,               # reward discount factor
  'gae_lambda' : 0.95,              # smoothing for advantage, reducing variance in training
  
  'ppo_clip' : 0.2,                 # clipping value of ppo
  'entropy_factor' : 0.001,     # entropy factor according to ppo paper
  'value_loss_factor' : 1.0,        # factor for value loss
  
  # ACTOR_TRAINING
  'normalize_advantages' : True,     # minibatch advantage normalization
  'normalize_observations' : True,   # running mean + variance normalization
  'normalize_rewards' : True,        # running variance normalization
  'scale_actions' : True,
  'clip_observations' : 10.0,
  'clip_rewards' : 10.0,
  'gamma_env_normalization' : 0.99,

  'actor_regloss_factor' : 1e-4,
  'critic_regloss_factor' : 1e-4,
  
  # ENVIRONMENT / TRAINING
  #'environment' : ReachEnv(control='ik', render=True, randomize_objects=False),
  'environment' : (lambda : ContinuousCartPoleEnv(seed=777)),
  #'environment' : gym.make('HalfCheetah-v2'),
  #'environment' : (lambda : ReachingDotEnv(seed=777)),
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