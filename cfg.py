
from networks import _simple_actor_net, _simple_critic_net, _super_simple_actor_net, _super_simple_critic_net
from ContCartpoalEnv import ContinuousCartPoleEnv
from ReachingDotEnv import ReachingDotEnv
import gym


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

# cont_ppo_reachdot_cfg = {
#   # ACTOR
#   'actor_model' : _super_simple_actor_net,
#   'alpha_actor' : 1e-4,             # learning rate actor
#   'actor_epochs' : 8,               #
#   'actor_batchsize' : 32,
#   'actor_shuffle' : True,
  
#   # CRITIC
#   'critic_model' : _super_simple_critic_net,
#   'alpha_critic' : 1e-4,            # learning rate critic
#   'critic_epochs' : 8,
#   'critic_shuffle' : True,

#   'gae_gamma' : 0.99,               # reward discount factor
#   'gae_lambda' : 0.95,              # smoothing for advantage, reducing variance in training
  
#   'ppo_clip' : 0.2,                 # clipping value of ppo
#   'ppo_entropy_factor' : 0.001,     # entropy factor according to ppo paper
  
#   # ACTOR_TRAINING
  
  
#   # ENVIRONMENT / TRAINING
#   'environment' : ReachingDotEnv(),
#   'num_stab' : 1e-10,   # value for numeric stabilization of div/log

#   'rollout_episodes' : 6,
#   'total_episodes' : 20000,
#   'print_intervall' : 50,
#   'rollout_steps' : 64
# }