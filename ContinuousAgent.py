import datetime
import gym
import math
import operator
import os
import pprint
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import numpy as np
import pickle as pkl
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow_probability import distributions as tfd
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

# config and local imports
import _cfg
from _utils import foldl, npscanr, NormalizeWrapper, tb_log_model_graph



# -----------------------------------------------------------------------------------------------------------
class Agent(object):
  
  def __init__(self, cfg, seed=1):
    self.cfg = cfg
    self.env = NormalizeWrapper(self.cfg['environment'],
                                norm_obs=self.cfg['normalize_observations'], norm_reward=self.cfg['normalize_rewards'],
                                clip_obs=self.cfg['clip_observations'], clip_reward=self.cfg['clip_rewards'],
                                gamma=self.cfg['gamma_env_normalization'], epsilon=self.cfg['num_stab_envnorm'])
    
    self.input_dim = self.env.observation_space.shape
    self.n_actions = self.env.action_space.shape[0]
    
    self.action_space_means = (self.env.action_space.high + self.env.action_space.low) / 2.0
    self.action_space_magnitude = (self.env.action_space.high - self.env.action_space.low) / 2.0
    
    if self.cfg['model_load_path_prefix']:
      self.load_model(self.cfg['model_load_path_prefix'])
    else:
      self.actor = self._build_network(self.cfg['actor_model'], self.input_dim, self.n_actions)
      self.critic = self._build_network(self.cfg['critic_model'], self.input_dim, 1)
      self.log_std_stateless = tf.Variable(tf.zeros(self.n_actions, dtype=tf.float32), trainable=True)
    
    self.actor_optimizer = Adam(learning_rate=self.cfg['adam_actor_alpha'], epsilon=self.cfg['adam_actor_epsilon'])
    self.critic_optimizer = Adam(learning_rate=self.cfg['adam_critic_alpha'], epsilon=self.cfg['adam_critic_epsilon'])

    ## MEMORY
    self._reset_memory()
    
    ## TENSORBOARD metrics and writers
    self.start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    self.train_log_dir = f"logs/ppoagent/{self.env.get_name()}/{str(self.start_time)}"
    self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
    
    # logging losses
    self.tb_actor_loss = tf.keras.metrics.Mean('actor_losses/total_loss', dtype=tf.float32)
    self.tb_ppo_loss = tf.keras.metrics.Mean('actor_losses/ppo_loss', dtype=tf.float32)
    self.tb_entropy_loss = tf.keras.metrics.Mean('actor_losses/entropy_loss', dtype=tf.float32)
    self.tb_actor_regloss = tf.keras.metrics.Mean('actor_losses/reg_loss', dtype=tf.float32)

    self.tb_critic_loss = tf.keras.metrics.Mean('critic_losses/total_loss', dtype=tf.float32)
    self.tb_value_loss = tf.keras.metrics.Mean('critic_losses/value_loss', dtype=tf.float32)
    self.tb_critic_regloss = tf.keras.metrics.Mean('critic_losses/reg_loss', dtype=tf.float32)

    if self.cfg['tb_log_graph']:
      tb_log_model_graph(self.train_summary_writer, self.actor, self.train_log_dir, 'actor_model')
      tb_log_model_graph(self.train_summary_writer, self.critic, self.train_log_dir, 'critic_model')
    
    cfg_as_list = [ [str(k), str(v)] for k, v in self.cfg.items() ]
    
    with self.train_summary_writer.as_default():
      tf.summary.text(name='hyperparameters', data=tf.convert_to_tensor(cfg_as_list), step=0)
    

  def _reset_memory(self):
    self.state_memory, self.not_done_memory = [], []
    self.action_memory, self.action_dist_memory = [], []
    self.reward_memory, self.v_est_memory = [], []
    self.last_vest_buffer = 0.0
    
  def _build_network(self, network_model, input_dim, output_dim):
    model = network_model(input_dim, output_dim)
    model.build(input_shape=input_dim)
    return model
  
  def save_model(self, filepath):
    file_prefix = f'{filepath}/models/{self.step}'
    os.makedirs(file_prefix)
    
    tf.keras.models.save_model(self.actor, f'{file_prefix}/actor.h5', overwrite=True, include_optimizer=False, save_format='h5')
    tf.keras.models.save_model(self.actor, f'{file_prefix}/critic.h5', overwrite=True, include_optimizer=False, save_format='h5')
    np.save(f'{file_prefix}/logstd.npy', self.log_std_stateless.numpy())
  
  def load_model(self, file_prefix):
    self.actor = tf.keras.models.load_model(f'{file_prefix}/actor.h5', compile=False)
    self.critic = tf.keras.models.load_model(f'{file_prefix}/critic.h5', compile=False)
    self.log_std_stateless = tf.Variable(np.load(f'{file_prefix}/logstd.npy'), trainable=True)
    
  def _get_dist(self, means, log_stds):
    return tfd.Normal(loc=means, scale=K.exp(log_stds))

  def actor_choose(self, state):
    a_mu = self.actor(K.expand_dims(state, axis=0))
    dist = self._get_dist(a_mu[0], self.log_std_stateless)
    unscaled_action = np.clip(dist.sample(), -1.0, 1.0)

    if self.cfg['scale_actions']:
      scaled_action = self.action_space_means + unscaled_action * self.action_space_magnitude
    else: scaled_action = unscaled_action

    return scaled_action, unscaled_action, a_mu[0]
  
  def critic_evaluate(self, state):
    return self.critic(K.expand_dims(state, axis=0))[0]

  def store_transition(self, state, action, action_dist, reward, v_est, not_done):
    self.state_memory.append(state), self.not_done_memory.append(not_done)
    self.action_memory.append(action), self.action_dist_memory.append(action_dist)
    self.reward_memory.append(reward), self.v_est_memory.append(v_est)

  def _calculate_returns_and_advantages(self, v_ests, rewards, not_dones):
    vests, rews, notdones = np.asarray(v_ests + [self.last_vest_buffer]).flatten(), np.asarray(rewards).flatten(), np.asarray(not_dones).flatten()
    
    # calculate actual returns (discounted rewards) based on observation
    def discounted_return_fn(accumulated_discounted_reward, reward_discount):
      reward, discount = reward_discount
      return accumulated_discounted_reward * discount + reward
    
    discounts = self.cfg['gae_gamma'] * notdones
    returns = npscanr(discounted_return_fn, self.last_vest_buffer, list(zip(rews, discounts)))

    # calculate actual advantages based on td residual (see gae paper, eq. 16)
    def weighted_cumulative_td_fn(accumulated_td, weights_td_tuple):
      td, weighted_discount = weights_td_tuple
      return accumulated_td * weighted_discount + td 
    
    deltas = rews + discounts * vests[1:] - vests[:-1]
    advantages = npscanr(weighted_cumulative_td_fn, 0, list(zip(deltas, discounts * self.cfg['gae_lambda'])))
    
    return returns, advantages
  
  def _ppo_clip_loss(self, log_pi_new, log_pi_old, advantage):
    ratio = K.exp(log_pi_new - log_pi_old)
    clip_ratio = K.clip(ratio, min_value=1 - self.cfg['ppo_clip'](self.step), max_value=1 + self.cfg['ppo_clip'](self.step))

    surrogate1 = ratio * advantage
    surrogate2 = clip_ratio * advantage
    
    return - K.mean(K.minimum(surrogate1, surrogate2))
  
  def _value_loss(self, values, values_old, returns):
    clipped_vest = K.clip(values, min_value=values_old - self.cfg['vest_clip'](self.step), max_value=values_old + self.cfg['vest_clip'](self.step))

    surrogate1 = K.square(values - returns)
    surrogate2 = K.square(clipped_vest - returns)

    return K.mean(K.minimum(surrogate1, surrogate2))
  
  def _entropy_loss(self, mu, log_std):
    return - K.mean(self._get_dist(mu, log_std).entropy())

  def _reg_loss(self, model):
    if model.losses:
      return tf.math.add_n(self.actor.losses)
    else : return 0.0
  
  def _train(self, returns, advantages, actions, v_ests):
    y_true_actions, y_pred_vest_old, y_true_returns = actions, v_ests, returns
    old_log_std = tf.Variable(self.log_std_stateless.value(), dtype=tf.float32)

    sample_amt = len(self.action_memory)
    sample_range, batches_amt = np.arange(sample_amt), sample_amt // self.cfg['batchsize']
    
    if self.cfg['permutate']:
      np.random.shuffle(sample_range)

    for _ in range(self.cfg['epochs']):
      for i in range(batches_amt):
        if self.cfg['shuffle']:
          np.random.shuffle(sample_range)
          sample_idx = sample_range[:self.cfg['batchsize']]
        else:
          sample_idx = sample_range[i * self.cfg['batchsize']:(i + 1) * self.cfg['batchsize']]
        
        batch_states = np.asarray([self.state_memory[i] for i in sample_idx])
        batch_action_dist = np.asarray([self.action_dist_memory[i] for i in sample_idx])
        
        batch_y_true_actions = np.asarray([y_true_actions[i] for i in sample_idx])
        batch_y_true_returns = np.asarray([y_true_returns[i] for i in sample_idx])
        batch_advantage = np.asarray([advantages[i] for i in sample_idx])
        batch_y_pred_vest_old = np.asarray([y_pred_vest_old[i] for i in sample_idx])
        
        if self.cfg['normalize_advantages']:
          batch_advantage = (batch_advantage - batch_advantage.mean()) / np.maximum(batch_advantage.std(), self.cfg['num_stab_advnorm'])
        
        with tf.GradientTape(persistent=True) as tape:
          batch_y_pred_mu = self.actor(batch_states)
          batch_y_pred_vest = self.critic(batch_states)

          # in case of multiple actions p(a_0, ..., a_N) = p(a_0) * ... * p(a_N)
          log_pi_new = K.sum(self._get_dist(batch_y_pred_mu, self.log_std_stateless).log_prob(batch_y_true_actions), axis=-1)
          log_pi_old = K.sum(self._get_dist(batch_action_dist, old_log_std).log_prob(batch_y_true_actions), axis=-1)
          
          # loss calculation
          ppo_clip_loss = self._ppo_clip_loss(log_pi_new=log_pi_new, log_pi_old=log_pi_old, advantage=batch_advantage)
          entropy_loss = self.cfg['entropy_factor'](self.step) * self._entropy_loss(batch_y_pred_mu, self.log_std_stateless)
          reg_loss_actor = self.cfg['actor_regloss_factor'] * self._reg_loss(self.actor)
          actor_loss = ppo_clip_loss + entropy_loss + reg_loss_actor
          
          value_loss = self.cfg['value_loss_factor'] * self._value_loss(batch_y_pred_vest, batch_y_pred_vest_old, batch_y_true_returns)
          reg_loss_critic = self.cfg['critic_regloss_factor'] * self._reg_loss(self.critic)
          critic_loss = value_loss + reg_loss_critic
          
          # tensorboard logging
          self.tb_actor_loss(actor_loss)
          self.tb_ppo_loss(ppo_clip_loss)
          self.tb_entropy_loss(entropy_loss)
          self.tb_actor_regloss(reg_loss_actor)

          self.tb_critic_loss(critic_loss)
          self.tb_value_loss(value_loss)
          self.tb_critic_regloss(reg_loss_critic)
        
        gradient = tape.gradient(actor_loss, [self.log_std_stateless])
        self.actor_optimizer.apply_gradients(zip(gradient, [self.log_std_stateless]))

        gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        gradient, _ = tf.clip_by_global_norm(gradient, clip_norm=self.cfg['clip_policy_gradient_norm'])
        self.actor_optimizer.apply_gradients(zip(gradient, self.actor.trainable_variables))
        
        gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradient, self.critic.trainable_variables))
  
  def train(self):    
    # calculate returns and advantages
    self.returns, self.advantages = self._calculate_returns_and_advantages(self.v_est_memory, self.reward_memory, self.not_done_memory)
    
    # train agent
    self._train(self.returns, self.advantages, self.action_memory, self.v_est_memory)
    self._log_training()
    self._reset_memory()

  def _log_training(self):
    with self.train_summary_writer.as_default():
      # log losses
      tf.summary.scalar('actor_losses/total_loss', self.tb_actor_loss.result(), step=self.step)
      tf.summary.scalar('actor_losses/ppo_loss', self.tb_ppo_loss.result(), step=self.step)
      tf.summary.scalar('actor_losses/entropy_loss', self.tb_entropy_loss.result(), step=self.step)
      tf.summary.scalar('actor_losses/reg_loss', self.tb_actor_regloss.result(), step=self.step)

      tf.summary.scalar('critic_losses/total_loss', self.tb_critic_loss.result(), step=self.step)
      tf.summary.scalar('critic_losses/value_loss', self.tb_value_loss.result(), step=self.step)
      tf.summary.scalar('critic_losses/reg_loss', self.tb_critic_regloss.result(), step=self.step)

      # log returns and advantages
      tf.summary.scalar('env_metrics/avg_returns_per_step', np.average(self.returns), step=self.step)
      tf.summary.scalar('env_metrics/avg_advantages_per_step', np.average(self.advantages), step=self.step)
      tf.summary.histogram('env_metrics/returns_per_step', self.returns, step=self.step)
      tf.summary.histogram('env_metrics/advantages_per_step', self.advantages, step=self.step)
      
      # log optimizer statistisc
      tf.summary.scalar('optimizer/actor_lr', self.actor_optimizer._decayed_lr(tf.float32), step=self.step)
      tf.summary.scalar('optimizer/critic_lr', self.critic_optimizer._decayed_lr(tf.float32), step=self.step)
          
    self.tb_actor_loss.reset_states()
    self.tb_ppo_loss.reset_states()
    self.tb_entropy_loss.reset_states()
    self.tb_actor_regloss.reset_states()

    self.tb_critic_loss.reset_states()
    self.tb_value_loss.reset_states()
    self.tb_critic_regloss.reset_states()

  def _log_episode(self, observations, actions, scores, episode, step):
    epscore = foldl(operator.add, scores)
    with self.train_summary_writer.as_default():
      tf.summary.scalar('env_metrics/episode_score_per_step', epscore, step=step)
      tf.summary.scalar('env_metrics/episode_score_per_episode', epscore, step=episode)
      tf.summary.histogram('env_metrics/rewards_per_episode', scores, step=episode)
      
      # observations logging
      obs = np.asarray(observations)
      for i in range(obs.shape[1]):
        tf.summary.histogram(f'env_metrics_obs/observation_{i}_per_episode', obs[:, i], step=episode)
          
      # action logging
      acts = np.asarray(actions)
      for i in range(acts.shape[1]):
        tf.summary.histogram(f'env_metrics_acts/action_{i}_per_episode', acts[:, i], step=episode)
      
      # std logging
      for i in range(self.log_std_stateless.shape[0]):
        tf.summary.scalar(f'env_metrics_acts/std_action_{i}_per_episode', np.exp(self.log_std_stateless[i]), step=step)

  def learn(self):
    s, episode, done = self.env.reset(), 0, False
    observations, actions, scores = [], [], []

    for self.step in tqdm(range(self.cfg['total_steps'])):
      # choose and take an action, advance environment and store data
      #self.env.render()
      observations.append(self.env.unnormalize_obs(s))

      scaled_a, unscaled_a, a_dist = self.actor_choose(s)
      actions.append(unscaled_a)

      s_, r, done, _ = self.env.step(scaled_a)
      scores.append(self.env.unnormalize_reward(r))

      v_est = self.critic_evaluate(s)
      
      self.store_transition(s, unscaled_a, a_dist, r, v_est, not done)
      s = s_
      
      # resetting environment if instance is terminated
      if done:
        self._log_episode(observations, actions, scores, episode, self.step)
        s, scores, observations, actions, done = self.env.reset(), [], [], [], False
        episode += 1

      if self.step % self.cfg['model_save_interval'] == 0:
        self.save_model(self.train_log_dir)
        
      if self.step % self.cfg['rollout'] == 0 and self.step > 0:
        self.cfg['adam_actor_alpha'].update_rollout_step(self.step)
        self.cfg['adam_critic_alpha'].update_rollout_step(self.step)
        
        self.last_vest_buffer = self.critic_evaluate(s_)
        self.train()
# -----------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
  tf.random.set_seed(1)
  np.random.seed(1)
  
  agt_cfg = _cfg.cont_ppo_test_split_cfg
  Agent(cfg=agt_cfg).learn()