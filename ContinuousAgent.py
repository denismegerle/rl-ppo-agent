import numpy as np
import gym
import operator
import tensorflow as tf 
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule, ExponentialDecay, InverseTimeDecay
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from functools import reduce
import tensorflow.keras.backend as K
import math, random, cfg
from scipy.signal import lfilter
import pickle as pkl
from scipy import io as scio
from itertools import accumulate
from _utils import npscanr
import datetime

class Agent(object):
  
  def __init__(self, cfg):
    self.cfg = cfg
    self.env = self.cfg['environment']
    
    self.input_dim = self.env.observation_space.shape
    self.n_actions = self.env.action_space.shape[0]
    
    self.actor_optimizer = Adam(self.cfg['alpha_actor'])
    self.actor = self._build_policy_network(self.input_dim, self.n_actions)

    self.critic_optimizer = Adam(self.cfg['alpha_critic'])
    self.critic = self._build_critic_network(self.input_dim, 1)
    
    ## MEMORY
    self._reset_memory()
    
    ## TENSORBOARD metrics and writers
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = f"logs/ppoagent/{type(self.env).__name__}/{str(current_time)}"
    self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    
    # - total loss, value loss, ppo clip loss, clip ratio, entropy loss
    self.tb_total_loss = tf.keras.metrics.Mean('total_loss', dtype=tf.float32)
    self.tb_ppo_loss = tf.keras.metrics.Mean('ppo_loss', dtype=tf.float32)
    self.tb_value_loss = tf.keras.metrics.Mean('value_loss', dtype=tf.float32)
    self.tb_entropy_loss = tf.keras.metrics.Mean('entropy_loss', dtype=tf.float32)
    # - average advantage/reward/return
    
    # - episode score
    self.tb_episode_score = tf.keras.metrics.Mean('episode_score', dtype=tf.float32)
    

  def _reset_memory(self):
    self.state_memory, self.not_done_memory = [], []
    self.action_memory, self.action_dist_memory = [], []
    self.reward_memory, self.v_est_memory = [], []
    self.last_vest_buffer = 0.0
    
  def _build_policy_network(self, input_dim, output_dim):
    model = self.cfg['actor_model'](input_dim, output_dim)
    model.build(input_shape=input_dim)
    return model
  
  def _build_critic_network(self, input_dim, output_dim):
    model = self.cfg['critic_model'](input_dim, output_dim)
    model.build(input_shape=input_dim)
    return model
  
  def actor_choose(self, state):
    a_mu, a_sig = self.actor(K.expand_dims(state, axis=0))
    action = np.random.normal(loc=a_mu[0], scale=a_sig[0], size=None)
    action = np.clip(action, -1.0, 1.0)
    action = self.cfg['environment'].action_space.low + ((action + 1.0) / 2.0) * (self.cfg['environment'].action_space.high - self.cfg['environment'].action_space.low)
    return action, [a_mu[0], a_sig[0]]
  
  def critic_evaluate(self, state):
    v_est = self.critic(K.expand_dims(state, axis=0))
    return v_est[0]

  def store_transition(self, state, action, action_dist, reward, v_est, not_done):
    self.state_memory.append(state)
    self.action_memory.append(action)
    self.action_dist_memory.append(action_dist)
    self.reward_memory.append(reward)
    self.v_est_memory.append(v_est)
    self.not_done_memory.append(not_done)

  def _calculate_gae(self, v_ests, rewards, not_dones):
    vests, rews, notdones = np.asarray(v_ests + [self.last_vest_buffer]).flatten(), np.asarray(rewards).flatten(), np.asarray(not_dones).flatten()
    
    # calculate actual returns (discounted rewards) based on observation
    def discounted_return_fn(accumulated_discounted_reward, reward_discount):
      reward, discount = reward_discount
      return accumulated_discounted_reward * discount + reward
    
    discounts = self.cfg['gae_gamma'] * notdones
    returns = npscanr(discounted_return_fn, self.last_vest_buffer, list(zip(rews, discounts)))

    # calculate actual advantages based on td residual (see gae paper)  -> TODO check whether this is correct
    def weighted_cumulative_td_fn(accumulated_td, weights_td_tuple):
      weighted_discount, td = weights_td_tuple
      return td + weighted_discount * accumulated_td
    
    deltas = rews + discounts * vests[1:] - vests[:-1]
    advantages = npscanr(weighted_cumulative_td_fn, 0, list(zip(discounts * self.cfg['gae_lambda'], deltas)))
    
    if self.cfg['normalize_advantages']:
      advantages = (advantages - advantages.mean()) / np.maximum(advantages.std(), self.cfg['num_stab'])
    
    return returns, advantages
    
  def _norm_pdf(self, x, mu, sig):
    var = tf.cast(K.square(sig), tf.float32)
    denom = tf.cast(K.sqrt(2 * math.pi * var), tf.float32)
    deviation = tf.cast(K.square(x - mu), tf.float32)
    expon = - (1 / 2) * deviation / (var + self.cfg['num_stab'])
    return (1 / denom) * K.exp(expon)
  
  def _ppo_clip_loss(self, pi_new, pi_old, advantage):
    ratio = pi_new / (pi_old + self.cfg['num_stab'])
    clip_ratio = K.clip(ratio, min_value=1 - self.cfg['ppo_clip'], max_value=1 + self.cfg['ppo_clip'])

    surrogate1 = ratio * advantage
    surrogate2 = clip_ratio * advantage
    
    return - K.mean(K.minimum(surrogate1, surrogate2))
  
  def _value_loss(self, values, values_old, returns):
    clipped_vest = K.clip(values, min_value=values_old - self.cfg['vest_clip'], max_value=values_old + self.cfg['vest_clip'])

    surrogate1 = K.square(values - returns)
    surrogate2 = K.square(clipped_vest - returns)

    return K.mean(K.minimum(surrogate1, surrogate2))
    
  def _entropy_norm_pdf(self, sig):
    entropy_loss = K.log(K.sqrt(2 * math.pi * math.e * K.square(sig)) + self.cfg['num_stab'])
    return - K.mean(entropy_loss)
  
  def _train(self, returns, advantages):
    y_true_actions = self.action_memory
    y_pred_vest_old = self.v_est_memory
    y_true_returns = returns
    
    sample_amt = len(self.action_memory)
    sample_range, batches_amt = np.arange(sample_amt), sample_amt // self.cfg['actor_batchsize']
    
    for _ in range(self.cfg['actor_epochs']):
      for i in range(batches_amt):
        if self.cfg['actor_shuffle']:
          np.random.shuffle(sample_range)
          sample_idx = sample_range[:self.cfg['actor_batchsize']]
        else:
          sample_idx = sample_range[i * self.cfg['actor_batchsize']:(i + 1) * self.cfg['actor_batchsize']]
        
        batch_states = np.asarray([self.state_memory[i] for i in sample_idx])
        batch_action_dist = np.asarray([self.action_dist_memory[i] for i in sample_idx])
        
        batch_y_true_actions = np.asarray([y_true_actions[i] for i in sample_idx])
        batch_y_true_returns = np.asarray([y_true_returns[i] for i in sample_idx])
        batch_advantage = np.asarray([advantages[i] for i in sample_idx])
        batch_y_pred_vest_old = np.asarray([y_pred_vest_old[i] for i in sample_idx])
        
        batch_action_mu = [x[0] for x in batch_action_dist]
        batch_action_sig = [x[1] for x in batch_action_dist]
        
        with tf.GradientTape(persistent=True) as tape:
          batch_y_pred_mu, batch_y_pred_sig = self.actor(batch_states)
          batch_y_pred_vest = self.critic(batch_states)
          
          # in case of multiple actions p(a_0, ..., a_N) = p(a_0) * ... * p(a_N)
          pi_new = K.prod(self._norm_pdf(batch_y_true_actions, batch_y_pred_mu, batch_y_pred_sig), axis=1)
          pi_old = K.prod(self._norm_pdf(batch_y_true_actions, batch_action_mu, batch_action_sig), axis=1)
          
          # loss calculation
          ppo_clip_loss = self._ppo_clip_loss(pi_new=pi_new, pi_old=pi_old, advantage=batch_advantage)
          entropy_loss = self.cfg['ppo_entropy_factor'] * self._entropy_norm_pdf(batch_action_sig)
          value_loss = self.cfg['value_loss_factor'] * self._value_loss(batch_y_pred_vest, batch_y_pred_vest_old, batch_y_true_returns)
          
          actor_loss = ppo_clip_loss + entropy_loss + value_loss
          critic_loss = value_loss
          
          # tensorboard logging
          self.tb_total_loss(actor_loss)
          self.tb_ppo_loss(ppo_clip_loss)
          self.tb_value_loss(value_loss)
          self.tb_entropy_loss(entropy_loss)
        
        gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        gradient, _ = tf.clip_by_global_norm(gradient, clip_norm=0.5)
        self.actor_optimizer.apply_gradients(zip(gradient, self.actor.trainable_variables))
        
        gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradient, self.critic.trainable_variables))
        
  def train(self):
    # calculate returns and advantages
    if self.cfg['clip_rewards']:
      rewards = np.clip(self.reward_memory, *self.cfg['clip_rewards'])
    else: rewards = self.reward_memory
    
    returns, advantages = self._calculate_gae(self.v_est_memory, rewards, self.not_done_memory)
    
    # train agent
    self._train(returns, advantages)
    self._reset_memory()

  def learn(self):
    scores, episode = [], 0
    s, ep_score, done = self.env.reset(), 0, True
    
    for step in range(self.cfg['total_steps']):
      # choose and take an action, advance environment and store data
      # self.env.render()
      a, a_dist = self.actor_choose(s)
      s_, r, done, _ = self.env.step(a)
      ep_score += r
      v_est = self.critic_evaluate(s)
      self.store_transition(s, a, a_dist, r, v_est, not done)
      s = s_
      
      # resetting environment if instance is terminated
      if done:
        scores.append(ep_score)
        
        self.tb_episode_score(ep_score)
        with self.train_summary_writer.as_default():
          tf.summary.scalar('episode_score', self.tb_episode_score.result(), step=step)
        self.tb_episode_score.reset_states()
        
        print(f'Episode {episode}, Score {ep_score}')
        
        if episode % self.cfg['print_interval'] == 0:
          print(f"Mean - Episode {episode}, Score {np.mean(scores[-self.cfg['print_interval']:])}")
        
        s, ep_score, done = self.env.reset(), 0, False
        episode += 1

      if step % self.cfg['rollout'] == 0 and step > 0:
        self.last_vest_buffer = self.critic_evaluate(s_)
        agent.train()
        
        # TENSORBOARD LOG
        with self.train_summary_writer.as_default():
          tf.summary.scalar('total_loss', self.tb_total_loss.result(), step=step)
          tf.summary.scalar('ppo_loss', self.tb_ppo_loss.result(), step=step)
          tf.summary.scalar('value_loss', self.tb_value_loss.result(), step=step)
          tf.summary.scalar('entropy_loss', self.tb_entropy_loss.result(), step=step)

        vest_loss = self.tb_value_loss.result()
        
        self.tb_total_loss.reset_states()
        self.tb_ppo_loss.reset_states()
        self.tb_value_loss.reset_states()
        self.tb_entropy_loss.reset_states()
        
        print(f'train agent (v_est_loss) at step {step} = {vest_loss}')
""" ******************************************************************************** """

if __name__ == "__main__":
  tf.random.set_seed(1)
  np.random.seed(1)
  
  agt_cfg = cfg.cont_ppo_test_split_cfg
  agent = Agent(cfg=agt_cfg)

  agent.learn()