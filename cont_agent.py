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


"""

TODO
* save model, checkpoints, ...
* cb tensorflow tensorboard

"""


class Agent(object):
  
  def __init__(self, cfg):
    self.cfg = cfg
    self.env = self.cfg['environment']
    
    self.input_dim = self.env.observation_space.shape
    self.n_actions = self.env.action_space.shape[0]
    
    self.actor_optimizer = Adam(self.cfg['alpha_actor'])
    self.actor = self._build_policy_network(self.input_dim, self.n_actions)

    self.critic_optimizer = Adam(self.cfg['alpha_critic'])
    self.critic = self._build_critic_network(self.input_dim)

    # memory
    self._reset_memory()

  def _reset_memory(self):
    self.state_memory, self.not_done_memory = [], []
    self.action_memory, self.action_dist_memory = [], []
    self.reward_memory, self.q_value_memory = [], []
    
  def _build_policy_network(self, input_dim, output_dim):
    model = self.cfg['actor_model'](input_dim, output_dim)
    model.build(input_shape=input_dim)
    return model

  def _build_critic_network(self, input_dim):
    model = self.cfg['critic_model'](input_dim)
    model.build(input_shape=input_dim)
    return model
  
  def actor_choose(self, state):
    a_mu, a_sig = self.actor(K.expand_dims(state, axis=0))
    action = np.random.normal(loc=a_mu[0], scale=a_sig[0], size=None)
    action = np.clip(action, -1.0, 1.0)
    return action, [a_mu[0], a_sig[0]]

  def critic_evaluate(self, state):
    return self.critic(K.expand_dims(state, axis=0))[0]
  
  def store_transition(self, state, action, action_dist, reward, q_value, not_done):
    self.state_memory.append(state)
    self.action_memory.append(action)
    self.action_dist_memory.append(action_dist)
    self.reward_memory.append(reward)
    self.q_value_memory.append(q_value)
    self.not_done_memory.append(not_done)

  def _calculate_gae(self, q_vals, rews, not_dones):
    ret, gae = [], 0
    
    q_vals.append(0.0)
    for i in reversed(range(len(rews))):
      delta = rews[i] + self.cfg['gae_gamma'] * q_vals[i + 1] * not_dones[i] - q_vals[i]
      gae = delta + self.cfg['gae_gamma'] * self.cfg['gae_lambda'] * gae * not_dones[i]
      ret.append(gae + q_vals[i])
    q_vals = q_vals[:-1]
    
    ret.reverse()
    adv = np.asarray(ret) - q_vals
    
    adv = adv - np.mean(adv)
    adv = adv / np.std(adv)
    
    return ret, adv
  
  def _norm_pdf(self, x, mu, sig):
    var = tf.cast(K.square(sig), tf.float32)
    denom = tf.cast(K.sqrt(2 * math.pi * var), tf.float32)
    deviation = tf.cast(K.square(x - mu), tf.float32)
    expon = - (1 / 2) * deviation / (var + self.cfg['num_stab'])
    return (1 / denom) * K.exp(expon)
  
  def _log_norm_pdf(self, x, mu, sig):
    var = tf.cast(K.square(sig), tf.float32)
    denom = tf.cast(K.sqrt(2 * math.pi * var), tf.float32)
    deviation = tf.cast(K.square(x - mu), tf.float32)
    expon = - (1 / 2) * deviation / (var + self.cfg['num_stab'])
    return expon - K.log(denom + self.cfg['num_stab'])
  
  def _ppo_clip_loss(self, pi_new, pi_old, advantage):
    ratio = pi_new / (pi_old + self.cfg['num_stab'])
    clip_ratio = K.clip(ratio, min_value=1 - self.cfg['ppo_clip'], max_value=1 + self.cfg['ppo_clip'])

    surrogate1 = ratio * advantage
    surrogate2 = clip_ratio * advantage
    
    return - K.mean(K.minimum(surrogate1, surrogate2))
  
  def _entropy_loss_discrete(self, pi_new):
    entropy_loss = (pi_new * K.log(pi_new + self.cfg['num_stab']))
    return - K.mean(entropy_loss)
  
  def _entropy_norm_pdf(self, sig):
    entropy_loss = K.log(K.sqrt(2 * math.pi * math.e * K.square(sig)) + self.cfg['num_stab'])
    return - K.mean(entropy_loss)
  
  def _train_actor(self, advantages):
    y_true = self.action_memory

    loss_total = 0
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
        batch_y_true = np.asarray([y_true[i] for i in sample_idx])
        batch_advantage = np.asarray([advantages[i] for i in sample_idx])
        
        batch_action_mu = [x[0] for x in batch_action_dist]
        batch_action_sig = [x[1] for x in batch_action_dist]
        
        with tf.GradientTape() as tape:
          batch_y_pred_mu, batch_y_pred_sig = self.actor(batch_states)
          
          pi_new = self._norm_pdf(batch_y_true, batch_y_pred_mu, batch_y_pred_sig)
          pi_old = self._norm_pdf(batch_y_true, batch_action_mu, batch_action_sig)
          
          ppo_clip_loss = self._ppo_clip_loss(pi_new=pi_new, pi_old=pi_old, advantage=batch_advantage)
          entropy_loss_norm_pdf = self._entropy_norm_pdf(batch_action_sig) * self.cfg['ppo_entropy_factor']
          
          loss = ppo_clip_loss + entropy_loss_norm_pdf
          loss_total += loss
          
        gradient = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradient, self.actor.trainable_variables))
    return loss_total / (batches_amt + 1)

  def _train_critic(self, returns):
    y_true = returns

    loss_total = 0
    sample_amt = len(self.state_memory)
    sample_range, batches_amt = np.arange(sample_amt), sample_amt // self.cfg['critic_batchsize']
    
    batch_y_pred_prev = np.zeros(shape=(self.cfg['critic_batchsize'], 1))

    for _ in range(self.cfg['critic_epochs']):
      for i in range(batches_amt):
        if self.cfg['critic_shuffle']:
          np.random.shuffle(sample_range)
          sample_idx = sample_range[:self.cfg['critic_batchsize']]
        else:
          sample_idx = sample_range[i * self.cfg['critic_batchsize']:(i + 1) * self.cfg['critic_batchsize']]
        
        batch_states = np.asarray([self.state_memory[i] for i in sample_idx])
        batch_y_true = np.asarray([y_true[i] for i in sample_idx])
        
        with tf.GradientTape() as tape:
          batch_y_pred = self.critic(batch_states)
          clipped_pred = K.clip(batch_y_pred, min_value=batch_y_pred_prev - self.cfg['qval_clip'], max_value=batch_y_pred_prev + self.cfg['qval_clip'])

          surrogate1 = K.square(batch_y_pred - batch_y_true)
          surrogate2 = K.square(clipped_pred - batch_y_true)

          loss = - K.mean(K.minimum(surrogate1, surrogate2))
          loss_total += loss
        
        gradient = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradient, self.critic.trainable_variables))
    return loss_total / (batches_amt + 1)

  def train_agent(self):
    # calculate returns and advantages
    returns, advantages = self._calculate_gae(self.q_value_memory, self.reward_memory, self.not_done_memory)
    
    avg_actor_loss = self._train_actor(advantages)
    avg_critic_loss = self._train_critic(returns)
    
    self._reset_memory()

    return avg_actor_loss, avg_critic_loss

  def _test_run(self):
    s, ep_score, done = self.env.reset(), 0, False
    
    print('Starting test run ...')
    while not done:
      self.env.render()
      a, a_dist = self.actor_choose(s)
      s_, r, done, _ = self.env.step(a)
      ep_score += r
      q_val = self.critic_evaluate(s)
      s = s_  
    print('Ending test run ...')
    return ep_score
  
  def test(self):
    avg_reward = np.mean([self._test_run() for _ in range(5)])
    print(f'Test returned avg reward of {avg_reward} over {5} runs')
  
  def save(self):
    pass

  def run(self):
    PRINT_INTERVALL, scores = self.cfg['print_intervall'], []
    EPISODES, ROLLOUT_EPISODES = self.cfg['total_episodes'], self.cfg['rollout_episodes']
    
    for episode in range(EPISODES):
      s, ep_score, done = self.env.reset(), 0, False
      
      while not done: 
        self.env.render()
        a, a_dist = self.actor_choose(s)
        s_, r, done, _ = self.env.step(a)
        ep_score += r
        q_val = self.critic_evaluate(s)
        self.store_transition(s, a, a_dist, r, q_val, not done)
        s = s_

      scores.append(ep_score)

      if episode % PRINT_INTERVALL == 0:
        print(f'Episode {episode}, Score {np.mean(scores[-PRINT_INTERVALL:])}')

      if episode % ROLLOUT_EPISODES:
        agent.train_agent()

""" ******************************************************************************** """

if __name__ == "__main__":
  agt_cfg = cfg.cont_ppo_cartpoal_cfg
  agent = Agent(cfg=agt_cfg)

  agent.run()