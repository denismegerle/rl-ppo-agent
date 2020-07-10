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


class Agent(object):
  
  def __init__(self, cfg):
    self.cfg = cfg
    self.env = self.cfg['environment']
    
    self.input_dim = self.env.observation_space.shape
    self.n_actions = self.env.action_space.shape[0]
    
    self.actor_optimizer = Adam(self.cfg['alpha_actor'])
    self.actor = self._build_policy_network(self.input_dim, self.n_actions)

    # memory
    self._reset_memory()

  def _reset_memory(self):
    self.state_memory, self.not_done_memory = [], []
    self.action_memory, self.action_dist_memory = [], []
    self.reward_memory, self.v_est_memory = [], []
    self.last_vest_buffer = 0.0
    
  def _build_policy_network(self, input_dim, output_dim):
    model = self.cfg['actor_model'](input_dim, output_dim)
    model.build(input_shape=input_dim)
    return model
  
  def actor_choose(self, state):
    a_mu, a_sig, _ = self.actor(K.expand_dims(state, axis=0))
    action = np.random.normal(loc=a_mu[0], scale=a_sig[0], size=None)
    action = np.clip(action, -1.0, 1.0)
    #action = self.cfg['environment'].action_space.low + ((action + 1.0) / 2.0) * (self.cfg['environment'].action_space.high - self.cfg['environment'].action_space.low)
    #action = np.clip(action, self.cfg['environment'].action_space.low, self.cfg['environment'].action_space.high)
    return action, [a_mu[0], a_sig[0]]
  
  def critic_evaluate(self, state):
    _, _, v_est = self.actor(K.expand_dims(state, axis=0))
    return v_est

  def store_transition(self, state, action, action_dist, reward, v_est, not_done):
    self.state_memory.append(state)
    self.action_memory.append(action)
    self.action_dist_memory.append(action_dist)
    self.reward_memory.append(reward)
    self.v_est_memory.append(v_est)
    self.not_done_memory.append(not_done)

  def _calculate_gae(self, v_ests, rews, not_dones):
    def discount(x, gamma):
      return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
    
    vests, notdones = np.asarray(v_ests + [self.last_vest_buffer]).flatten(), np.asarray(not_dones)

    deltas = rews + self.cfg['gae_gamma'] * vests[1:] * notdones - vests[:-1]
    advs = discount(deltas, self.cfg['gae_gamma'] * self.cfg['gae_lambda'])
    
    qvals = advs + vests[:-1]
    advs = (advs - advs.mean()) / np.maximum(advs.std(), self.cfg['num_stab'])
    
    return qvals, advs
    
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
  
  def _entropy_norm_pdf(self, sig):
    entropy_loss = K.log(K.sqrt(2 * math.pi * math.e * K.square(sig)) + self.cfg['num_stab'])
    return - K.mean(entropy_loss)
  
  def _train(self, returns, advantages):
    y_true_actions = self.action_memory
    y_true_returns = returns
    
    loss_total, vest_loss_total, ppo_loss_total = 0, 0, 0
    sample_amt = len(self.action_memory)
    sample_range, batches_amt = np.arange(sample_amt), sample_amt // self.cfg['actor_batchsize']
    
    batch_y_pred_vest_prev = np.zeros(shape=(self.cfg['actor_batchsize'], 1))
    
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
        
        batch_action_mu = [x[0] for x in batch_action_dist]
        batch_action_sig = [x[1] for x in batch_action_dist]
        
        with tf.GradientTape() as tape:
          batch_y_pred_mu, batch_y_pred_sig, batch_y_pred_vest = self.actor(batch_states)
          
          pi_new = self._norm_pdf(batch_y_true_actions, batch_y_pred_mu, batch_y_pred_sig)
          pi_old = self._norm_pdf(batch_y_true_actions, batch_action_mu, batch_action_sig)
          
          ppo_clip_loss = self._ppo_clip_loss(pi_new=pi_new, pi_old=pi_old, advantage=batch_advantage)
          entropy_loss_norm_pdf = self._entropy_norm_pdf(batch_action_sig) * self.cfg['ppo_entropy_factor']

          clipped_vest = K.clip(batch_y_pred_vest, min_value=batch_y_pred_vest_prev - self.cfg['vest_clip'], max_value=batch_y_pred_vest_prev + self.cfg['vest_clip'])

          surrogate1 = K.square(batch_y_pred_vest - batch_y_true_returns)
          surrogate2 = K.square(clipped_vest - batch_y_true_returns)

          vest_loss = K.mean(K.minimum(surrogate1, surrogate2))
          
          loss = ppo_clip_loss + entropy_loss_norm_pdf + 0.5 * vest_loss
          
          ppo_loss_total += ppo_clip_loss
          vest_loss_total += vest_loss
          loss_total += loss
        
        batch_y_pred_vest_prev = batch_y_pred_vest
        
        gradient = tape.gradient(loss, self.actor.trainable_variables)
        gradient, _ = tf.clip_by_global_norm(gradient, clip_norm=0.5)
        self.actor_optimizer.apply_gradients(zip(gradient, self.actor.trainable_variables))
    return loss_total / (batches_amt * self.cfg['actor_epochs']), ppo_loss_total / (batches_amt * self.cfg['actor_epochs']), vest_loss_total / (batches_amt * self.cfg['actor_epochs'])

  def train(self):
    # calculate returns and advantages
    returns, advantages = self._calculate_gae(self.v_est_memory, self.reward_memory, self.not_done_memory)
    returns = returns / np.max(np.abs(returns))
    advantages = advantages / np.max(np.abs(advantages))
    
    avg_actor_loss, avg_ppo_loss, avg_vest_loss = self._train(returns, advantages)
    
    self._reset_memory()

    return avg_actor_loss, avg_ppo_loss, avg_vest_loss

  def learn(self):
    scores, losses, episode = [], {'total' : {'x' : [], 'y' : []}, 'ppo' : {'x' : [], 'y' : []}, 'vest' : {'x' : [], 'y' : []}}, 0
    s, ep_score, done = self.env.reset(), 0, True
    
    for step in range(self.cfg['total_steps']):
      # choose and take an action, advance environment and store data
      self.env.render()
      a, a_dist = self.actor_choose(s)
      s_, r, done, _ = self.env.step(a)
      ep_score += r
      v_est = self.critic_evaluate(s)
      self.store_transition(s, a, a_dist, r, v_est, not done)
      s = s_
      
      # resetting environment if instance is terminated
      if done:
        scores.append(ep_score)
        print(f'Episode {episode}, Score {ep_score}')
        
        if episode % self.cfg['print_interval'] == 0:
          print(f"Mean - Episode {episode}, Score {np.mean(scores[-self.cfg['print_interval']:])}")
        
        s, ep_score, done = self.env.reset(), 0, False
        episode += 1

      if step % self.cfg['rollout'] == 0 and step > 0:
        self.last_vest_buffer = self.critic_evaluate(s_)
        loss_total, ppo_loss, vest_loss = agent.train()
        print(f'train agent (v_est_loss) at step {step} = {vest_loss}')
        
        losses['total']['x'].append(step)
        losses['ppo']['x'].append(step)
        losses['vest']['x'].append(step)
        
        losses['total']['y'].append(loss_total)
        losses['ppo']['y'].append(ppo_loss)
        losses['vest']['y'].append(vest_loss)
        
        if episode >= 3000:
          scio.savemat('losses_total.mat', losses['total'])
          scio.savemat('losses_ppo.mat', losses['ppo'])
          scio.savemat('losses_vest.mat', losses['vest'])
          scio.savemat('epscores.mat', {'x' : [i for i in range(len(scores))], 'y' : scores})
""" ******************************************************************************** """

if __name__ == "__main__":
  tf.random.set_seed(1)
  np.random.seed(1)
  
  agt_cfg = cfg.cont_ppo_test_cartpole_cfg
  agent = Agent(cfg=agt_cfg)

  agent.learn()