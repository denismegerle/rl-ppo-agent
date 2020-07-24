import numpy as np

from functools import reduce
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tensorflow.keras.optimizers.schedules import InverseTimeDecay

#################### FUNCTIONALS ####################
def scan(func, acc, xs):  # implementation of haskell scanl
  for x in xs:
    acc = func(acc, x)
    yield acc


foldl = reduce
foldr = lambda func, acc, xs: reduce(lambda x, y: func(y, x), xs[::-1], acc)
scanl = lambda func, acc, xs: list(scan(func, acc, xs))
scanr = lambda func, acc, xs: list(scan(func, acc, xs[::-1]))[::-1]
npscanr = lambda func, acc, xs: np.asarray(scanr(func, acc, xs))


#################### ENVIRONMENTS ####################

class NormalizeWrapper(VecNormalize):
  """TEMPORARY (!) simple wrapper that implements the stable baselines VecNormalize
  environment, but only uses one (the first) actual environment.
  
  Thus this allows to normalize single environments and keep the expected output
  format of that environment.

  Args:
      VecNormalize (VecNormalize): stable baselines vectorized wrapper environment
      that applies normalization and clipping of reward/observation.
  """
  
  def __init__(self, venv, training=True, norm_obs=True, norm_reward=True,
                            clip_obs=10., clip_reward=10., gamma=0.99, epsilon=1e-8):
    env = DummyVecEnv([venv])
    VecNormalize.__init__(self, env, training, norm_obs, norm_reward, clip_obs, clip_reward, gamma, epsilon)

  def reset(self):
    return super().reset()[0]
  
  def step(self, action):
    return [x[0] for x in super().step([action])]


class RunningMeanStd(object):
  
  def __init__(self, shape=()):
    self.shape = shape
    self.reset()
  
  def reset(self):
    self.k = 0
    self.M_k = np.zeros(self.shape)
    self.S_k = np.zeros(self.shape)
  
  def update(self, x):
    self.k += 1
    delta_x_mean = x - self.M_k
    self.M_k = self.M_k + delta_x_mean / self.k
    self.S_k = self.S_k + delta_x_mean * (x - self.M_k)
    
  def __call__(self, x):
    self.update(x)
    
    if self.k == 1:
      return self.M_k, np.ones(self.shape)
    else: return self.M_k, self.S_k / self.k


  # def _normalize_observations(self, observations, not_dones):
  #   def func(obs, not_done):
  #     mean, variance = self.obs_rms(obs)
  #     if not not_done: self.obs_rms.reset()
  #     return mean, variance
    
  #   mean_vars = map(func, observations, not_dones)
  #   normalized_observations = map(lambda obs, m_v: (obs - m_v[0]) / np.sqrt(m_v[1] + 1e-4), observations, mean_vars)
    
  #   return list(normalized_observations)

class RolloutInverseTimeDecay(InverseTimeDecay):
  
  def __init__(self, initial_learning_rate, decay_steps, decay_rate, staircase=False):
    super(RolloutInverseTimeDecay, self).__init__(initial_learning_rate, decay_steps, decay_rate, staircase, name='RolloutInverseTimeDecay')
    self.rollout_step = 0
  
  def update_rollout_step(self, new_step):
    assert new_step > 0 and new_step > self.rollout_step, 'invalid new rollout step defined'
    self.rollout_step = new_step
    
  def __call__(self, step):
    return super().__call__(self.rollout_step)