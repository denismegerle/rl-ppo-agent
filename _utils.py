import numpy as np

from functools import reduce
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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
    return [x[0] for x in super().step(action)]


class RunningMeanStd(object):
  
  def __init__(self):
    self.reset()
  
  def reset(self):
    self.k = 0
    self.M_k = 0
    self.S_k = 0
  
  def update(self, x):
    self.k += 1
    delta_x_mean = x - self.M_k
    self.M_k = self.M_k + delta_x_mean / self.k
    self.S_k = self.S_k + delta_x_mean * (x - self.M_k)
    
  def __call__(self, x):
    self.update(x)
    return self.M_k, self.S_k / self.k