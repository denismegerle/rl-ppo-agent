import gym, os, sys
import numpy as np
import tensorflow as tf

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ContCartpoalEnv import ContinuousCartPoleEnv

sys.path.append('../SimulationFramework/simulation/src/gym_envs//mujoco/')
from gym_envs.mujoco.reach_env import ReachEnv

"""
class RewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(RewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0 and self.n_calls > 1:

          print(self.globals['rewards'])

        return True
"""      

net_arch = [dict(pi=[128, 128, 128], vf=[128, 128, 128])]
policy_kwargs = dict(net_arch=net_arch)

env = DummyVecEnv([lambda: ReachEnv(render=True)])
# Automatically normalize the input features and reward
env = VecNormalize(env, norm_obs=True, norm_reward=True,
                   clip_obs=10.)

model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=1000000)

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()