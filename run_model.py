import gym

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from matplotlib import animation
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tensorflow_probability import distributions as tfd

# config and local imports
import _cfg
from _utils import NormalizeWrapper




run_cfg = {
    **_cfg.reach_env_nonrandom_cfg,

    'render' : False,
    'generate_gif' : False,
    'load_prefix': 'logs/ppoagent/ReachEnv/20200813-043259/models/499999',
}

"""
Frames to GIF part is from
  botforge@github [https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553]
"""
def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

def scale_action(env, unscaled_action):
    action_space_means = (env.action_space.high + env.action_space.low) / 2.0
    action_space_magnitude = (env.action_space.high - env.action_space.low) / 2.0
    return action_space_means + unscaled_action * action_space_magnitude

def get_dist(means, log_stds, discrete):
    if discrete:
        return tfd.Categorical(logits=means)
    else:
        return tfd.Normal(loc=means, scale=K.exp(log_stds))

def generate_action(env, model, log_std, state, discrete, scale):
    a_mu = model(K.expand_dims(state, axis=0))[0]

    dist = get_dist(a_mu, log_std, discrete)

    if discrete:
      scaled_action = unscaled_action = dist.sample().numpy()
    else:
      unscaled_action = np.clip(dist.sample(), -1.0, 1.0)
      
      if scale:
        scaled_action = scale_action(env, unscaled_action)
      else: scaled_action = unscaled_action

    return scaled_action

# reloading environment and status
env = NormalizeWrapper.load(f"{run_cfg['load_prefix']}/env.pkl", run_cfg['environment'])
discrete = True if isinstance(env.action_space, gym.spaces.Discrete) else False

# reloading policy model
model = tf.keras.models.load_model(f"{run_cfg['load_prefix']}/actor.h5", compile=False)
logstd = np.load(f"{run_cfg['load_prefix']}/logstd.npy")

# running a simulation of
observation, done, frames = env.reset(), False, []
steps, STEP_LIMIT = 0, 500
while not done:
    if run_cfg['render']:
        env.render()
    
    if run_cfg['generate_gif']:
        frames.append(env.render(mode="rgb_array"))

    action = generate_action(env, model, logstd, observation, discrete, run_cfg['scale_actions'])
    observation, _, done, _ = env.step(action)

    if done and steps < STEP_LIMIT:
        observation, done = env.reset(), False
    
    if steps >= STEP_LIMIT:
        break
    
    steps += 1
    print(steps)
env.close()

if run_cfg['generate_gif']:
    save_frames_as_gif(frames)