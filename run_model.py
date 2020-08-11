from matplotlib import animation
import matplotlib.pyplot as plt
import gym 
from envs.ContCartpoalEnv import ContinuousCartPoleEnv
import tensorflow.keras.backend as K
import tensorflow as tf
#import mujoco_py
from tensorflow_probability import distributions as tfd
import numpy as np
from _utils import NormalizeWrapper
import _cfg

run_cfg = {
    **_cfg.pendulum_v0_cfg,

    'actor_path': 'logs/ppoagent/Pendulum-v0/20200811-073758/models/999999/actor.h5',
    'logstd_path' : 'logs/ppoagent/Pendulum-v0/20200811-073758/models/999999/logstd.npy'
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


env = NormalizeWrapper(run_cfg['environment'],
                    norm_obs=False, norm_reward=run_cfg['normalize_rewards'],
                    clip_obs=10.0, clip_reward=run_cfg['clip_rewards'],
                    gamma=run_cfg['gamma_env_normalization'], epsilon=run_cfg['num_stab_envnorm'])
discrete = True if isinstance(env.action_space, gym.spaces.Discrete) else False

model = tf.keras.models.load_model(f"{run_cfg['actor_path']}", compile=True)
logstd = np.load(f"{run_cfg['logstd_path']}")

observation, done = env.reset(), False
frames = []
steps, step_limit = 0, 500
while not done:
    env.render()
    #frames.append(env.render(mode="rgb_array"))

    action = generate_action(env, model, logstd, observation, discrete, run_cfg['scale_actions'])
    observation, _, done, _ = env.step(action)

    if done and steps < step_limit:
        observation, done = env.reset(), False
    
    if steps >= step_limit:
        break
    
    steps += 1
    print(steps)
env.close()

#save_frames_as_gif(frames)