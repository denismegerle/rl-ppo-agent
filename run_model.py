from matplotlib import animation
import matplotlib.pyplot as plt
import gym 
from envs.ContCartpoalEnv import ContinuousCartPoleEnv
import tensorflow.keras.backend as K
import tensorflow as tf
import mujoco_py
from tensorflow_probability import distributions as tfd
import numpy as np
from _utils import NormalizeWrapper

run_cfg = {
    'environment' : (lambda : gym.make('HalfCheetah-v2')),
    'actor_path': 'logs/ppoagent/TimeLimit/20200805-145233/models/990000/actor.h5',
    'logstd' : 'logs/ppoagent/TimeLimit/20200805-145233/models/990000/logstd.npy'
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

def _get_dist(means, log_stds):
    return tfd.Normal(loc=means, scale=K.exp(log_stds))

def generate_action(env, model, log_std, state):
    a_mu = model(K.expand_dims(state, axis=0))
    dist = _get_dist(a_mu[0], log_std)
    unscaled_action = np.clip(dist.sample(), -1.0, 1.0)
    return scale_action(env, unscaled_action)



env = NormalizeWrapper(run_cfg['environment'],
                    norm_obs=True, norm_reward=True,
                    clip_obs=10., clip_reward=10.,
                    gamma=0.99, epsilon=1e-8)
model = tf.keras.models.load_model(f"{run_cfg['actor_path']}", compile=False)
logstd = np.load(f"{run_cfg['logstd']}")

observation, done = env.reset(), False
frames = []
while not done:
    frames.append(env.render(mode="rgb_array"))
    action = generate_action(env, model, logstd, observation)
    observation, _, done, _ = env.step(action)
    if done:
        break
        observation, done = env.reset(), False
env.close()

save_frames_as_gif(frames)