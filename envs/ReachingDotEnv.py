import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import copy
import cv2

class ReachingDotEnv(gym.Env):
  metadata = { 'render.modes' : ['rgb_array'] }
  
  def __init__(self, seed=None):
    super(ReachingDotEnv, self).__init__()
    
    # ENV PARAMETERS
    self.dot_size = [1, 1]
    self.random_start = True
    self.max_steps = 100
    self.env_size = 32 #32
    self.min_action, self.max_action = -1.0, 1.0
    self.reached_thresh = 2.0
    
    # ENV SETUP
    self.observation_space = spaces.Box(low=-self.env_size,
                                        high=self.env_size,
                                        shape=(4,))
    self.action_space = spaces.Box(low=self.min_action,
                                   high=self.max_action,
                                   shape=(2,))
    seed, self.viewer = None, None

    # allocate for render
    self.render_img = np.zeros(shape=(2 * self.env_size, 2 * self.env_size, 3), dtype=np.uint8)
    
    self.seed(seed)
    self.np_random, _ = seeding.np_random(seed)
    self.reset()

  def reset(self):
    if self.random_start:
      x_goal = self.np_random.randint(low=-self.env_size / 2, high=self.env_size / 2)
      y_goal = self.np_random.randint(low=-self.env_size / 2, high=self.env_size / 2)
      x_agent = self.np_random.randint(low=-self.env_size / 2, high=self.env_size / 2)
      y_agent = self.np_random.randint(low=-self.env_size / 2, high=self.env_size / 2)
      
      self.pos_goal = [x_goal, y_goal]
      self.pos_agent = [x_agent, y_agent]
    self.steps = 0
    self.render_img[self.render_img > 0] = 0
    return self._get_obs()
  
  def _get_obs(self):
    return np.asarray(self.pos_goal + self.pos_agent)
  
  def _update_render(self):
    self.render_img[self.render_img > 0] -= 5         # fade out trajectory

    # update render of agent pos
    x, y, c = int(self.pos_goal[0] + self.env_size), int(self.pos_goal[1] + self.env_size), [0]
    #x = [x-1, x, x+1] if x > 0 and x < 2 * self.env_size else [x]
    #y = [y-1, y, y+1] if y > 0 and y < 2 * self.env_size else [y]
    (x, y) = np.clip((x, y), a_min=0, a_max=2 * self.env_size - 1)
    self.render_img[x, y, c] = 255

    # update render of agent pos
    x, y, c = int(self.pos_agent[0] + self.env_size), int(self.pos_agent[1] + self.env_size), [0, 1, 2]
    (x, y) = np.clip((x, y), a_min=0, a_max=2 * self.env_size - 1)
    self.render_img[x][y][c] = 255

  def step(self, action):
    assert self.action_space.contains(action), "action not in specified bound"
    
    prev_pos_agent, prev_pos_goal = copy.deepcopy(self.pos_agent), copy.deepcopy(self.pos_goal)
    xd, yd = action
    
    self.pos_agent[0] += xd
    self.pos_agent[1] += yd
    
    self._update_render()

    if self.observation_space.contains(self._get_obs()):
      prev_dist = self._eucl_dist(np.asarray(prev_pos_agent), np.asarray(prev_pos_goal))
      cur_dist = self._eucl_dist(np.asarray(self.pos_agent), np.asarray(self.pos_goal))
      
      if cur_dist < prev_dist:
        reward = 1
      elif cur_dist == prev_dist:
        reward = 0
      else:
        reward = -1
      
      #reward = - cur_dist / (self.env_size)
      reward = - cur_dist / self.env_size

      if cur_dist <= self.reached_thresh:
        reward = 100
        done = True
        print("Success")
      else:
        done = False
    else:
      reward = -100
      done = True
    
    self.steps += 1
    if self.steps >= self.max_steps: done = True
    
    return self._get_obs(), reward, done, {}
    
  def _eucl_dist(self, x, y):
    return np.linalg.norm(x - y)

  def _get_image(self):
    resized = cv2.resize(self.render_img, (512, 512))
    return resized

  def render(self, mode='human'):
    img = self._get_image()
    if mode == 'rgb_array':
      return img
    elif mode == 'human':
      from gym.envs.classic_control import rendering
      if self.viewer is None:
        self.viewer = rendering.SimpleImageViewer()
      self.viewer.imshow(img)
      return self.viewer.isopen


if __name__ == "__main__":
  env = ReachingDotEnv()
  s, ep_score, done = env.reset(), 0, False
  
  while not done: 
    env.render()
    a = env.action_space.sample()
    s_, r, done, _ = env.step(a)
  
    print(f'State:{s}, Action:{a}\nReward:{r}\n\n')

    #if done:
    #  s_ = self.env.reset()
      
    ep_score += r
    s = s_