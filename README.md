# Proximal Policy Optimization with state-of-the-art code level optimizations



## Implementation

### basic policy gradient algorithm
  - [x] basic policy gradient algorithm, good summary see [1]
  - [x] a2c architecture (adding the critic + return/simple advantage estimation), see [2] for details
  - [x] minibatch gradient descent with automatic differentiation (i.e `tf.GradientTape`)
  - [x] shuffle and permutate-only for MGD

### proper PPO agent and most common improvements 
  - [x] ppo policy (apply ppo clip loss), see Schulmans paper [3]
  - [x] generalized advantage estimation, see GAE paper [4]
  - [x] general improvements in most implementations, see "Implementation Matters" [5]
    - [x] #1, #9: value function clipping and global (policy) gradient clipping
    - [x] #2, #5, #6, #7: reward/observation scaling and clipping, according to `stablebaselines3` [6] VecNormalize environment
    - [x] #3, #4, #8: orth. layer initialization, Adam annealing, tanh activations
  - [x] further improvements
    - [x] minibatch-wise advantage normalization
    - [x] regularization and entropy loss for regularization/exploration
    - [x] stateless (learnable) log std as variance
  - [ ] parallelized environments
  - [x] scaling actions to proper range for environment
  - [x] discrete action space agent

### tf features
- [x] saving/loading `tf.keras` models
- [x] tensorboard integration, logging of
  - [x] hyperparameters
  - [x] graph + image of model
  - [x] losses, optimizer lrs
  - [x] environment (rewards, actions, observations histograms)
  - [x] stateless logstd and clip ratio

- [x] remove prints in terminals, only use a progressbar and tensorboard for the rest
- [x] provide configs / gifs for some environments
- [x] compile seeds together for replicability
- [x] run_env file that loads model, runs env and prints reward + video if possible
- [ ] force types in parameters
- [ ] code point references to the optimizations made




## Sample Runs

### Custom Environments

| ContCartpoalEnv  | Episode Scores / Steps   |  ReachingDotEnv  | Episode Scores / Steps   |
|------------------|--------------------------|------------------|--------------------------|
| ![][contpoalrun] |   ![][contpoalscores]    | ![][reachdotrun] |   ![][reachdotscores]    |
|                  |                          |                  |                          |

---

### Classic Control Environments

| CartPole-v1      | Episode Scores / Steps   |   Pendulum-v0    |  Episode Scores / Steps  |
|------------------|--------------------------|------------------|--------------------------|
|![][cartpolev1run]|  ![][cartpolev1scores]   |![][pendulumv0run]|  ![][pendulumv0scores]   |
|                  |                          |                  |                          |

---

### SimFramework Environments

| ReachEnv-v0      | Episode Scores / Steps   | ReachEnvRandom-v0|  Episode Scores / Steps  |
|------------------|--------------------------|------------------|--------------------------|
|![][reachenvnorun]|  ![][reachenvnoscores]   |![][reachenvyerun]|  ![][reachenvyescores]   |
|                  |                          |                  |                          |

---


## Dependencies

- pip requirements
- imagemagick for creating gifs of env runs
- graphviz for tf keras model to graph in tensorboard
- mujoco_py's offscreen rendering is buggy in gym, for using run_model (GIF generation)
  - adjust mujoco_py.MjRenderContextOffscreen(sim, None, device_id=0) in gym/envs/mujoco/mujoco_env.MujocoEnv._get_viewer(...)



## References
* _[1]_ Basic Policy Gradient Algorithm -> *https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html*
* _[2]_ A2C architecture -> *Mnih, Volodymyr, et al. "Asynchronous methods for deep reinforcement learning." International conference on machine learning. 2016.*
* _[3]_ Basic PPO Agent -> *Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).*
* _[4]_ GAE -> *Schulman, John, et al. "High-dimensional continuous control using generalized advantage estimation." arXiv preprint arXiv:1506.02438 (2015).*
* _[5]_ Common Improvements -> *Engstrom, Logan, et al. "Implementation Matters in Deep RL: A Case Study on PPO and TRPO." International Conference on Learning Representations. 2019.*
* _[6]_ StableBaselines3 -> *Raffin et al, "StableBaselines3", GitHub, https://github.com/DLR-RM/stable-baselines3*
* _[7]_ ContCartpoalEnv -> *this environment is from Ian Danforth https://gist.github.com/iandanforth/e3ffb67cf3623153e968f2afdfb01dc8*


<!-- links -->
[contpoalrun]: logs/ppoagent/ContinuousCartPoleEnv/20200813-014433/gym_animation.gif "ContCartpoalEnv-Run"
[contpoalscores]: logs/ppoagent/ContinuousCartPoleEnv/20200813-014433/scores.png "ContCartpoalEnv-Scores"
[reachdotrun]: logs/ppoagent/ReachingDotEnv/20200813-003741/gym_animation.gif "ReachingDotEnv-Run"
[reachdotscores]: logs/ppoagent/ReachingDotEnv/20200813-003741/scores.png "ReachingDotEnv-Scores"

[cartpolev1run]: logs/ppoagent/CartPole-v1/20200813-022551/gym_animation.gif "CartPoleV1-Run"
[cartpolev1scores]: logs/ppoagent/CartPole-v1/20200813-022551/scores.png "CartPoleV1-Scores"
[pendulumv0run]: logs/ppoagent/Pendulum-v0/20200814-044336/gym_animation.gif "PendulumV0-Run"
[pendulumv0scores]: logs/ppoagent/Pendulum-v0/20200814-044336/scores.png "PendulumV0-Scores"

[reachenvnorun]: logs/ppoagent/ReachEnv/20200813-043259/gym_animation.gif "ReachEnvNoRandom-Run"
[reachenvnoscores]: logs/ppoagent/ReachEnv/20200813-043259/scores.png "ReachEnvNoRandom-Scores"
[reachenvyerun]: logs/ppoagent/ReachEnv/20200816-221850/gym_animation.gif "ReachEnvRandom-Run"
[reachenvyescores]: logs/ppoagent/ReachEnv/20200816-221850/scores.png "ReachEnvRandom-Scores"