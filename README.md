# Proximal Policy Optimization with state-of-the-art code level optimizations

A basic, configurable CONTINUOUS PPO agent implemented in TF 2.2 (TF Keras).

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


- [x] saving/loading `tf.keras` models
- [ ] tensorboard integration, logging of
  - [x] hyperparameters
  - [x] graph + image of model
  - [x] losses, optimizer lrs
  - [x] environment (rewards, actions, observations histograms)
  - [x] stateless logstd and clip ratio

- [x] remove prints in terminals, only use a progressbar and tensorboard for the rest
- [ ] provide configs / gifs for some environments
  - ...
- [ ] compile seeds together for replicability
- [ ] run_env file that loads model, runs env and prints reward + video if possible




 

-- rest:
- use forced types in parameters dosmth(epsilon: float = 1e-4, ...)
- document optimizations ... (and give sources of each :))
- rewrite readme
- comment the agent...
- allow for discrete environments.

- refine readme, add todolist there + whats done already.
- embed gifs of working agents + learning rates/adv/rewards per step/episode for different tasks
- create a nice test file for saved models, "model zoo" thingy

## Dependencies

- pip requirements
- imagemagick for creating gifs of env runs
- graphviz for tf keras model to graph in tensorboard


## References
* _[1]_ Basic Policy Gradient Algorithm -> *https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html*
* _[2]_ A2C architecture -> *Mnih, Volodymyr, et al. "Asynchronous methods for deep reinforcement learning." International conference on machine learning. 2016.*
* _[3]_ Basic PPO Agent -> *Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).*
* _[4]_ GAE -> *Schulman, John, et al. "High-dimensional continuous control using generalized advantage estimation." arXiv preprint arXiv:1506.02438 (2015).*
* _[5]_ Common Improvements -> *Engstrom, Logan, et al. "Implementation Matters in Deep RL: A Case Study on PPO and TRPO." International Conference on Learning Representations. 2019.*
* _[6]_ StableBaselines3 -> *Raffin et al, "StableBaselines3", GitHub, https://github.com/DLR-RM/stable-baselines3*



* [4] ContCartpoalEnv -> this environment is from Ian Danforth https://gist.github.com/iandanforth/e3ffb67cf3623153e968f2afdfb01dc8
