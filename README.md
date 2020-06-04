# rl-ppo-agent
A basic, configurable CONTINUOUS PPO agent implemented in TF 2.2 (TF Keras).


## Implementation

### TODO
* ...

### NEXT
* Solving jumping out of local minimum after some training by using Exponential/InverseTime decay of learning rate (#4 of [2])

### DONE
* PPO agent as in [1], except episode instead of constant batch look ahead
* GAE advantage estimation as in [3]
* Simple Reaching the dot environment implemented
* Improvement #1 from [2] (clipping loss of PPO + value loss clipping)

## Paper Sources
* [1] Basic PPO Agent -> *Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).*
* [2] Common Improvements -> *Engstrom, Logan, et al. "Implementation Matters in Deep RL: A Case Study on PPO and TRPO." International Conference on Learning Representations. 2019.*
* [3] GAE -> *Schulman, John, et al. "High-dimensional continuous control using generalized advantage estimation." arXiv preprint arXiv:1506.02438 (2015).*
* [3] ContCartpoalEnv -> this environment is from Ian Danforth https://gist.github.com/iandanforth/e3ffb67cf3623153e968f2afdfb01dc8
