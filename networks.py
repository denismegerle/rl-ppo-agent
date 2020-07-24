import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.initializers import Constant, Orthogonal

orth_weights_initializer = lambda scale, seed=1 : Orthogonal(gain=scale, seed=seed)
const_bias_initializer = lambda value : Constant(value=value)

# TODO think about normalization after denses

def _twolayer_mlp_actor_net(input_dim, output_dim):
  state = Input(shape=input_dim)

  dense1 = Dense(64,  activation='tanh')(state)
  dense2 = Dense(64,  activation='tanh')(dense1)

  action_mu = Dense(output_dim,   activation='tanh')(dense2)
  action_sig = Dense(output_dim,  activation='softplus')(dense2)

  return Model(inputs=state, outputs=[action_mu, action_sig])

def _twolayer_mlp_critic_net(input_dim, output_dim):
  state = Input(shape=input_dim)
    
  dense1 = Dense(64,  activation='tanh', 
                      kernel_initializer=orth_weights_initializer(np.sqrt(2)), 
                      bias_initializer=const_bias_initializer(0.0))(state)
  dense2 = Dense(64,  activation='tanh', 
                      kernel_initializer=orth_weights_initializer(np.sqrt(2)), 
                      bias_initializer=const_bias_initializer(0.0))(dense1)

  v_est = Dense(output_dim,   activation=None)(dense2)
  
  return Model(inputs=state, outputs=v_est)

def _twolayer_mlp_actor_net_orth(input_dim, output_dim):
  state = Input(shape=input_dim)

  dense1 = Dense(64,  activation='tanh', 
                      kernel_initializer=orth_weights_initializer(np.sqrt(2)), 
                      bias_initializer=const_bias_initializer(0.0))(state)
  dense2 = Dense(64,  activation='tanh', 
                      kernel_initializer=orth_weights_initializer(np.sqrt(2)), 
                      bias_initializer=const_bias_initializer(0.0))(dense1)

  action_mu = Dense(output_dim,   activation='tanh', 
                                  kernel_initializer=orth_weights_initializer(0.01), 
                                  bias_initializer=const_bias_initializer(0.0))(dense2)
  action_sig = Dense(output_dim,  activation='softplus',
                                  kernel_initializer=orth_weights_initializer(np.sqrt(2)), 
                                  bias_initializer=const_bias_initializer(0.0))(dense2)

  return Model(inputs=state, outputs=[action_mu, action_sig])

def _twolayer_mlp_critic_net_orth(input_dim, output_dim):
  state = Input(shape=input_dim)
    
  dense1 = Dense(64,  activation='tanh', 
                      kernel_initializer=orth_weights_initializer(np.sqrt(2)), 
                      bias_initializer=const_bias_initializer(0.0))(state)
  dense2 = Dense(64,  activation='tanh', 
                      kernel_initializer=orth_weights_initializer(np.sqrt(2)), 
                      bias_initializer=const_bias_initializer(0.0))(dense1)

  v_est = Dense(output_dim,   activation=None,
                              kernel_initializer=orth_weights_initializer(1.0), 
                              bias_initializer=const_bias_initializer(0.0))(dense2)
  
  return Model(inputs=state, outputs=v_est)