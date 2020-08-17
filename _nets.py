import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.initializers import Constant, Orthogonal
from tensorflow.keras.regularizers import l2


orth_weights_initializer = lambda scale : Orthogonal(gain=scale)
const_bias_initializer = lambda value : Constant(value=value)
l2_regularizer = lambda base_val=1.0 : l2(l=base_val)

# BN + dropout only for offline algorithms, see Liu et al Feb. 2020

def _mlp_actor_net_orth(hidden_layers=[64, 64]):
  def network_func(input_dim, output_dim):
    state = Input(shape=input_dim)
    
    x = state
    for i in range(len(hidden_layers)):
      x = Dense(hidden_layers[i],  activation='tanh', 
                      kernel_initializer=orth_weights_initializer(np.sqrt(2)), 
                      bias_initializer=const_bias_initializer(0.0),
                      kernel_regularizer=l2_regularizer())(x)
    
    action_mu = Dense(output_dim, activation='tanh', 
                                  kernel_initializer=orth_weights_initializer(0.01), 
                                  bias_initializer=const_bias_initializer(0.0),
                                  kernel_regularizer=l2_regularizer())(x)
    
    return Model(inputs=state, outputs=[action_mu])

  return network_func


def _mlp_critic_net_orth(hidden_layers=[64, 64]):
  def network_func(input_dim, output_dim):
    state = Input(shape=input_dim)
    
    x = state
    for i in range(len(hidden_layers)):
      x = Dense(hidden_layers[i],  activation='tanh', 
                      kernel_initializer=orth_weights_initializer(np.sqrt(2)), 
                      bias_initializer=const_bias_initializer(0.0),
                      kernel_regularizer=l2_regularizer())(x)
    
    v_est = Dense(output_dim,   activation=None,
                              kernel_initializer=orth_weights_initializer(1.0), 
                              bias_initializer=const_bias_initializer(0.0),
                              kernel_regularizer=l2_regularizer())(x)
    
    return Model(inputs=state, outputs=[v_est])

  return network_func



def _mlp_forward_dynamics(hidden_layers=[64, 64]):
  def network_func(input_dim, output_dim):
    emb_state = Input(shape=input_dim[0])
    action = Input(shape=input_dim[1])
    
    x = tf.keras.layers.concatenate([emb_state, action])
    for i in range(len(hidden_layers)):
      x = Dense(hidden_layers[i],  activation='elu')(x)
    
    emb_next_state = Dense(output_dim)(x)
    
    return Model(inputs=[emb_state, action], outputs=[emb_next_state])

  return network_func
