from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

def _simple_actor_net(input_dim, output_dim):
  state = Input(shape=input_dim)

  dense1 = Dense(64, activation='relu')(state)
  dense2 = Dense(64, activation='relu')(dense1)
  dense3 = Dense(32, activation='relu')(dense2)
  action_mu = Dense(output_dim, activation='tanh')(dense3)
  action_sig = Dense(output_dim, activation='softplus')(dense3)

  return Model(inputs=state, outputs=[action_mu, action_sig])

def _simple_critic_net(input_dim):
  state = Input(shape=input_dim)
    
  dense1 = Dense(64, activation='relu')(state)
  dense2 = Dense(64, activation='relu')(dense1)
  dense3 = Dense(32, activation='relu')(dense2)
  q_val = Dense(1, activation='tanh')(dense3)
    
  model = Model(inputs=state, outputs=q_val)
  return model

def _twolayer_mlp_policy_net(input_dim, output_dim):
  state = Input(shape=input_dim)

  dense1 = Dense(200, activation='tanh')(state)
  dense2 = Dense(100, activation='tanh')(dense1)
  dense3 = Dense(64, activation='tanh')(dense2)
  action_mu = Dense(output_dim, activation='tanh')(dense2)
  action_sig = Dense(output_dim, activation='softplus')(dense2)
  v_est = Dense(1, activation='linear')(dense3)

  return Model(inputs=state, outputs=[action_mu, action_sig, v_est])

def _twolayer_complex_mlp_policy_net(input_dim, output_dim):
  state = Input(shape=input_dim)

  dense1 = Dense(128, activation='tanh')(state)
  dense2 = Dense(128, activation='tanh')(dense1)
  dense3 = Dense(64, activation='tanh')(dense2)
  action_mu = Dense(output_dim, activation='tanh')(dense2)
  action_sig = Dense(output_dim, activation='softplus')(dense2)
  v_est = Dense(1, activation='tanh')(dense3)

  return Model(inputs=state, outputs=[action_mu, action_sig, v_est])




def _twolayer_mlp_actor_net(input_dim, output_dim):
  state = Input(shape=input_dim)

  dense1 = Dense(64, activation='tanh')(state)
  dense2 = Dense(64, activation='tanh')(dense1)
  action_mu = Dense(output_dim, activation='tanh')(dense2)
  action_sig = Dense(output_dim, activation='softplus')(dense2)

  return Model(inputs=state, outputs=[action_mu, action_sig])

def _twolayer_mlp_critic_net(input_dim, output_dim):
  state = Input(shape=input_dim)
    
  dense1 = Dense(64, activation='relu')(state)
  dense2 = Dense(64, activation='relu')(dense1)
  v_est = Dense(output_dim, activation=None)(dense2)
  
  return Model(inputs=state, outputs=v_est)




def _super_simple_actor_net(input_dim, output_dim):
  state = Input(shape=input_dim)

  dense1 = Dense(64, activation='relu')(state)
  action_mu = Dense(output_dim, activation='tanh')(dense1)
  action_sig = Dense(output_dim, activation='softplus')(dense1)

  return Model(inputs=state, outputs=[action_mu, action_sig])

def _super_simple_critic_net(input_dim):
  state = Input(shape=input_dim)
    
  dense1 = Dense(64, activation='relu')(state)
  q_val = Dense(1, activation='tanh')(dense1)
    
  model = Model(inputs=state, outputs=q_val)
  return model