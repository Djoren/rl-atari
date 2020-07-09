import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop


def atari_model(n_actions): 
    """ n_actions: OHE matrix of actions."""
    # Input layers
    input_actions = layers.Input((n_actions,), name='input_actions')
    input_frames = layers.Input((105, 80, 4), name='input_frames', dtype=tf.float32)
    normed_frames = layers.Lambda(lambda x: tf.cast(x, tf.float32) / 255.0)(input_frames)  # Convert frames uint8[0, 255] to float[0, 1]

    # Convolutional layers
    conv_1 = layers.Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(normed_frames)
    conv_2 = layers.Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv_1)

    # Fully connected layers
    conv_flat = layers.Flatten()(conv_2)
    hidden = layers.Dense(256, activation='relu')(conv_flat)

    # Output layer. Q values are masked by actions
    # so only selected actions have non-0 Q value
    output = layers.Dense(n_actions, name='Q_est')(hidden)
    output_masked = layers.Multiply(name='Q_est_masked')([output, input_actions])

    # Model
    model = keras.Model(inputs=[input_frames, input_actions], outputs=output_masked)
    optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    # model.compile(optimizer=optimizer, loss='mse')
    model.compile(optimizer=optimizer, loss='huber_loss')
    return model


def fit_batch(model, gamma, state_now, action, reward, game_over, state_next):
    """Do Q-learning update on a batch of transitions."""
    # Reshape variables
    action = np.eye(model.output.shape[1])[action.tolist()]  # OHE encode actions
    state_now = np.stack(state_now)
    state_next = np.stack(state_next)
    game_over = game_over.tolist()
    
    # Predict Q values of the next states for all actions
    Q_val = model.predict([state_next, np.ones(action.shape)])
    Q_val[game_over] = 0  # Q value of terminal state is 0 by definition
    
    # Set Q target. We only run SGD updates for those actions taken.
    # Hence, Q target for non-taken actions is set to 0
    Q_tgt = reward + gamma * Q_val.max(axis=1)
    Q_tgt = np.float32(action * Q_tgt[:, None])
    
    # Run SGD update
    model.fit([state_now, action], Q_tgt, batch_size=len(Q_tgt), epochs=1, verbose=0)


def fit_batch_DQN(model, model_tgt, gamma, state_now, action, reward, game_over, state_next):
    """Do Q-learning update on a batch of transitions."""
    # Reshape variables
    action = np.eye(model.output.shape[1])[action.tolist()]  # OHE encode actions
    state_now = np.stack(state_now)
    state_next = np.stack(state_next)
    game_over = game_over.tolist()
    
    # Predict Q values of the next states for all actions
    Q_val = model_tgt.predict([state_next, np.ones(action.shape)])
    Q_val[game_over] = 0  # Q value of terminal state is 0 by definition
    
    # Set Q target. We only run SGD updates for those actions taken.
    # Hence, Q target for non-taken actions is set to 0
    Q_tgt = reward + gamma * Q_val.max(axis=1)
    Q_tgt = np.float32(action * Q_tgt[:, None])
    
    # Run SGD update
    model.fit([state_now, action], Q_tgt, batch_size=len(Q_tgt), epochs=1, verbose=0)


def fit_batch_DDQN(model, model_tgt, gamma, state_now, action, reward, game_over, state_next):
    """Do Q-learning update on a batch of transitions."""
    # Reshape variables
    action = np.eye(model.output.shape[1])[action.tolist()]  # OHE encode actions
    state_now = np.stack(state_now)
    state_next = np.stack(state_next)
    game_over = game_over.tolist()

    # Get actions that generates highest Q from main net
    Q_val = model.predict([state_next, np.ones(action.shape)])
    Q_val[game_over] = 0  # Q value of terminal state is 0 by definition
    actions_max = Q_val.max(axis=1)
    
    # Predict Q values of the next states for all actions
    # Use target net to get Q values for prev. selected actions
    Q_val = model_tgt.predict([state_next, np.ones(action.shape)])
    Q_val[game_over] = 0  # Q value of terminal state is 0 by definition
    Q_val = Q_val[range(Q_val.shape[0]), actions_max]
    
    # Set Q target. We only run SGD updates for those actions taken.
    # Hence, Q target for non-taken actions is set to 0
    Q_tgt = reward + gamma * Q_val
    Q_tgt = np.float32(action * Q_tgt[:, None])
    
    # Run SGD update
    model.fit([state_now, action], Q_tgt, batch_size=len(Q_tgt), epochs=1, verbose=0)


