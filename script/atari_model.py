"""Implements all Atari agent models, as well as train functions.

Contains models for:
    1. Base DQN arch
    2. Dueling arch
    3. Distributional arch
    4. Dueling x distributional arch
"""

import numpy as np
from typing import List, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow_addons import layers as layers_tfa
from noisy_dense import NoisyDense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import Huber
from tensorflow.math import reduce_mean


@tf.function
def train_on_batch(
        model: tf.keras.Model, x: np.array, y_tgt: np.array, actions: np.array, 
        sample_weight: np.array =None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Custom fit function such that we can obtain td-error, w/o having to recompute it.
    
    Assumptions:
        1. Model has been passed an optimizer object as argument, not a string.

    Note:
        - Decorating this function as a graph function makes it much faster.
        - To calc loss for each sample: TF takes some form of mean across the output dimension (axis=-1).
          E.g. for MAE it's mean, for cat. cross-entropy it's the sum.
        - TF then by default takes mean across losses for all other dimensions. The name `sum_over_batch_size` seems like a misnomer,
          because it actually divides by the sum of size of all other dimenions but the output dimension.
        - We might not want this for our RL purposes, as our intention is to train for only a single action per sample.
          It might make more sense to take the NOT take the mean across the outputs, but instead a sum. 
          And then divide by true batch size, i.e. first dimension (axis=1).
        - This should not affect direction of gradients, but only the magnitude => has interplay with learning-rate.
        
    Args:
        model: Tf model to train.
        x: Model input values (dims = batch_size x (...)).
        y_tgt: Target values (dims = batch_size x (...)):
        actions: Taken actions for mini batch (dims = batch_size x 1).
        sample_weight: Training sample weights.

    Returns:
        y_pred: Predicted values. Can be used downstream, avoiding need to recompute.
        loss: Training loss.

    TODO: pass in only tensors etc. to avoid retracing (if this is even happening).
    """
     # Forward pass, keep only pred values for actions actually taken
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        indices = tf.expand_dims(actions, -1)  # 1D -> 2D
        y_pred = tf.reshape(tf.gather_nd(y_pred, indices, batch_dims=1), y_tgt.get_shape())
        loss = model.loss(y_tgt, y_pred, sample_weight=sample_weight)

    # Backward pass
    grads = tape.gradient(loss, model.trainable_weights)
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return y_pred, loss


@tf.function
def model_call(model: tf.keras.Model, inputs: np.array) -> tf.Tensor:
    """Call model (making model predictions.)
    
    Note:
        - Decorating this function as a graph function makes it much faster.
    """
    return model(inputs)


def Q_from_Z_distr(Z: np.array, p: np.array) -> float:
    """Computes the estimate Q_hat from Z distribution.
    
    Args:
        Z: Fixed categorical distribution values.
        p: Prob. density for each z in Z.
    """
    return np.sum(Z * p, axis=-1)


def abs_td_error(
        y_pred: np.array, y_tgt: np.array, distr_net: bool = False, Z: np.array = None
    ) -> np.array:
    """Compute TD-error for predicted Q vs target Q, on a mini batch.
    
    Used for setting priorities in Prioritized Experience Replay.

    Args:
        Q_pred: Predicted Q values, or pZ values for distr. net.
        Q_tgt: Target Q values, or pZ values for distr. net.
        distr_net: Indicator if model is a distributional network.
        Z: Fixed categorical distribution values.
    
    TODO: Z should be class attribute when we OOP all this code.
    """
    # If distributive target convert p_Z to Q
    if distr_net:
        y_pred = Q_from_Z_distr(Z, y_pred)
        y_tgt = Q_from_Z_distr(Z, y_tgt)
    return np.abs(y_tgt - y_pred).ravel()


def atari_model(
        N_actions: int, loss, lr: float, state_shape: tuple, 
        kernel_init: str = 'glorot_uniform', noisy_net: bool = False, 
        large_net: bool = False
    ) -> tf.keras.Model: 
    """Composes a DQN convolutional neural network. 

    Args:
        N_actions: Len of action space.
        lr: Learning rate.
        state_shape: Shape of input states.
        kernel_init: Weight initializer.
        noisy_net: Toggle to set dense layers as noisy dense layers.
        large_net: Toggle to set model arch to larger or small.
    """
    # Input layer
    input_frames = layers.Input(state_shape, name='input_frames', dtype=tf.float32)
    normed_frames = layers.Lambda(lambda x: tf.cast(x, tf.float32) / 255.0)(input_frames)  # Cast frames uint8[0, 255] -> float[0, 1]

    # Convolutional and dense layers
    if large_net:
        # Conv layers
        conv_1 = layers.Conv2D(
            32, (8, 8), strides=(4, 4), activation='relu', 
            name='conv1', kernel_initializer=kernel_init
        )(normed_frames)
        conv_2 = layers.Conv2D(
            64, (4, 4), strides=(2, 2), activation='relu', 
            name='conv2', kernel_initializer=kernel_init
        )(conv_1)
        conv_3 = layers.Conv2D(
            64, (3, 3), strides=(1, 1), activation='relu', 
            name='conv3', kernel_initializer=kernel_init
        )(conv_2)

        # FC layers
        layer_dense = NoisyDense if noisy_net else layers.Dense
        conv_flat = layers.Flatten()(conv_3)
        hidden = layer_dense(
            512, activation='relu', name='hid', kernel_initializer=kernel_init
        )(conv_flat)
    else:
        # Conv layers
        conv_1 = layers.Conv2D(
            16, (8, 8), strides=(4, 4), activation='relu', 
            name='conv1', kernel_initializer=kernel_init
        )(normed_frames)
        conv_2 = layers.Conv2D(
            32, (4, 4), strides=(2, 2), activation='relu', 
            name='conv2', kernel_initializer=kernel_init
        )(conv_1)

        # FC layers
        # layer_dense = layers_tfa.NoisyDense if noisy_net else layers.Dense
        # layer_dense = layers.Dense
        layer_dense = NoisyDense if noisy_net else layers.Dense
        conv_flat = layers.Flatten()(conv_2)
        hidden = layer_dense(
            256, activation='relu', name='hid', kernel_initializer=kernel_init
        )(conv_flat)

    # Output layer. Q values are masked by actions so only selected actions have non-0 Q value
    output = layer_dense(N_actions, name='Q', kernel_initializer=kernel_init)(hidden)

    # Configure model
    model = keras.Model(inputs=input_frames, outputs=output)
    optimizer = RMSprop(learning_rate=lr, rho=0.95, epsilon=0.01)
    model.compile(optimizer=optimizer, loss=loss)
    return model


def atari_model_dueling(
        N_actions: int, loss, lr: float, state_shape: tuple, 
        kernel_init: str = 'glorot_uniform', noisy_net: bool = False, 
        large_net: bool = False
    ) -> tf.keras.Model: 
    """Composes a Dueling-DQN convolutional neural network. 

    Args:
        N_actions: Len of action space.
        lr: Learning rate.
        state_shape: Shape of input states.
        kernel_init: Weight initializer.
        noisy_net: Toggle to set dense layers as noisy dense layers.
        large_net: Toggle to set model arch to larger or small.
    """
    # Input layers
    in_frames = layers.Input(state_shape, name='input_frames', dtype=tf.float32)
    normed_frames = layers.Lambda(lambda x: tf.cast(x, tf.float32) / 255.0)(in_frames)  # Cast frames uint8[0, 255] -> float[0, 1]

    # Convolutional and dense layers
    if large_net:
        # Conv layers
        conv_1 = layers.Conv2D(
            32, (8, 8), strides=(4, 4), activation='relu', 
            name='conv1', kernel_initializer=kernel_init
        )(normed_frames)
        conv_2 = layers.Conv2D(
            64, (4, 4), strides=(2, 2), activation='relu', 
            name='conv2', kernel_initializer=kernel_init
        )(conv_1)
        conv_3 = layers.Conv2D(
            64, (3, 3), strides=(1, 1), activation='relu', 
            name='conv3', kernel_initializer=kernel_init
        )(conv_2)

        # FC layers for both value and advantage streams
        layer_dense = NoisyDense if noisy_net else layers.Dense
        conv_flat = layers.Flatten()(conv_3)
        hidden_val = layer_dense(
            512, activation='relu', name='V_hid', kernel_initializer=kernel_init
        )(conv_flat)
        hidden_adv = layer_dense(
            512, activation='relu', name='A_hid', kernel_initializer=kernel_init
        )(conv_flat)
    else:
        # Conv layers
        conv_1 = layers.Conv2D(
            16, (8, 8), strides=(4, 4), activation='relu', 
            name='conv1', kernel_initializer=kernel_init
        )(normed_frames)
        conv_2 = layers.Conv2D(
            32, (4, 4), strides=(2, 2), activation='relu', 
            name='conv2', kernel_initializer=kernel_init
        )(conv_1)

        # FC layers for both value and advantage streams
        layer_dense = NoisyDense if noisy_net else layers.Dense
        conv_flat = layers.Flatten()(conv_2)
        hidden_val = layer_dense(
            256, activation='relu', name='V_hid', kernel_initializer=kernel_init
        )(conv_flat)
        hidden_adv = layer_dense(
            256, activation='relu', name='A_hid', kernel_initializer=kernel_init
        )(conv_flat)

    # Output layers for value and advantage streams
    # Advantage layer is masked so only selected actions have non-0 value
    out_val = layer_dense(1, name='V', kernel_initializer=kernel_init)(hidden_val)
    out_adv = layer_dense(N_actions, name='A', kernel_initializer=kernel_init)(hidden_adv)

    # Adjust advantage output
    out_adv_avg = layers.Lambda(
        lambda x: reduce_mean(x, axis=1, keepdims=True), name='A_avg'
    )(out_adv)
    out_adv_adj = layers.Subtract(name='A_adj')([out_adv, out_adv_avg])

    # Combine both outputs to estimate Q values
    output = layers.Add(name='Q')([out_val, out_adv_adj])

    # Configure model
    model = keras.Model(inputs=in_frames, outputs=output)
    optimizer = RMSprop(learning_rate=lr, rho=0.95, epsilon=0.01)
    model.compile(optimizer=optimizer, loss=loss)
    return model


def atari_model_distr(
        N_atoms: int, N_actions: int, loss, lr: float, state_shape: tuple, 
        kernel_init: str = 'glorot_uniform', noisy_net: bool = False, large_net: bool = False
    ) -> tf.keras.Model: 
    """Composes a Distributional-DQN convolutional neural network. 

    Args:
        N_atoms: Number of Z values for categorical distr.
        N_actions: Len of action space.
        lr: Learning rate.
        loss: Loss object.
        state_shape: Shape of input states.
        kernel_init: Weight initializer.
        noisy_net: Toggle to set dense layers as noisy dense layers.
        large_net: Toggle to set model arch to larger or small.
    """
    # Input layers
    input_frames = layers.Input(state_shape, name='input_frames', dtype=tf.float32)
    normed_frames = layers.Lambda(lambda x: tf.cast(x, tf.float32) / 255.0)(input_frames)  # Cast frames uint8[0, 255] -> float[0, 1]

    # Convolutional and dense layers
    if large_net:
        # Conv layers
        conv_1 = layers.Conv2D(
            32, (8, 8), strides=(4, 4), activation='relu', 
            name='conv1', kernel_initializer=kernel_init
        )(normed_frames)
        conv_2 = layers.Conv2D(
            64, (4, 4), strides=(2, 2), activation='relu', 
            name='conv2', kernel_initializer=kernel_init
        )(conv_1)
        conv_3 = layers.Conv2D(
            64, (3, 3), strides=(1, 1), activation='relu', 
            name='conv3', kernel_initializer=kernel_init
        )(conv_2)

        # FC layers
        layer_dense = NoisyDense if noisy_net else layers.Dense
        conv_flat = layers.Flatten()(conv_3)
        hidden = layer_dense(
            512, activation='relu', name='hid', kernel_initializer=kernel_init
        )(conv_flat)
    else:
        # Conv layers
        conv_1 = layers.Conv2D(
            16, (8, 8), strides=(4, 4), activation='relu', 
            name='conv1', kernel_initializer=kernel_init
        )(normed_frames)
        conv_2 = layers.Conv2D(
            32, (4, 4), strides=(2, 2), activation='relu', 
            name='conv2', kernel_initializer=kernel_init
        )(conv_1)

        # FC layers
        layer_dense = NoisyDense if noisy_net else layers.Dense
        conv_flat = layers.Flatten()(conv_2)
        hidden = layer_dense(
            256, activation='relu', name='hid', kernel_initializer=kernel_init
        )(conv_flat)

    # Output layers Z(a) for every action
    outputs = []
    for i in range(N_actions):
        output_a = layer_dense(N_atoms, activation='softmax', name=f'p_{i}', kernel_initializer=kernel_init)(hidden)
        output_a = layers.Reshape((1, N_atoms))(output_a)  # Adds a dim => (batch_size, 1, N_atoms)
        outputs.append(output_a)
    output = layers.Concatenate(name='p_concat', axis=1)(outputs)

    # Configure model
    # Notes
    #   - tf cross-entropy can handle real values for y_true
    #   - CE is computed across the axis=1 by default (what we want)
    model = keras.Model(inputs=input_frames, outputs=output)
    optimizer = RMSprop(learning_rate=lr, rho=0.95, epsilon=0.01)
    model.compile(optimizer=optimizer, loss=loss)
    return model


def atari_model_dueling_distr(
        N_atoms: int, N_actions: int, loss, lr: float, state_shape: tuple, 
        kernel_init: str = 'glorot_uniform', noisy_net: bool = False, large_net: bool = False
    ) -> tf.keras.Model: 
    """Composes a Dueling-Distributional-DQN convolutional neural network. 

    Args:
        N_atoms: Number of Z values for categorical distr.
        N_actions: Len of action space.
        lr: Learning rate.
        loss: Loss object.
        state_shape: Shape of input states.
        kernel_init: Weight initializer.
        noisy_net: Toggle to set dense layers as noisy dense layers.
        large_net: Toggle to set model arch to larger or small.
    """
    # Input layers
    in_frames = layers.Input(state_shape, name='input_frames', dtype=tf.float32)
    normed_frames = layers.Lambda(lambda x: tf.cast(x, tf.float32) / 255.0)(in_frames)  # Cast frames uint8[0, 255] -> float[0, 1]

    # Convolutional and dense layers
    if large_net:
        # Conv layers
        conv_1 = layers.Conv2D(
            32, (8, 8), strides=(4, 4), activation='relu', 
            name='conv1', kernel_initializer=kernel_init
        )(normed_frames)
        conv_2 = layers.Conv2D(
            64, (4, 4), strides=(2, 2), activation='relu', 
            name='conv2', kernel_initializer=kernel_init
        )(conv_1)
        conv_3 = layers.Conv2D(
            64, (3, 3), strides=(1, 1), activation='relu', 
            name='conv3', kernel_initializer=kernel_init
        )(conv_2)

        # FC layers for both value and advantage streams
        layer_dense = NoisyDense if noisy_net else layers.Dense
        conv_flat = layers.Flatten()(conv_3)
        hidden_val = layer_dense(
            512, activation='relu', name='V_hid', kernel_initializer=kernel_init
        )(conv_flat)
        hidden_adv = layer_dense(
            512, activation='relu', name='A_hid', kernel_initializer=kernel_init
        )(conv_flat)
    else:
        # Conv layers
        conv_1 = layers.Conv2D(
            16, (8, 8), strides=(4, 4), activation='relu', 
            name='conv1', kernel_initializer=kernel_init
        )(normed_frames)
        conv_2 = layers.Conv2D(
            32, (4, 4), strides=(2, 2), activation='relu', 
            name='conv2', kernel_initializer=kernel_init
        )(conv_1)

        # FC layers for both value and advantage streams
        layer_dense = NoisyDense if noisy_net else layers.Dense
        conv_flat = layers.Flatten()(conv_2)
        hidden_val = layer_dense(
            256, activation='relu', name='V_hid', kernel_initializer=kernel_init
        )(conv_flat)
        hidden_adv = layer_dense(
            256, activation='relu', name='A_hid', kernel_initializer=kernel_init
        )(conv_flat)

    # Output layers for value and advantage streams
    # Advantage layer is masked so only selected actions have non-0 value
    out_val_flat = layer_dense(N_atoms, name='V_flat', kernel_initializer=kernel_init)(hidden_val)
    out_val = layers.Reshape((1, N_atoms), name='V')(out_val_flat)
    out_adv_flat = layer_dense(N_actions * N_atoms, name='A_flat', kernel_initializer=kernel_init)(hidden_adv)
    out_adv = layers.Reshape((N_actions, N_atoms), name='A')(out_adv_flat)
    
    # Adjust advantage output
    out_adv_avg = layers.Lambda(
        lambda x: reduce_mean(x, axis=1, keepdims=True), name='A_avg'
    )(out_adv) 
    out_adv_adj = layers.Subtract(name='A_adj')([out_adv, out_adv_avg])

    # Combine both outputs to estimate p_Z values
    out_score = layers.Add(name=f'score')([out_val, out_adv_adj])
    out_pZ = layers.Softmax(name=f'p_Z', axis=2)(out_score)

    # Configure model
    model = keras.Model(inputs=in_frames, outputs=out_pZ)
    optimizer = RMSprop(learning_rate=lr, rho=0.95, epsilon=0.01)
    model.compile(optimizer=optimizer, loss=loss)
    return model


def update_noisy_layers(model: tf.keras.Model) -> None:
    """Reset noisy layer parameters for a tf model.

    TODO: tf.function?
    """
    for l in model.layers:
        if l.__class__.__name__ == 'NoisyDense':
            l.reset_noise()


def fit_batch_DDQNn_PER(
        model: tf.keras.Model, model_tgt: tf.keras.Model, action_space: dict, 
        gamma: float, state_now: np.array, action: np.array, rewards: np.array, 
        game_over: np.array, state_next: np.array, w_imps: np.array, 
        noisy_net: bool = False, double_learn: bool = False
    ) -> Tuple[np.array, float]:
    """Q-learning update on a mini batch of transitions.
    
    Handles DQN variants:
        1. DQN: Deep Q-learning
        2. DDQN: Double DQN
        3. PER: Prioritized experience replay 
        4. Noisy Net layers TODO: reconfirm this is working as should
        5. N-step returns
        6. Dueling network

    Assumptions: 
        1. state(t) and state(t+n) do not cross a life lost or game-over

    Args:
        model: Online model, to train.
        model_tgt: Target model.
        action_space: Action space map.
        gamma: Reward discount rate. 
        state_now: Batch of current states (t).
        action: Batch of actions taken at t.
        rewards: Batch of R[t+1:t+n] for computing n-step return.
        game_over: Batch of terminal state indicators.
        state_next: Batch of next states (t+n).
        w_imps: Importance sampling weights.
        noisy_net: Indicator whether model is using noisy layers.
        double_learn: Toogle whether to apply "Double Learning" algo.

    Returns:
        - Array of mini batch TD-errors.
        - Composite loss.
    """
    # Reshape variables
    action_idx = np.array([action_space.index(a) for a in action], dtype='int32')  # Convert to indexed actions
    state_now = np.stack(state_now)
    state_next = np.stack(state_next)
    rewards = np.stack(rewards)
    game_over = game_over.tolist()
    batch_sz = state_now.shape[0]

    # Update noisy layers params
    if noisy_net:
        update_noisy_layers(model)
        update_noisy_layers(model_tgt)

    # Predict Q values of next states, from target model
    Q_next_tgt = model_call(model_tgt, state_next).numpy()
    Q_next_tgt[game_over] = 0  # Q value of terminal state is 0 by definition

    if double_learn:
        # 1. Get actions that predict highest Q values for next states, from online model
        # 2. Obtain Q values of max action, from the target model
        Q_next_onl = model_call(model, state_next).numpy()
        Q_next_onl[game_over] = 0  # Q value of terminal state is 0 by definition
        Q_next_tgt_max = Q_next_tgt[range(batch_sz), Q_next_onl.argmax(axis=1), None]  # Dims (batch_sz, 1)
    else:
        Q_next_tgt_max = Q_next_tgt.max(axis=1, keepdims=True)
    
    # Set Q target. We only want to run SGD updates for those actions taken
    n = rewards.shape[1]
    R_n = (gamma ** np.arange(n) * rewards).sum(axis=1, keepdims=True)  # n-step forward return
    Q_now_tgt = np.float32(R_n + (gamma ** n) * Q_next_tgt_max)  # Dims (batch_sz, 1)

    # Run SGD update and compute TD-error
    Q_now_pred, loss = train_on_batch(model, state_now, Q_now_tgt, action_idx, sample_weight=w_imps)
    Q_now_pred = Q_now_pred.numpy()
    td_err = abs_td_error(Q_now_pred, Q_now_tgt)
    return td_err, loss


def fit_batch_DDQNn_PER_DS(
        model: tf.keras.Model, model_tgt: tf.keras.Model, action_space: dict, gamma: float, 
        state_now: np.array, action: np.array, rewards: np.array, game_over: np.array, 
        state_next: np.array, w_imps: np.array, Z: np.array, Z_repN: np.array, dZ: float, V: tuple, 
        noisy_net: bool = False, double_learn: bool = False
    ) -> Tuple[np.array, float]:
    """Q-learning update on a mini batch of transitions.
    
    Handles DQN variants:
        1. Distributional Network

    Args:
        model: Online model, to train.
        model_tgt: Target model.
        action_space: Action space map.
        gamma: Reward discount rate. 
        state_now: Batch of current states (t).
        action: Batch of actions taken at t.
        rewards: Batch of R[t+1:t+n] for computing n-step return.
        game_over: Batch of terminal state indicators.
        state_next: Batch of next states (t+n).
        w_imps: Importance sampling weights.
        Z: Domain of categorical distr.
        Z_repN: Utility matrix of Z repeated (#Z) times horizontally.
        dZ: Increment of Z values.
        V: V_min, V_max.
        noisy_net: Indicator whether model is using noisy layers.
        double_learn: Toogle whether to apply "Double Learning" algo.

    Assumptions: 
        1. state(t) and state(t+n) do not cross a life lost or game-over
    
    Returns:
        - Array of mini batch TD-errors.
        - Composite loss.
    """
    # Reshape variables
    action_idx = np.array([action_space.index(a) for a in action], dtype='int32')  # Convert to indexed actions
    state_now = np.stack(state_now)
    state_next = np.stack(state_next)
    rewards = np.stack(rewards)
    game_over = game_over.astype('bool')
    batch_sz = state_now.shape[0]
    N = Z.shape[0]

    # Update noisy layers params
    if noisy_net:
        update_noisy_layers(model)
        update_noisy_layers(model_tgt)

    # Predict Q values of next states, from target model
    p_next_tgt = model_call(model_tgt, state_next).numpy()

    if double_learn:
        p_next_onl = model_call(model, state_next).numpy()
        Q_next_onl = Q_from_Z_distr(Z, p_next_onl)
        Q_next_onl[game_over] = 0
        p_next_tgt_max = p_next_tgt[range(batch_sz), Q_next_onl.argmax(axis=1)]
    else:
        Q_next_tgt = Q_from_Z_distr(Z, p_next_tgt)
        Q_next_tgt[game_over] = 0
        p_next_tgt_max = p_next_tgt[range(batch_sz), Q_next_tgt.argmax(axis=1)]

    n = rewards.shape[1]
    R_n = (gamma ** np.arange(n) * rewards).sum(axis=1)[:, None]  # n-step forward return
    TZ = R_n + (gamma ** n) * Z * ~game_over[:, None]
    TZ = np.clip(TZ, *V)
    w = np.clip(1 - np.abs(np.repeat([TZ.T], N, axis=0).T - Z_repN) / dZ, 0, None)
    p_now_tgt = np.float32((p_next_tgt_max[..., None] * w).sum(axis=1))

    # Run SGD update
    p_now_pred, loss = train_on_batch(
        model, state_now, p_now_tgt, action_idx, sample_weight=w_imps
    )
    td_err = abs_td_error(p_now_pred, p_now_tgt, distr_net=True, Z=Z)
    return td_err, loss
