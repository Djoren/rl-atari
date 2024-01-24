import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow_addons import layers as layers_tfa
from noisy_dense import NoisyDense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import Huber, CategoricalCrossentropy
from tensorflow.math import reduce_mean


# TODO: pass in only tensors etc. To avoid retracing.
@tf.function  # Somehow this causes y_pred to have slightly different decimals
def train_on_batch(x, y_tgt, model, sample_weight=None):
    """Custom fit function such that we can obtain td-error, w/o having to recompute it.
    
    Assumptions:
        1. The sum in td_error assumes y_tgt == y_pred for all but one action per data point.
        2. Model has been passed an optimizer object as argument, not a string.

    Notes:
        - Unsure if td_error calc should be within context-mngr. runtime testing wasn't conclusive.
        - To calc loss for each data point TF takes sum across multiple outputs (in a single output layer).
        - TF then by default takes mean across losses in batch (across all data points and other remaining dimensions).
        - We might not want this for our RL purposes, as our intention is to train for only a single action per sample.
          Thus might make more sense to set "SUM" as `reduction` parameter in the loss object.
        - This should not affect direction of gradients, but only the magnitude => has interplay with learning-rate.
    """
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss_mean = model.loss(y_tgt, y_pred, sample_weight=sample_weight)
        
    grads = tape.gradient(loss_mean, model.trainable_weights)
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return y_pred


# TODO: This seems like it's faster, but unsure if there's risk of issues
@tf.function
def model_call(model, inputs):
    return model(inputs)


def abs_td_error(y_pred, y_tgt, distr=False, Z=None):
    """Compute TD-error for predicted Q vs target Q.
    
    Used for setting priorities in Prioritized Experience Replay.
    TODO: Z should be class attribute when we OOP all this code.
    """
    # If distributive target convert p to Q
    if distr:
        y_pred = Q_from_Z_distr(Z, y_pred)
        y_tgt = Q_from_Z_distr(Z, y_tgt)
    return np.sum(np.abs(y_tgt - y_pred), axis=0)


def Q_from_Z_distr(Z, p):
    """Computes the estimate Q_hat from Z distribution."""
    return np.sum(Z * p, axis=-1)


def atari_model(
        n_actions, lr, state_shape, kernel_init='glorot_uniform', noisy_net=False, 
        large_net=False
    ): 
    """ n_actions: OHE matrix of actions.
        
    Notes: 
        - that the 4th action (FIRE) starts the game for some games.
    """
    # Input layers
    input_actions = layers.Input((n_actions,), name='input_actions')
    input_frames = layers.Input(state_shape, name='input_frames', dtype=tf.float32)
    normed_frames = layers.Lambda(lambda x: tf.cast(x, tf.float32) / 255.0)(input_frames)  # Convert frames uint8[0, 255] to float[0, 1]
    # TODO: more efficient to leave out the 255 at frames processing, and here as well?

    if large_net:
        # Convolutional layers
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

        # Fully connected layers
        layer_dense = NoisyDense if noisy_net else layers.Dense
        conv_flat = layers.Flatten()(conv_3)
        hidden = layer_dense(
            512, activation='relu', name='hid', kernel_initializer=kernel_init
        )(conv_flat)
    else:
        # Convolutional layers
        conv_1 = layers.Conv2D(
            16, (8, 8), strides=(4, 4), activation='relu', 
            name='conv1', kernel_initializer=kernel_init
        )(normed_frames)
        conv_2 = layers.Conv2D(
            32, (4, 4), strides=(2, 2), activation='relu', 
            name='conv2', kernel_initializer=kernel_init
        )(conv_1)

        # Fully connected layers
        # layer_dense = layers_tfa.NoisyDense if noisy_net else layers.Dense
        # layer_dense = layers.Dense
        layer_dense = NoisyDense if noisy_net else layers.Dense
        conv_flat = layers.Flatten()(conv_2)
        hidden = layer_dense(
            256, activation='relu', name='hid', kernel_initializer=kernel_init
        )(conv_flat)

    # Output layer. Q values are masked by actions so only selected actions have non-0 Q value
    output = layer_dense(n_actions, name='Q', kernel_initializer=kernel_init)(hidden)
    output_masked = layers.Multiply(name='Q_masked')([output, input_actions])

    # Model
    model = keras.Model(inputs=[input_frames, input_actions], outputs=output_masked)
    optimizer = RMSprop(learning_rate=lr, rho=0.95, epsilon=0.01)
    model.compile(optimizer=optimizer, loss=Huber())
    return model


def atari_model_dueling(
        n_actions, lr, state_shape, kernel_init='glorot_uniform', noisy_net=False, 
        large_net=False    
    ): 
    """ n_actions: OHE matrix of actions.
        Note that the 4th action (FIRE) starts the game.
    """
    # Input layers
    in_actions = layers.Input((n_actions,), name='input_actions')
    in_frames = layers.Input(state_shape, name='input_frames', dtype=tf.float32)
    normed_frames = layers.Lambda(lambda x: tf.cast(x, tf.float32) / 255.0)(in_frames)  # Convert frames uint8[0, 255] to float[0, 1]
    # TODO: more efficient to leave out the 255 at frames processing, and here as well?

    if large_net:
        # Convolutional layers
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

        # Fully connected layers for both value and advantage streams
        layer_dense = NoisyDense if noisy_net else layers.Dense
        conv_flat = layers.Flatten()(conv_3)
        hidden_val = layer_dense(
            512, activation='relu', name='V_hid', kernel_initializer=kernel_init
        )(conv_flat)
        hidden_adv = layer_dense(
            512, activation='relu', name='A_hid', kernel_initializer=kernel_init
        )(conv_flat)
    else:
        # Convolutional layers
        conv_1 = layers.Conv2D(
            16, (8, 8), strides=(4, 4), activation='relu', 
            name='conv1', kernel_initializer=kernel_init
        )(normed_frames)
        conv_2 = layers.Conv2D(
            32, (4, 4), strides=(2, 2), activation='relu', 
            name='conv2', kernel_initializer=kernel_init
        )(conv_1)

        # Fully connected layers for both value and advantage streams
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
    out_adv = layer_dense(n_actions, name='A', kernel_initializer=kernel_init)(hidden_adv)

    # Adjust advantage output
    out_adv_avg = layers.Lambda(
        lambda x: reduce_mean(x, axis=1, keepdims=True), name='A_avg'
    )(out_adv)
    out_adv_adj = layers.Subtract(name='A_adj')([out_adv, out_adv_avg])

    # Combine both outputs to estimate Q values
    out = layers.Add(name='Q')([out_val, out_adv_adj])
    out_masked = layers.Multiply(name='Q_masked')([out, in_actions])

    # Model
    model = keras.Model(inputs=[in_frames, in_actions], outputs=out_masked)
    optimizer = RMSprop(learning_rate=lr, rho=0.95, epsilon=0.01)
    model.compile(optimizer=optimizer, loss=Huber())
    return model


def atari_model_distr(N, n_actions, lr, kernel_init='glorot_uniform', noisy_net=False): 
    # Input layers
    input_frames = layers.Input((105, 80, 4), name='input_frames', dtype=tf.float32)
    normed_frames = layers.Lambda(lambda x: tf.cast(x, tf.float32) / 255.0)(input_frames)  # Convert frames uint8[0, 255] to float[0, 1]

    # Convolutional layers
    conv_1 = layers.Conv2D(
        16, (8, 8), strides=(4, 4), activation='relu', name='conv1', kernel_initializer=kernel_init
    )(normed_frames)
    conv_2 = layers.Conv2D(
        32, (4, 4), strides=(2, 2), activation='relu', name='conv2', kernel_initializer=kernel_init
    )(conv_1)

    # Fully connected layers
    # layer_dense = layers_tfa.NoisyDense if noisy_net else layers.Dense
    # layer_dense = layers.Dense
    layer_dense = NoisyDense if noisy_net else layers.Dense
    conv_flat = layers.Flatten()(conv_2)
    hidden = layer_dense(
        256, activation='relu', name='hid', kernel_initializer=kernel_init
    )(conv_flat)

    # Output layers Z(a) for every action
    outputs = []
    for i in range(n_actions):
        output_a = layer_dense(N, activation='softmax', name=f'Z_{i}', kernel_initializer=kernel_init)(hidden)
        output_a = layers.Reshape((1, N))(output_a)
        outputs.append(output_a)
    output = layers.Concatenate(name='Z_concat', axis=1)(outputs)

    # Model
    model = keras.Model(inputs=[input_frames], outputs=output)
    optimizer = RMSprop(learning_rate=lr, rho=0.95, epsilon=0.01)
    model.compile(optimizer=optimizer, loss=CategoricalCrossentropy())

    return model


def compute_distr_target(model, states, Z, Z_rep, dZ, V_min, V_max, N, rewards, gamma):
    """Computes the new target Z-distribution probabilities."""
    p = model(states).numpy()  # Predicted Z-distr. probabilities for model inputs
    Q = Q_from_Z_distr(Z, p)  # Get pred Q from Z-distr.
    a_max = Q.argmax(axis=1)  
    p_max = p[range(p.shape[0]), a_max]  # Subset of p matrix: per data point only the action that maxes Q(a)

    # Compute distributional bellman update -> TZ
    TZ = rewards + gamma * Z  # Applying distr. belman operator T^pi
    TZ = np.clip(TZ, V_min, V_max)  # Bound update to support of Z, [V_min, V_max]

    # Project bellman update onto support of TZ -> Phi TZ
    TZ_rep = np.repeat([TZ], N, axis=0).T
    w_project = np.clip(1 - np.abs(TZ_rep - Z_rep) / dZ, 0, None)  # Projection weights to distribute TZ probs to Z
    
    # Assign new weighted (target) probs to p matrix
    # Keeping probs of the non-selected (non-Q-maxing) actions fixed 
    p[range(p.shape[0]), a_max] = np.dot(p_max, w_project)
    
    return p


def fit_batch(model, action_space, gamma, state_now, action, reward, game_over, state_next):
    """Do Q-learning update on a batch of transitions."""
    # Reshape variables
    action_idx = [action_space.index(a) for a in action]  # Convert to indexed actions
    action_ohe = np.eye(model.output.shape[1])[action_idx]  # OHE encode actions
    state_now = np.stack(state_now)
    state_next = np.stack(state_next)
    game_over = game_over.tolist()
    
    # Predict Q values of the next states for all actions
    Q_val = model([state_next, np.ones_like(action)]).numpy()
    Q_val[game_over] = 0  # Q value of terminal state is 0 by definition
    
    # Set Q target. We only run SGD updates for those actions taken.
    # Hence, Q target for non-taken actions is set to 0
    Q_tgt = reward + gamma * Q_val.max(axis=1)
    Q_tgt = np.float32(action_ohe * Q_tgt[:, None])
    
    # Run SGD update
    model.fit([state_now, action_ohe], Q_tgt, batch_size=len(Q_tgt), epochs=1, verbose=0)


def fit_batch_DQN(
        model, model_tgt, action_space, gamma, state_now, action, 
        reward, game_over, state_next, custom_fit=False
    ):
    """Q-learning update on a batch of transitions."""
    # Reshape variables
    action_idx = [action_space.index(a) for a in action]  # Convert to indexed actions
    action_ohe = np.eye(model.output.shape[1])[action_idx]  # OHE encode actions
    state_now = np.stack(state_now)
    state_next = np.stack(state_next)
    game_over = game_over.tolist()
    
    # Predict Q values of the next states for all actions
    Q_val = model_tgt([state_next, np.ones_like(action_ohe)]).numpy()
    Q_val[game_over] = 0  # Q value of terminal state is 0 by definition
    
    # Set Q target. We only run SGD updates for those actions taken.
    # Hence, Q target for non-taken actions is set to 0
    Q_tgt = reward + gamma * Q_val.max(axis=1)
    Q_tgt = np.float32(action_ohe * Q_tgt[:, None])

    # # Tensorboard
    # logs = 'logs/' + datetime.now().strftime('%Y%m%d-%H%M%S')
    # tboard_callback = tf.keras.callbacks.TensorBoard(
    #     log_dir=logs, histogram_freq=1, profile_batch='1'
    # )
    
    # Run SGD update
    if custom_fit:
        train_on_batch([state_now, action_ohe], Q_tgt, model)
    else:
        model.train_on_batch([state_now, action_ohe], Q_tgt)


def fit_batch_DDQN(model, model_tgt, action_space, gamma, state_now, action, reward, game_over, state_next):
    """Double Q-learning update on a batch of transitions."""
    # Reshape variables
    action_idx = [action_space.index(a) for a in action]  # Convert to indexed actions
    action_ohe = np.eye(model.output.shape[1])[action_idx]  # OHE encode actions
    state_now = np.stack(state_now)
    state_next = np.stack(state_next)
    game_over = game_over.tolist()

    # Get actions that generate highest Q from online net
    action_ones = np.ones_like(action_ohe)
    Q_val = model([state_next, action_ones]).numpy()
    Q_val[game_over] = 0  # Q value of terminal state is 0 by definition
    Q_max_a_idx = Q_val.argmax(axis=1)
    
    # Predict Q values of the next states for all actions
    # Use target net to get Q values for prev. selected actions
    Q_val = model_tgt([state_next, action_ones]).numpy()
    Q_val[game_over] = 0  # Q value of terminal state is 0 by definition
    Q_val = Q_val[range(Q_val.shape[0]), Q_max_a_idx]
    
    # Set Q target. We only run SGD updates for those actions taken.
    # Hence, Q target for non-taken actions is set to 0
    Q_tgt = reward + gamma * Q_val
    Q_tgt = np.float32(action_ohe * Q_tgt[:, None])
    
    # Run SGD update
    model.train_on_batch([state_now, action_ohe], Q_tgt)


def fit_batch_DQNn(
        model, model_tgt, action_space, gamma, state_now, action, 
        rewards, game_over, state_next, custom_fit=False
    ):
    """Q-learning update on a batch of transitions.

    Rewards: R[t+1:t+n] for computing n-step return
    Assumption: state(t) and state(t+n) do not cross a life lost or game-over
    """
    # Reshape variables
    action_idx = [action_space.index(a) for a in action]  # Convert to indexed actions
    action_ohe = np.eye(model.output.shape[1])[action_idx]  # OHE encode actions
    state_now = np.stack(state_now)
    state_next = np.stack(state_next)
    rewards = np.stack(rewards)
    game_over = game_over.tolist()
    
    # Predict Q values of the next states for all actions
    Q_val = model_tgt([state_next, np.ones_like(action_ohe)]).numpy()
    Q_val[game_over] = 0  # Q value of terminal state is 0 by definition
    
    # Set Q target. We only run SGD updates for those actions taken.
    # Hence, Q target for non-taken actions is set to 0
    n = rewards.shape[1]
    R_n = (gamma ** np.arange(n) * rewards).sum(axis=1)
    Q_tgt = R_n + (gamma ** n) * Q_val.max(axis=1)
    Q_tgt = np.float32(action_ohe * Q_tgt[:, None])

    # Run SGD update (train_on_batch() is faster than fit())
    if custom_fit:
        train_on_batch([state_now, action_ohe], Q_tgt, model)
    else:
        model.train_on_batch([state_now, action_ohe], Q_tgt)


# TODO: tf.function?
def update_noisy_layers(model):
    for l in model.layers:
        if l.__class__.__name__ == 'NoisyDense':
            l.reset_noise()


def fit_batch_DQNn_PER(
        model, model_tgt, action_space, gamma, state_now, action, rewards, 
        game_over, state_next, w_imps, noisy_net=False
    ):
    """Q-learning update on a batch of transitions.

    Rewards: R[t+1:t+n] for computing n-step return
    Assumption: state(t) and state(t+n) do not cross a life lost or game-over
    """
    # Reshape variables
    action_idx = [action_space.index(a) for a in action]  # Convert to indexed actions
    action_ohe = np.eye(model.output.shape[1])[action_idx]  # OHE encode actions
    state_now = np.stack(state_now)
    state_next = np.stack(state_next)
    rewards = np.stack(rewards)
    game_over = game_over.tolist()

    # Update noisy layers params
    if noisy_net:
        update_noisy_layers(model)
        update_noisy_layers(model_tgt)
    
    # Predict Q values of the next states for all actions
    # Q_next_ = model_tgt([state_next, np.ones_like(action_ohe)]).numpy()
    Q_next_ = model_call(model_tgt, [state_next, np.ones_like(action_ohe)]).numpy()
    Q_next_[game_over] = 0  # Q of terminal state is 0 by def.
    
    # Set Q target. We only run SGD updates for those actions taken.
    # Hence, Q target for non-taken actions is set to 0
    n = rewards.shape[1]
    R_n = (gamma ** np.arange(n) * rewards).sum(axis=1)  # n-step forward return
    Q_tgt = R_n + (gamma ** n) * Q_next_.max(axis=1)
    Q_tgt = np.float32(action_ohe * Q_tgt[:, None])

    # Run SGD update
    td_err = train_on_batch([state_now, action_ohe], Q_tgt, model, sample_weight=w_imps)

    return td_err.numpy()


def fit_batch_DQNn_PER_(
        model, model_tgt, action_space, gamma, state_now, action, rewards, 
        game_over, state_next, w_imps
    ):
    """Q-learning update on a batch of transitions.

    Rewards: R[t+1:t+n] for computing n-step return
    Assumption: state(t) and state(t+n) do not cross a life lost or game-over
    """
    # Reshape variables
    action_idx = [action_space.index(a) for a in action]  # Convert to indexed actions
    action_ohe = np.eye(model.output.shape[1])[action_idx]  # OHE encode actions
    state_now = np.stack(state_now)
    state_next = np.stack(state_next)
    rewards = np.stack(rewards)
    game_over = game_over.tolist()

    # Predict Q values of the current states for taken actions
    # NOTE: current states shouldn't contain game-overs/life-losts, per sample logic
    Q_now = model([state_now, action_ohe]).numpy()
    
    # Predict Q values of the next states for all actions
    Q_next_ = model_tgt([state_next, np.ones_like(action_ohe)]).numpy()
    Q_next_[game_over] = 0  # Q of terminal state is 0 by def.
    
    # Set Q target. We only run SGD updates for those actions taken.
    # Hence, Q target for non-taken actions is set to 0
    n = rewards.shape[1]
    R_n = (gamma ** np.arange(n) * rewards).sum(axis=1)  # n-step forward return
    Q_tgt = R_n + (gamma ** n) * Q_next_.max(axis=1)
    Q_tgt = np.float32(action_ohe * Q_tgt[:, None])

    # TD-error
    td_err = np.abs(Q_tgt - Q_now).sum(axis=1)

    # Run SGD update (train_on_batch() is faster than fit())
    model.train_on_batch([state_now, action_ohe], Q_tgt, sample_weight=w_imps)
    return td_err


def fit_batch_DDQNn_PER(
        model, model_tgt, action_space, gamma, state_now, action, rewards, 
        game_over, state_next, w_imps, noisy_net=False, double_learn=False
    ):
    """Q-learning update on a batch of transitions.
    Enables features:
        1. DQN: Deep Q-learning
        2. DDQN: Double DQN
        3. PER: Prioritized experience replay 
        4. Noisy Net layers TODO: reconfirm this is working as should
        5. N-step returns
        6. Distributional Net TODO: implement this

    Rewards: R[t+1:t+n] for computing n-step return
    Assumptions: 
        - state(t) and state(t+n) do not cross a life lost or game-over
    """
    # Reshape variables
    action_idx = [action_space.index(a) for a in action]  # Convert to indexed actions
    action_ohe = np.eye(model.output.shape[1])[action_idx]  # OHE encode actions
    state_now = np.stack(state_now)
    state_next = np.stack(state_next)
    rewards = np.stack(rewards)
    game_over = game_over.tolist()

    # Update noisy layers params
    if noisy_net:
        update_noisy_layers(model)
        update_noisy_layers(model_tgt)

    # Predict Q values of next states, from target model
    action_ones = np.ones_like(action_ohe)
    Q_next_tgt = model_call(model_tgt, [state_next, action_ones]).numpy()
    Q_next_tgt[game_over] = 0  # Q value of terminal state is 0 by definition

    if double_learn:
        # 1. Get actions that predict highest Q values for next states, from online model
        # 2. Obtain Q values of max action, from the target model
        Q_next_onl = model_call(model, [state_next, action_ones]).numpy()
        Q_next_onl[game_over] = 0  # Q value of terminal state is 0 by definition
        Q_next_tgt_max = Q_next_tgt[range(Q_next_tgt.shape[0]), Q_next_onl.argmax(axis=1)]
    else:
        Q_next_tgt_max = Q_next_tgt.max(axis=1)
    
    # Set Q target. We only run SGD updates for those actions taken.
    # Hence, Q target for non-taken actions is set to 0, as preds for those actions are also 0
    n = rewards.shape[1]
    R_n = (gamma ** np.arange(n) * rewards).sum(axis=1)  # n-step forward return
    Q_tgt = R_n + (gamma ** n) * Q_next_tgt_max
    Q_tgt = np.float32(action_ohe * Q_tgt[:, None])

    # Run SGD update
    td_err = train_on_batch([state_now, action_ohe], Q_tgt, model, sample_weight=w_imps)

    return td_err.numpy()



