from datetime import datetime
import random
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
import gym

from utils import preprocess_frame_v1, preprocess_frame_v2, preprocess_frame_v3, \
    choose_action, clip_reward, get_lin_anneal_eps
from atari_model import atari_model, fit_batch
from replay_memory import ReplayMemory


# tf.compat.v1.disable_eager_execution()

import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

tf.config.threading.set_intra_op_parallelism_threads(2)  
tf.config.threading.set_inter_op_parallelism_threads(16)

os.environ["KMP_BLOCKTIME"] = "1"                                  # Seems to reduce time
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
os.environ["KMP_SETTINGS"] = "0"
os.environ["OMP_NUM_THREADS"] = "24" #  MKL-DNN equivalent of intra_op_parallelism_threads


# Set parameters
total_train_len = 15    # Total no. of episodes to train over
max_episode_len = None     # Max no. of frames agent is allowed to see per episode
state_len = 4              # No. of stacked frames that comprise a state
train_interval = 1         # Every four actions a gradient descend step is performed
burnin_sz = 250          # Replay mem. burn-in: random policy is run for this many frames, training starts after
replay_mem_sz = 1000000    # Max no. of frames cached in replay memory
batch_sz = 32              # No. of training cases (sample from replay mem.) for each SGD update
disc_rate = 0.99           # Q-learning discount factor (gamma)
seed = 1234


# Initialize Atari environment
env = gym.make('BreakoutDeterministic-v4')

# Set seeds
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
env.seed(seed)
env.action_space.seed(seed)

# Initialize new Q-net model and replay memory
model = atari_model(env.action_space.n)
replay_mem = ReplayMemory(replay_mem_sz, state_len)
frame_num = 0


from datetime import datetime
start_time = datetime.now()

# Start the fun
for episode_num in range(total_train_len):
    # Start a new game (episode)
    init_frame = env.reset()
    new_life = True
    game_over = False
    
    # Keep track of episode's total reward
    cum_reward = 0
    
    # Play episode until game over
    while not game_over:
        if new_life:
            # Start a new life in the game
            frame, _, game_over, info = env.step(1)  # Fire to start playing
            lives = info['ale.lives']
            frame = preprocess_frame_v1(frame)
            state = np.stack(state_len * [frame], axis=2)
        else:
            state = np.append(state[:, :, 1:], frame[:, :, None], axis=2)
        
        # Get action
        burnin_done = frame_num > burnin_sz
        action = choose_action(env, model, state, 0)

        # Take action
        frame, reward, game_over, info = env.step(action)
        cum_reward += reward
        
        # Process env outputs
        frame = preprocess_frame_v1(frame)
        reward = clip_reward(reward)
        new_life = info['ale.lives'] < lives 
        lives = info['ale.lives']
        
        # Add new transition to replay memory
        transition = (action, reward, game_over, new_life, frame)
        replay_mem.store_memory(transition)
                                
        # After burn-in period, train every `train_interval`
        if burnin_done and frame_num % train_interval == 0:
            mini_batch = replay_mem.get_sample(batch_sz)
            fit_batch(model, disc_rate, *mini_batch)
         
        # Increase frame it
        frame_num += 1
    
    if episode_num % 1 == 0:
         print(datetime.now(), episode_num, frame_num, cum_reward)


elapsed_time = datetime.now() - start_time
print(elapsed_time)