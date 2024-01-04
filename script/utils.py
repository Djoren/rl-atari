import matplotlib.pyplot as plt
import numpy as np
import random
import imageio
from skimage.color import rgb2gray
from skimage.transform import resize
from tensorflow import image as tf_image
from tensorflow.keras import backend as K
import sys
sys.path.append('../script')
from atari_model import model_call


def downsample(frame):
    return frame[::2, ::2]


def preprocess_frame_v1(frame):
    return downsample(rgb2gray(frame) * 255).astype(np.uint8)


def preprocess_frame_v2(frame):
    """Note for Breakout pixel disappears in starting frames."""
    return np.uint8(resize(
        rgb2gray(frame), (105, 80), preserve_range=False, order=0, 
        mode='constant', anti_aliasing=False) * 255
    )


def preprocess_frame_v3(frame):
    frame = tf_image.rgb_to_grayscale(frame)
    # frame = tf_image.crop_to_bounding_box(frame, 34, 0, 160, 160)
    frame = tf_image.resize(frame, (105, 80), method=tf_image.ResizeMethod.NEAREST_NEIGHBOR)
    return frame


def clip_reward(reward):
    # IS THIS CORRECT? THIS DOESN'T SEEM TO CLIP
    return np.sign(reward)


def get_lin_anneal_eps(i, eps_init, eps_final, n_final):
    """Linearly anneals epsilon across n_final steps.
    NOTE: this can be refactored as a generator.
    """
    eps_decay = (eps_init - eps_final) / n_final
    return max(eps_init - i * eps_decay, eps_final)


def sample_ran_action(action_space):
    return np.random.choice(action_space)


def choose_action(action_space, model, state, eps, return_isran=False):
    """Eps-greedy policy."""
    if random.random() < eps:
        a = sample_ran_action(action_space)
        return (a, True) if return_isran else a
    else:
        action_all = np.ones_like(action_space)  # all actions OHE encoded
        # Q_val = model([state[None, :], action_all[None, :]]).numpy()
        Q_val = model_call(model, [state[None, :], action_all[None, :]]).numpy()
        a = action_space[Q_val.argmax()]
        return (a, False) if return_isran else a


def plot_state(state):
    ax = plt.subplots(1, 4, figsize=(14, 5), sharex=True, sharey=True)[1].flat
    for i in range(4):
        ax[i].imshow(state[:,:,i], cmap='gray', vmin=0, vmax=255)
        for j in range(10, 110, 10):
            ax[i].axhline(j, color='lightblue', lw=0.5)
        for j in range(10, 80, 10):
            ax[i].axvline(j, color='lightblue', lw=0.5)


def frames_to_mp4(opath, frames):
    imageio.mimsave(opath, frames, fps=30, macro_block_size=None)


def play_episode(model, env, action_space, state_len):
    # Start a new game (episode)
    init_frame = env.reset()
    new_life = True
    game_over = False
    epsilon = 0.001
    
    # Keep track of episode figures
    episode_reward = 0
    episode_frames = [init_frame]
    
    # Play episode until game over
    while not game_over:
        if new_life:
            # Start a new life in the game
            frame, _, _, info = env.step(1)  # Fire to start playing
            episode_frames.append(frame)
#             for _ in range(random.randint(1, 5)):
#                 frame, _, game_over, info = env.step(env.action_space.sample())
#                 episode_frames.append(frame)
            
            lives = info['ale.lives']
            frame = preprocess_frame_v1(frame)
            state = np.stack(state_len * [frame], axis=2)
        else:
            state = np.append(state[:, :, 1:], frame[:, :, None], axis=2)
        
        action = choose_action(action_space, model, state, epsilon)
        frame, reward, game_over, info = env.step(action)
        episode_frames.append(frame)
        
        # Process env outputs
        frame = preprocess_frame_v1(frame)
        new_life = info['ale.lives'] < lives 
        lives = info['ale.lives']
         
        # Increase 
        episode_reward += reward
    
    return episode_frames, episode_reward 