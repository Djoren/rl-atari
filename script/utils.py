import pandas as pd
import numpy as np
import random
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.animation as ani

import imageio
from skimage.transform import resize
from tensorflow import image as tf_image
import cv2

import tensorflow as tf
from tensorflow.keras import backend as K

from tf_keras_vis.saliency import Saliency
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.utils.scores import CategoricalScore, InactiveScore

import sys
sys.path.append('../script')
from atari_model import model_call


FRAME_CROP_SETTINGS = {
    'space_invaders': (slice(10, 200), slice(15, 145))
}


def frame_max_pooling(frames):
    """Take maximum value for each pixel value across several frames.
    This is needed to remove flickering in some games where some objects appear in and out in even/odd frames.
    """
    return np.max(frames, axis=0)


def downsample(frame):
    return frame[::2, ::2]


def preprocess_frame_v1(frame):
    """
    Note:
        - for Space Invaders has issues.
    """
    return downsample(rgb2gray(frame) * 255).astype(np.uint8)


def preprocess_frame_v2(frame):
    """
    Note:
        - for Breakout pixel disappears in starting frames.
        - for Space Invaders has issues.
    """
    return np.uint8(resize(
        rgb2gray(frame), (105, 80), preserve_range=False, order=0, 
        mode='constant', anti_aliasing=False) * 255
    )


def preprocess_frame_v3(frame):
    """
    Note:
        - for Space Invaders has issues.
    """
    frame = tf_image.rgb_to_grayscale(frame)
    # frame = tf_image.crop_to_bounding_box(frame, 34, 0, 160, 160)
    frame = tf_image.resize(frame, (105, 80), method=tf_image.ResizeMethod.NEAREST_NEIGHBOR)
    return frame


def preprocess_frame_v4(frame, resize_div=2, crop=None):
    """Preprocess a single frame to format agent will ingest.

    TODO: can also try AREA, supposed to work well
    """
    if crop:
        frame = frame[crop[0], crop[1]]

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Faster than skimage
    new_shape = tuple(d // resize_div for d in frame.shape)[::-1]
    frame = cv2.resize(frame, new_shape, interpolation=cv2.INTER_LINEAR)
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


def choose_action(action_space, model, state, eps, ret_stats=False):
    """Eps-greedy policy.
    
    Assumptions:
        1. Only a single action is requested, by passing just a single state.
    """
    if random.random() < eps:
        a = sample_ran_action(action_space)
        Q = np.array(len(action_space) * [np.nan]) 
        return (a, True, Q) if ret_stats else a
    else:
        action_all = np.ones_like(action_space)  # all actions OHE encoded
        Q = model_call(model, [state[None, :], action_all[None, :]]).numpy()[0]
        a = action_space[Q.argmax()]
        return (a, False, Q) if ret_stats else a


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


class EpisodeLogger:
    def __init__(self, fname):
        self.fname = f'{fname}.csv'
        self.cols = [
            'ts', 'episode', 'train_cnt', 'frame_num', 'score', 
            'action_perc', 'action_ran_perc', 'mean_max_Q'
        ]
        pd.DataFrame(columns=self.cols).to_csv(self.fname, index=False, header=True)

    def append(
            self, episode, train_cnt, frame_num, reward, actions, 
            a_israns, Qs
        ):
        values = [
            datetime.utcnow(),
            episode,
            train_cnt,
            frame_num,
            reward,
            pd.value_counts(actions, normalize=True).round(3).sort_index().to_dict(),
            np.mean(a_israns),
            np.nanmean(np.max(Qs, axis=1))
        ]
        pd.DataFrame([values], columns=self.cols, index=[0]).to_csv(self.fname, index=False, header=False, mode='a')


def run_saliency_map(
        model, states, actions, action_space, alpha=2, dueling=False, 
        sal_type='sal', sal_kwargs=None
    ):
    """Computes saliency maps for a sequence of states.
    
    Alpha: attenuates the saliency map overlay exponentially.
    Dueling: if using dueling model V and A stream saliency will be plotted independently.
    """
    if sal_kwargs is None: sal_kwargs = {}
    action_space = np.array(action_space).tolist()
    
    if sal_type == 'sal':
        sal_obj = Saliency
    elif sal_type == 'gcam':
        sal_obj = Gradcam
    elif sal_type == 'gcampp':
        sal_obj = GradcamPlusPlus
    elif sal_type == 'scam':
        sal_obj = Scorecam
    
    if dueling:
        # Clone model and omit all layers after V and A_adj
        model_clone = tf.keras.Model(
            inputs = [model.get_layer('input_frames').input], 
            outputs = [model.get_layer('V').output, model.get_layer('A_adj').output]
        )
        score_v = [CategoricalScore([0]), InactiveScore()]

        sal_list = []
        for i, s in enumerate(states):
            s_input = s[None, :].astype('float32')
            max_a = action_space.index(actions[i])
            score_a = [InactiveScore(), CategoricalScore([max_a])]
            sal_v = sal_obj(model_clone)(score_v, s_input, **sal_kwargs)
            sal_a = sal_obj(model_clone)(score_a, s_input, **sal_kwargs)

            # Combine maps to RGB image. Display only last frame from each state
            # Action-sal is red, frame is gree, Value-sal is blue
            sal = np.stack([sal_a[0] ** alpha] + [s[...,-1] / 255.0] + [sal_v[0] ** alpha], axis=2)
            sal_list.append(sal)
    else:
        model_clone = tf.keras.Model(
            inputs = [model.get_layer('input_frames').input], 
            outputs = [model.get_layer('Q').output]
        )

        sal_list = []
        for i, s in enumerate(states):
            s_input = s[None, :].astype('float32')
            max_a = action_space.index(actions[i])
            score = [InactiveScore(), CategoricalScore([max_a])]
            sal = sal_obj(model_clone)(score, s_input, **sal_kwargs)

            # Combine maps to RGB image. Display only last frame from each state
            # Action-sal is red, frame is gree, Value-sal is blue
            sal = np.stack([sal[0] ** alpha] + [s[...,-1] / 255.0] + [s[...,-1] / 255.0], axis=2)
            sal_list.append(sal)
    
    return sal_list


def animate_episode(frames, frames_pp, Qs, action_labels, opath):
    """Outputs mp4 animation of a set (e.g. an episode) of frames.

    Per frame it displays:
        1. Raw game frame
        2. Preprocessed game frame (as agent sees it)
        3. Full series of max Q(a)
        4. Distribution of Q(a)
    """
    fig = plt.Figure(figsize=(9, 6))  # Somehow much faster than 'plt.figure'

    for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
        plt.rcParams[param] = '0.9'

    for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
        plt.rcParams[param] = '#111625'

    # Initial image of raw frames
    img_x0 = 0.11
    img_w = 0.38
    d = 0.0
    ax1 = fig.add_axes([img_x0, 0.24, img_w, 0.76])
    img = ax1.imshow(frames[0], animated=True)
    ax1.set_xlabel(None)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Initial image of preprocessed frames
    ax4 = fig.add_axes([img_x0 + img_w + d, 0.24, img_w, 0.76])
    img2 = ax4.imshow(frames_pp[0], animated=True)
    ax4.set_xlabel(None)
    ax4.set_xticks([])
    ax4.set_yticks([])

    # Initial line plot of max-Q series
    ax2 = fig.add_axes([0.11, 0.145, .76, 0.105])
    max_q_series = [np.max(q) for q in Qs]
    ax2.plot(max_q_series, color='#FE53BB', lw=.5, animated=True)
    ax2.axhline(0, ls=':', lw=0.5, color='grey', alpha=.5, animated=True)
    span = ax2.axvspan(0, 0, facecolor='#08F7FE', alpha=.1, animated=True)
    vline = ax2.axvline(0, color='#08F7FE', lw=.5, animated=True)
    q_txt = ax2.text(.96, .90, 0, ha='right', va='top', transform=ax2.transAxes, fontsize=6, color='#FE53BB')
    ax2.set_title('Max Q-value', x=0.5, y=0.70, fontsize=6, color='#00ff41')
    ax2.set_xticks([])
    ax2.tick_params(labelsize=5)

    # Initial bar plot episode-wise Q(a) distribution
    ax3 = fig.add_axes([0.11, 0.04, .76, 0.105])
    p_prior = 1 / len(action_labels)
    init_qs = np.array(len(action_labels) * [1])
    bar = ax3.bar(action_labels, init_qs / np.sum(init_qs), color='#08F7FE', animated=True)
    ax3.axhline(p_prior, ls=':', lw=0.5, color='grey', alpha=.5, animated=True)
    ax3.set_title('Q-values', x=0.5, y=0.70, fontsize=6, color='#00ff41')
    ax3.set_ylim(.50 * p_prior, 2 * p_prior)
    ax3.tick_params(labelsize=5)

    for loc in ['top', 'bottom', 'left', 'right']:
        col = '#323e6a' # #00ff41
        ax1.spines[loc].set_color(col)
        ax2.spines[loc].set_color(col)
        ax3.spines[loc].set_color(col)
        ax4.spines[loc].set_color(col)

    # Animate for each frame
    Qs_norm = [q / q.sum() for q in Qs]

    def animate(i):
        img.set_array(frames[i])
        img2.set_array(frames_pp[i])
        vline.set_xdata(i)
        span.set_xy([[0, 0], [0, 1], [i, 1], [i, 0]])
        for j, b in enumerate(bar):
            b.set_height(Qs_norm[i][j])
        
        bar[animate.a_max].set_color('#08F7FE')  # Set old max to def color
        animate.a_max = np.argmax(Qs_norm[i])
        bar[animate.a_max].set_color('#024A4C')  # Set new max to max color
        q_txt.set_text(Qs[i].max().round(1))

        return [img, vline, span, q_txt] + list(bar)

    animate.a_max = 0  # Add function attribute
    an = ani.FuncAnimation(fig, animate, frames=len(frames), blit=False) # Blit seems to make it slower
    writer = ani.FFMpegWriter(fps=30)
    an.save(opath, writer=writer, dpi=200)
    plt.close()


def animate_episode_sal(
        model, episode_states, episode_frames, episode_actions, 
        episode_Qs, action_space, opath, dueling=False
    ):
    """Animates an episode with saliency map overlayed on preprocessed frames.
    """
    sals = run_saliency_map(model, episode_states, episode_actions, action_space, dueling=dueling)
    animate_episode(episode_frames, sals, episode_Qs, action_space, opath)