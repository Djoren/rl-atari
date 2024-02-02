import pandas as pd
import numpy as np
import random
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.animation as ani
from matplotlib.ticker import MultipleLocator

import imageio
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
from atari_model import model_call, Q_from_Z_distr


# Defines settings for croping the frames for each game
# Note that settings are obtained from manual inspection
FRAME_CROP_SETTINGS = {
    'space_invaders': (slice(10, 200), slice(15, 145))
}


def frame_max_pooling(frames):
    """Take maximum value for each pixel value across several frames.

    This is needed to remove flickering in some games where some objects appear in and out in even/odd frames.
    """
    return np.max(frames, axis=0)


def preprocess_frame_v4(frame, crop_lims=None, resize_div=2):
    """Preprocess a single frame to format agent will ingest.

    TODO: can also try AREA, supposed to work well
    """
    if crop_lims:
        frame = frame[crop_lims[0], crop_lims[1]]

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Faster than skimage
    new_shape = tuple(d // resize_div for d in frame.shape)[::-1]
    frame = cv2.resize(frame, new_shape, interpolation=cv2.INTER_LINEAR)
    return frame


def clip_reward(reward):
    """Clips rewards to the values {-1, 1}."""
    return np.sign(reward)


def get_lin_anneal_eps(i, eps_init, eps_final, n_final):
    """Linearly anneals epsilon across n_final steps.
    NOTE: this can be refactored as a generator.
    """
    eps_decay = (eps_init - eps_final) / n_final
    return max(eps_init - i * eps_decay, eps_final)


def sample_ran_action(action_space):
    """Randomly samples an action from action space."""
    return np.random.choice(action_space)


def choose_action(model, state, action_space, eps, duel_net=False, distr_net=False, Z=None):
    """Eps-greedy policy.
    
    Assumptions:
        1. Only a single action is requested, by passing just a single state.
    """
    if random.random() < eps:
        a = sample_ran_action(action_space)

        # Dummy values as they are used downstream
        Q = A = np.array(len(action_space) * [np.nan]) 
        V = np.array([np.nan])
        pZ = Z if Z is None else np.array(len(action_space) * [len(Z) * [np.nan]]) 
        return (a, True, Q, V, A, pZ)
    else:      
        pZ = V = A = None  # `None` as values not used downstream
        if distr_net:
            pZ = model_call(model, state[None, :]).numpy()[0]
            Q = Q_from_Z_distr(Z, pZ)
        elif duel_net:
            V, A = model_call(model, state[None, :])
            V, A = V.numpy()[0], A.numpy()[0]
            Q = V + A
        else:
            Q = model_call(model, state[None, :]).numpy()[0]

        a = action_space[Q.argmax()]
        return (a, False, Q, V, A, pZ)


def plot_state(state):
    ax = plt.subplots(1, 4, figsize=(14, 5), sharex=True, sharey=True)[1].flat
    for i in range(4):
        ax[i].imshow(state[:,:,i], cmap='gray', vmin=0, vmax=255)
        for j in range(10, 110, 10):
            ax[i].axhline(j, color='lightblue', lw=0.5)
        for j in range(10, 80, 10):
            ax[i].axvline(j, color='lightblue', lw=0.5)


def frames_to_mp4(opath, frames):
    """Convert a list of frames (numpy arrays) to mop4 animation."""
    imageio.mimsave(opath, frames, fps=30, macro_block_size=None)


def play_episode(model, env, action_space, state_len):
    # TODO: update this function
    pass


class EpisodeLogger:
    """Logger class to write episode statistics to a csv file."""

    def __init__(self, fname):
        self.fname = f'{fname}.csv'
        self.cols = [
            'ts', 'episode', 'train_cnt', 'frame_num', 'score', 
            'action_perc', 'mean_max_Q', 'mean_loss', 'mean_tderr'
        ]
        pd.DataFrame(columns=self.cols).to_csv(self.fname, index=False, header=True)

    def append(
            self, episode, train_cnt, frame_num, reward, actions, 
            a_israns, Qs, loss, td_err
        ):
        values = [
            datetime.utcnow(),
            episode,
            train_cnt,
            frame_num,
            reward,
            pd.value_counts(actions, normalize=True).round(3).sort_index().to_dict(),
            np.nanmean(np.max(Qs, axis=1)), 
            np.mean(loss),
            np.mean(td_err)
        ]
        pd.DataFrame([values], columns=self.cols, index=[0]).to_csv(self.fname, index=False, header=False, mode='a')


def run_saliency_map(
        model, states, actions, action_space, duel_net=False, distr_net=False,
        sal_type='sal', sal_kwargs=None, alpha=2
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
    
    if duel_net:
        # Clone model and omit all layers after V and A_adj
        model_clone = tf.keras.Model(
            inputs=[model.get_layer('input_frames').input], 
            outputs=[model.get_layer('V').output, model.get_layer('A_adj').output]
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
    elif distr_net:
        model_clone = tf.keras.Model(
            inputs=[model.get_layer('input_frames').input], 
            outputs=[model.get_layer('p_concat').output]
        )

        sal_list = []
        for i, s in enumerate(states):
            s_input = s[None, :].astype('float32')
            max_a = action_space.index(actions[i])
            score = lambda output: output[:, max_a, :] # AFAIK takes mean across probs when computing grads
            sal = sal_obj(model_clone)(score, s_input, **sal_kwargs)

            # Combine maps to RGB image. Display only last frame from each state
            sal = np.stack([sal[0] ** alpha] + [s[...,-1] / 255.0] + [s[...,-1] / 255.0], axis=2)
            sal_list.append(sal)
    else:
        model_clone = tf.keras.Model(
            inputs=[model.get_layer('input_frames').input], 
            outputs=[model.get_layer('Q').output]
        )

        sal_list = []
        for i, s in enumerate(states):
            s_input = s[None, :].astype('float32')
            max_a = action_space.index(actions[i])
            score = [CategoricalScore([max_a])]
            sal = sal_obj(model_clone)(score, s_input, **sal_kwargs)

            # Combine maps to RGB image. Display only last frame from each state
            sal = np.stack([sal[0] ** alpha] + [s[...,-1] / 255.0] + [s[...,-1] / 255.0], axis=2)
            sal_list.append(sal)
    
    return sal_list


def animate_episode(
        frames, frames_pp, Qs, Vs, As, action_labels, duel_net, opath, Z_distr=None
    ):
    fig = plt.Figure(figsize=(9, 6))  # Somehow much faster than 'plt.figure'

    for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
        plt.rcParams[param] = '0.9'

    for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
        plt.rcParams[param] = '#111625'

    distr_net = Z_distr[0] is not None
    
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
    ax4 = fig.add_axes([img_x0 + img_w + d, 0.24, img_w, 0.75])
    img2 = ax4.imshow(frames_pp[0], animated=True)
    ax4.set_xlabel(None)
    ax4.set_xticks([])
    ax4.set_yticks([])

    # Initial line plot of max-Q series
    ax2 = fig.add_axes([0.11, 0.145, .76, 0.105])
    max_Q_series = [np.max(Q) for Q in Qs]
    Q_line, = ax2.plot(max_Q_series, color='#FE53BB', lw=.5, animated=True)
    # ax2.axhline(0, ls=':', lw=0.5, color='grey', alpha=.5, animated=True)
    span = ax2.axvspan(0, 0, facecolor='#08F7FE', alpha=.1, animated=True)
    axvline = ax2.axvline(0, color='#08F7FE', lw=.5, animated=True)
    Q_txt = ax2.text(.98, .90, 0, ha='right', va='top', transform=ax2.transAxes, fontsize=5, color='#FE53BB')
    ax2.set_title('Max Q-value', x=0.5, y=0.70, fontsize=6, color='#00ff41')
    ax2.set_xticks([])
    ax2.tick_params(labelsize=5)

    # Initial duel V-A series
    if duel_net:
        A_series =  [A[np.argmax(Q)] for A, Q in zip(As, Qs)] 
        V_line, = ax2.plot(Vs, color='#08F7FE', lw=.5, animated=True)
        A_line, = ax2.plot(A_series, color='r', lw=.5, animated=True)
        ax2.set_title('Max Q-value, V and A', x=0.5, y=0.70, fontsize=6, color='#00ff41')
        # ax2.legend(
        #     [Q_line, V_line, A_line], ['Q', 'V', 'A'], loc='upper left', fontsize=4, handlelength=1.5, 
        #     columnspacing=1, frameon=False, ncols=len(action_labels)
        # )
        V_txt = ax2.text(.98, .75, 0, ha='right', va='top', transform=ax2.transAxes, fontsize=5, color='#08F7FE')
        A_txt = ax2.text(.98, .6, 0, ha='right', va='top', transform=ax2.transAxes, fontsize=5, color='r')
        
    # Initial plot of episode-wise Q(a)/Z(a) distribution
    ax3 = fig.add_axes([0.11, 0.04, .76, 0.105])
    if distr_net:
        q_axvlines = []
        Z_steps = []
        Z, pZ = Z_distr
        # Z_colors = sns.color_palette('Set2', len(action_labels))
        Z_colors = [
            '#00ff9f',
            '#00b8ff',
            '#001eff',
            '#d600ff',
            '#024059',
            '#F96CFF',
        ]

        for pZ_a, Q, c in zip(pZ[0], Qs[0], Z_colors):
            Z_step, = ax3.step(x=Z, y=pZ_a, where='post', animated=True, lw=0.5, color=c)
            q_axvline = ax3.axvline(Q, animated=True, lw=0.5, color=c, ymax=0.075)
            Z_steps.append(Z_step)
            q_axvlines.append(q_axvline)
        ax3.set_title('Z-distribution', x=0.5, y=0.70, fontsize=6, color='#00ff41')
        ylim = np.nanmax([p.max() for p in pZ]) * 1.05
        ax3.set_ylim(0, ylim if ylim is np.nan else 1)
        ax3.legend(
            Z_steps, action_labels, loc='upper left', fontsize=4, handlelength=1.5, 
            columnspacing=1, frameon=False, ncols=len(action_labels)
        )
        Z_txt = ax3.text(.98, .90, 0, ha='right', va='top', transform=ax3.transAxes, fontsize=5, color='#08F7FE')
    else:
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
        axvline.set_xdata(i)
        span.set_xy([[0, 0], [0, 1], [i, 1], [i, 0]])
        art_ls = [img, axvline, span, Q_txt]
        
        if distr_net:
            for a, (Z_step, q_axvline) in enumerate(zip(Z_steps, q_axvlines)):
                Z_step.set_ydata(pZ[i][a])
                q_axvline.set_xdata(Qs[i][a])
            art_ls += [Z_steps + q_axvlines]
            Z_txt.set_text(Qs[i].round(1))
        else:
            for j, b in enumerate(bar):
                b.set_height(Qs_norm[i][j])
            bar[animate.a_max].set_color('#08F7FE')  # Set old max to def color
            animate.a_max = np.argmax(Qs_norm[i])
            bar[animate.a_max].set_color('#024A4C')  # Set new max to max color
            Q_txt.set_text(f'Q {Qs[i].max().round(2)}')
            art_ls += list(bar) + [Q_txt]
        
        if duel_net:
            V_txt.set_text(f'V {Vs[i].round(2)[0]}')
            A_txt.set_text(f'A {As[i][Qs[i].argmax()].round(2)}')
            art_ls += [V_txt, A_txt]

        return art_ls

    animate.a_max = 0  # Add function attribute
    an = ani.FuncAnimation(fig, animate, frames=len(frames), blit=False) # Blit seems to make it slower
    writer = ani.FFMpegWriter(fps=30)
    an.save(opath, writer=writer, dpi=200)
    plt.close()


def animate_episode_sal(
        model, episode_states, episode_frames, episode_actions, episode_Qs, 
        episode_Vs, episode_As, action_space, opath, duel_net=False, 
        distr_net=False, Z_distr=None, sal_type='sal', sal_kwargs=None
    ):
    """Animates an episode with saliency map overlayed on preprocessed frames."""
    sals = run_saliency_map(
        model, episode_states, episode_actions, action_space, duel_net, 
        distr_net, sal_type, sal_kwargs
    )
    animate_episode(
        episode_frames, sals, episode_Qs, episode_Vs, episode_As,
        action_space, duel_net, opath, Z_distr
    )


def plot_log_stats(df_stats, axes=None):
    """Plots the logged episode statistics."""
    rc_cntxt = {
        'lines.linewidth': 0.75, 'xtick.labelsize': 7, 
        'ytick.labelsize': 7, 'axes.labelsize': 9
    }
    with plt.rc_context(rc_cntxt):
        if axes is None:
            fig, axes = plt.subplots(8, 1, figsize=(18, 10), sharex=True, sharey=False)
            axes[0].set_ylabel('Score')
            axes[1].set_ylabel('Max(cum)')
            axes[2].set_ylabel('Max(50)')
            axes[3].set_ylabel('Mean(50)')
            axes[4].set_ylabel('Epis len mean(50)')
            axes[5].set_ylabel('Mean-max Q')
            axes[6].set_ylabel('Time(ms) / EL')
            axes[7].set_ylabel('Time(h)')
        
            axes[-1].xaxis.set_major_locator(MultipleLocator(100000))
            fig.align_ylabels(axes)

        epis_len = df_stats['frame_num'].diff()

        df_stats['score'].plot(ax=axes[0])
        df_stats['score'].expanding().max().plot(ax=axes[1])
        df_stats['score'].rolling(50).max().plot(ax=axes[2])
        df_stats['score'].rolling(50).mean().plot(ax=axes[3])
        epis_len.rolling(50).mean().plot(ax=axes[4])
        df_stats['mean_max_Q'].plot(ax=axes[5])
        (df_stats['runtime_h_d'] * 3600 * 1000 / epis_len).plot(ax=axes[6])
        df_stats['runtime_h'].plot(ax=axes[7])
    
    if axes is None:
        return axes


def plot_set_xlim(axes, xlim):
    """Sets xlim for a plot and auto-adjust ylim accordingly."""
    plt.xlim(*xlim)

    for ax in axes:
        ax_ymin = []
        ax_ymax = []
        for line in ax.get_lines():
            x, y = line.get_data()
            y_sub = y[(x >= xlim[0]) & (x <= xlim[1])]
            ax_ymin.append(y_sub.min())
            ax_ymax.append(y_sub.max())
        ax.set_ylim(np.min(ax_ymin) * 0.95, np.max(ax_ymax) * 1.05)
