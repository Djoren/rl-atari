import matplotlib.pyplot as plt
import numpy as np
import random
import imageio
from skimage.color import rgb2gray
from skimage.transform import resize
from tensorflow import image as tf_image
from tensorflow.keras import backend as K


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
    return np.sign(reward)


def get_lin_anneal_eps(i, eps_init, eps_final, n_final):
    """Linearly anneals epsilon across n_final steps."""
    eps_decay = (eps_init - eps_final) / n_final
    return max(eps_init - i * eps_decay, eps_final)


def choose_action(env, model, state, eps):
    """Eps-greedy policy."""
    if random.random() < eps:
        return env.action_space.sample()
    else:
        action_all = np.ones(env.action_space.n)  # all actions OHE encoded
        Q_val = model.predict([state[None, :], action_all[None, :]])
        return Q_val.argmax()


def plot_state(state):
    ax = plt.subplots(1, 4, figsize=(14, 5), sharex=True, sharey=True)[1].flat
    for i in range(4):
        ax[i].imshow(state[:,:,i], cmap='gray', vmin=0, vmax=255)
        for j in range(10, 110, 10):
            ax[i].axhline(j, color='lightblue', lw=0.5)
        for j in range(10, 80, 10):
            ax[i].axvline(j, color='lightblue', lw=0.5)


def frames_to_mp4(opath, frames):
    imageio.mimsave(opath, frames, fps=30)

