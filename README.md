# Atari 2600 Deep-Q Learning AI
Rl-atari implements a set of reinforcement learning algorithms to learn control policies for playing Atari games. <br>
This implementation recreates the Deep Q-Networks (DQN) model and configuration as proposed first by Google DeepMind.<br>
The following deep-RL variants and features are built out, the integrated combination of which is known as the Rainbow agent.
###

Enabled | Algorithm | Reference
:------------ | :-------------| :-----------------------------------|
:heavy_check_mark:   | Deep Q-Learning (DQN) | https://arxiv.org/abs/1312.5602 https://www.nature.com/articles/nature14236  |
:heavy_check_mark: | Double Q-Learning (DDQN) | https://arxiv.org/abs/1509.06461 |
:heavy_check_mark: |Prioritized Experience Replay (PER) | https://arxiv.org/abs/1511.05952 |
:heavy_check_mark: |Multi-step Learning | https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf |
:heavy_check_mark: | Dueling Network Architecture|  https://arxiv.org/abs/1511.06581|
:heavy_check_mark: |Noisy Network | https://arxiv.org/abs/1706.10295 |
:heavy_check_mark: |Distributional Network (C51) | https://arxiv.org/abs/1707.06887 |
| | Async Advantage Actor-Critic (A3C)| https://arxiv.org/abs/1602.01783 |
:heavy_check_mark: | Rainbow Agent | https://arxiv.org/abs/1710.02298 |
###


# Visualization

## 1. Agent Play Viz
Custom functions are written to output comprehensive animations of agent playing episodes during training, with details such as Q values, Value and Advantage stream values, Z distribution and activation maps.
The following shows a (partially) trained DQN agent playing Space Invaders attaining a training score of 2145. Displayed are:
1. **Left** Original Atari frame.
2. **Right** Preprocessed frame as viewed by agent.
   - Overlayed on this are Conv. Neural Net saliency maps to display pixel attribution in agent decision making.
   - Dueling network was used, where value stream is displayed in blue and advantage in red.
3. **Middle** Max Q-value series as estimated by agent.
4. **Bottom** Q-values for each action.

Note: saliency/activation maps can often be rather noisy, however some particular attentions stand out, such as agent focussing on the bonus (round) flying saucer.

https://github.com/Djoren/rf-atari/assets/10808578/eeabd46e-c78d-457e-8565-ae24f1060cd5

## 2. Train Statistics Viz


###

# Implementation

AI agents are build on top of convolutional neural networks coded up in Tensorflow 2. Both a small (2 conv layers) and a large network (3 conv layers) are supported.

###

Example 1: a large network with Dueling (V and A) streams, as well as noisy layers.

###

![output2](https://github.com/Djoren/rf-atari/assets/10808578/4843b2fe-c70e-47e8-9397-31a6bf672ece)

###

Example 2: of a large network with Dueling (V and A) streams, as well as distributional (C51) output.

###

![output2](https://github.com/Djoren/rf-atari/assets/10808578/4843b2fe-c70e-47e8-9397-31a6bf672ece)

###

# Requirements
See requirements yaml file.

