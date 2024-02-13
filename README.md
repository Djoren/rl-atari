# Atari 2600 Deep-Q Learning AI
Rl-atari implements a set of reinforcement learning algorithms to learn control policies for playing Atari games. <br>
This implementation recreates the Deep Q-Networks (DQN) model and configuration as proposed first by Google DeepMind.<br>
It combines computer vision (using convolutional neural networks) with reinforcement learning algorithms (Q-learning), in order to train an agent
that autonomously (read: without supervision) learns how to play a computer game from only pixel and reward inputs.
The following deep-RL variants and features are built out, the integrated combination of which is known as the Rainbow agent.

Training these agents takes a very long time, bringing with it an extended turnaround to obtain (new) results. New progress and figures will be pushed as they come in.
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



# Requirements
Virtual env and deps were managed with Conda and Poetry resp.
Conda builds environment from the yaml file. Poetry installs the exact requirements and dependencies from the .lock and .toml files.

# Visualization

### 1. Agent play
Custom functions are written to output comprehensive animations of agent playing episodes during training, with details such as Q values, Value and Advantage stream values, Z distribution and activation maps. Animation is generated from functions in utils.py.

Example 1: Dueling agent

The following shows a (partially) trained DQN agent playing Space Invaders attaining a training score of 2145. Displayed are:
1. **Left** Original Atari frame.
2. **Right** Preprocessed frame as viewed by agent.
   - Overlayed on this are Conv. Neural Net saliency maps to display pixel attribution in agent decision making.
   - Dueling network was used, where value stream is displayed in blue and advantage in red.
3. **Middle** Max Q-value series as estimated by agent, along with the Value and Advantage stream values.
4. **Bottom** Q-values for each action.

Note: 
- Animation is sped up to 60 fps.
- saliency/activation maps can often be rather noisy, however some particular attentions stand out, such as agent focussing on the bonus (round) flying saucer.


https://github.com/Djoren/rl-atari/assets/10808578/86d986f7-d2d4-4ef1-b97c-cc27a7b0c87a


Example 2: Distibutional agent

Idem ditto as example 1, however the Q-values have been replaced with a categorical value distribution.

https://github.com/Djoren/rl-atari/assets/10808578/4f2144b0-5ab7-4d63-b2fd-852462a9b60c



### 2. Train statistics
Visual inspection of metrics during training is imperative to measure model performance, analyze agent behavior, 
and predict progress and runtimes. First plot below displays statistics aggregated per train episode, showing total reward score,
episode length (in terms of played frames), mean of max Q values, mean TD-errors and runtime.
The second plots show distribution of actions taken over time.

Plots are generated from functions in utils.py.

![res](https://github.com/Djoren/rl-atari/assets/10808578/b515287e-fe02-4982-aeb5-96fdd09c803c)

###
![actions](https://github.com/Djoren/rl-atari/assets/10808578/78080a57-636b-4f19-a7b9-9873c76db789)




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

![dddd](https://github.com/Djoren/rl-atari/assets/10808578/d59f1cdb-8868-4af6-bd3f-a4deea36a70c)


###


