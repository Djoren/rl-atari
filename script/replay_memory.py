import numpy as np
import random 

from ring_buffer import RingBuffer


class ReplayMemory(RingBuffer):
    def __init__(self, replay_len, state_len):
        super().__init__(replay_len)
        self.state_len = state_len
        
    def store_memory(self, memory):
        """
        Assumes stored are, in order: 
        action(t), reward(t+1), game_over(t+1), life_lost(t+1), frame(t+1)
        """
        self.append((
            np.uint8(memory[0]), np.float32(memory[1]), np.bool_(memory[2]),
            np.bool_(memory[3]), memory[4]
        ))
    
    def get_action(self, idx):
        return self[idx][0]
    
    def get_reward(self, idx):
        return self[idx][1]
    
    def get_gameover(self, idx):
        return self[idx][2]
    
    def get_lifelost(self, idx):
        return self[idx][3]
    
    def get_frame(self, idx):
        return self[idx][4]
    
    def get_state(self, idx):
        """Stack of frames of length `state_len`."""
        if idx < self.state_len - 1:
            raise ValueError(f'Index must be larger than state length ({self.state_len - 1})')
        return np.stack([self.get_frame(i) for i in range(idx - self.state_len + 1, idx + 1)], axis=2)
    
    def get_memory(self, idx):
        """
        Gets the transition sequence:
            state(t), action(t), reward(t+1), game_over(t+1), life_lost(t+1), state(t+1)
        idx: t+1
        """
        return (self.get_state(idx - 1), self.get_action(idx), self.get_reward(idx), 
                self.get_gameover(idx), self.get_state(idx))        
    
    def get_sample(self, sample_size):
        """Note: samples indices correspond to the next state, not current state."""
        if sample_size > self.len:
            raise ValueError('Requested sample size larger than replay memory length.')
        
        sample_indices = set()
        while len(sample_indices) < sample_size:
            sample_idx = random.sample(range(self.state_len, self.len), 1)[0]
            
            # Sample cannot have states that cross different episodes
            # meaning the stacked frames must be in same episode
            crosses_episode = any([self.get_gameover(i) for i in range(sample_idx - self.state_len, sample_idx)])
            if not crosses_episode:
                sample_indices.add(sample_idx)
        return np.array([self.get_memory(idx) for idx in sample_indices]).T
    