import numpy as np
import random 

from ring_buffer import RingBuffer
from segment_tree import SumTree, MaxTree, MinTree


class UniformReplayMemory(RingBuffer):
    def __init__(self, replay_len, state_len, n):
        super().__init__(replay_len)
        self.state_len = state_len
        self.n = n  # Determines n-step returns
        self.N_lower_bound = state_len + n - 1  # Inclusive
        
    def store_memory(self, memory):
        """
        Assumes stored are, in order: 
        A(t), R(t+1), game_over(t+1), life_lost(t+1), S(t+1)
        """
        self.append((
            np.uint8(memory[0]), np.float16(memory[1]), np.bool_(memory[2]),
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
        return np.stack([self.get_frame(i) for i in range(idx + 1 - self.state_len, idx + 1)], axis=2)
    
    def get_memory(self, idx, n):
        """
        Gets the transition sequence:
            S(t), A(t), R[t+1:t+n], game_over(t+n)/life_lost(t+n), S(t+n)
        
        idx: t + n
        NOTE: A(t) is not stored with S(t), but instead with S(t+1), hence we
              get A(t) from idx-n+1 = t+1
        """
        return (
            self.get_state(idx - n), self.get_action(idx - n + 1), 
            [self.get_reward(j) for j in range(idx - n + 1, idx + 1)], 
            self.get_gameover(idx), self.get_state(idx), idx
        )  
    
    def get_memories(self, idxs, n):
        return np.array([self.get_memory(idx, n) for idx in idxs]).T

    def _get_ran_index(self):
        return random.sample(range(self.N_lower_bound, self.len), 1)[0]

    def get_sample(self, sample_size):
        """
        Note: Sample indices correspond to the next state, not current state.
              Samples without replacement.
        
        n: used for multi-step learning, to get n-step return.
        """        
        if sample_size > self.len - self.N_lower_bound:
            raise ValueError('Requested sample size larger than effective replay memory length.')
        
        sample_idxs = set()  # Ensures sampling w/o replacement
        while len(sample_idxs) < sample_size:
            idx = self._get_ran_index()
            
            # Sample cannot have states that cross different lives (and by extension episodes)
            crosses_lives = any([self.get_gameover(i) for i in range(idx - self.N_lower_bound, idx)])
            if not crosses_lives:
                sample_idxs.add(idx)

        return self.get_memories(sample_idxs, self.n)
    
    def _effective_replay_len(self):
        return max(0, self.len - self.N_lower_bound)  # Effective replay len (i.e. no. of transitions)


class ReplayMemory(UniformReplayMemory):
   """Deprecated: only used for old pickled filed."""
   def  __init__(self, *args):
        super().__init__(*args)


class PrioritizedReplayMemory(UniformReplayMemory):
        
    def __init__(self, replay_len, state_len, n=1, alpha=1, beta=1):
        # TODO: add an epsilon param
        # TODO: does the below logic ignore that transitions can be seen before?
        UniformReplayMemory.__init__(self, replay_len, state_len, n)  # Inherit replay buffer

        # Make composition with priority trees
        self.priorities = SumTree(replay_len) 
        self.p_maxtree = MaxTree(replay_len)
        self.p_mintree = MinTree(replay_len)
        self.p_new_mem = 1  # Init priority given to newly stored memories
        self.alpha = alpha  # Exponent of priority probabilities
        self.beta = beta  # Exponent of importance sampling weights

    def get_priority_prob(self, p):
        return p / self.priorities.get_total_sum()
    
    def get_w_max(self):
        prob_min = self.get_priority_prob(self.p_mintree.get_min())
        return self._calc_imps_weight(prob_min) 

    def get_imps_weight(self, buff_idx):
        idx = self.buffer_idx_to_tree_idx(buff_idx)
        prob = self.get_priority_prob(self.priorities.tree[idx])
        w_normed = self._calc_imps_weight(prob) / self.get_w_max()  # Normalize to [0, 1]
        return w_normed
    
    def get_imps_weights(self, buff_idxs):
        return list(map(self.get_imps_weight, buff_idxs))
    
    def get_priority(self, buff_idx):
        return self.priorities.tree[self.buffer_idx_to_tree_idx(buff_idx)]

    def get_priorities(self, buff_idxs):
        return list(map(self.get_priority, buff_idxs))

    def _calc_imps_weight(self, prob):
        N = self._effective_replay_len()
        return 1 / (N * prob) ** self.beta

    def store_memory(self, memory):
        # First check of any of k=(state_len + n - 1) last inserted memories belong to different life/episode.
        # If so make set priority to 0 such that this memory will not be sampled.
        # Why: when we sample two transitions, we 1. don't want the two states to be from different lives/eps,
        # and 2. we don't want any transition to be composed of frames from different lives/eps.
        # Hence, (state_len + n - 1) consecutive frames need to be from same life/eps. The last frame can be a game-over,
        # but the last frame cannot be a new life [TODO: currently a bug needs to be fixed]. 
        crosses_lives = any([self.get_gameover(i) for i in np.arange(max(self.len - self.N_lower_bound, 0), self.len)])
        
        # Store memory data in replay buffer
        super().store_memory(memory)

        # Add priority values
        if crosses_lives:
            self.priorities.append(0)
            self.p_maxtree.append(-np.inf)
            self.p_mintree.append(np.inf)
        else:
            self.priorities.append(self.p_new_mem ) # TODO: no need for alpha exponent here [?]
            self.p_maxtree.append(self.p_new_mem)
            self.p_mintree.append(self.p_new_mem)

        # Zero out k=(state_len + n - 1) oldest priorities in the memory buffer
        self._zero_out_lower_memories()

    def _zero_out_lower_memories(self):
        """Excludes all memories below lowerbound from sampling."""
        idxs = self.buffer_idx_to_tree_idx(np.arange(self.N_lower_bound))
        for idx in idxs:
            self.priorities.update(idx, 0)
            self.p_maxtree.update(idx, -np.inf)
            self.p_mintree.update(idx, np.inf)

    def update_priority(self, buff_idx, p):
        # Map buffer index to tree index
        idx = self.buffer_idx_to_tree_idx(buff_idx)

        # Update priority in tree and min/max priority
        p **= self.alpha
        self.priorities.update(idx, p)
        self.p_maxtree.update(idx, p)
        self.p_mintree.update(idx, p)
        self._zero_out_lower_memories()

        # Update the priority given to new memories
        self.p_new_mem = max(self.p_new_mem, self.p_maxtree.get_max())

    def update_priorities(self, idxs, ps):
        for idx, p in zip(idxs, ps):
            self.update_priority(idx, p)

    def buffer_idx_to_leaf_idx(self, buff_idx):
        return (buff_idx + self.priorities.app_idx) % self.len

    def leaf_idx_to_buffer_idx(self, leaf_idx):
        return (leaf_idx - self.priorities.app_idx) % self.len
    
    def buffer_idx_to_tree_idx(self, buff_idx):
        idx = self.buffer_idx_to_leaf_idx(buff_idx)
        return self.priorities.leaf_to_tree_index(idx)

    def _get_ran_index(self):
        idxs_excl = np.arange(self.N_lower_bound)
        leaf_idxs_excl = self.buffer_idx_to_leaf_idx(idxs_excl)
        ran_uni = np.random.uniform(low=0, high=self.priorities.get_total_sum())
        ran_leaf_idx = self.priorities.value_index(ran_uni)
        ran_buff_idx = self.leaf_idx_to_buffer_idx(ran_leaf_idx)

        return ran_buff_idx
