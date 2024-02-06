"""Implements replay memory classes for reinforcement learning.

A replay memory is intended to store information about transitions of 
a Markov Decision Process (MDP), such that we can replay these past experiences and
continue to learn from them. 
Module defines both a uniform replay memory as well as a derived
prioritized replay memory. Literature ref for latter: https://arxiv.org/abs/1511.05952/.
"""

import numpy as np
import random 
from typing import List

from ring_buffer import RingBuffer
from segment_tree import SumTree, MaxTree, MinTree


class UniformReplayMemory(RingBuffer):
    """Implements a uniform replay memory.

    "Uniform" refers to the sampling mechanism, i.e. when we 
    request a block of experiences it uses uniform sampling.
    Uses a ring buffer as it's underlying data structure for storage.

    Args:
        replay_len: Fixed max size of memories to store.
        state_len: State length, given in therms of no. of frames.
        n: Number of steps in "n-step return".

    Attributes:
        N_lower_bound: Lowest bound of memories we can sample from.
            This bound is determined by len of state and n-step return.
    """
    def __init__(self, replay_len: int, state_len: int, n: int) -> None:
        super().__init__(replay_len)
        self.state_len = state_len
        self.n = n 
        self.N_lower_bound = state_len + n - 1  # Inclusive
        
    def store_memory(self, memory: tuple) -> None:
        """Store a transition memory.

        Assumes stored elements are, in order: 
            A(t), R(t+1), game_over(t+1), life_lost(t+1), S(t+1)
        """
        self.append((
            np.uint8(memory[0]), np.float16(memory[1]), np.bool_(memory[2]),
            np.bool_(memory[3]), memory[4]
        ))
    
    def get_action(self, idx: int) -> np.uint8:
        """Retrieve action of specific memory."""
        return self[idx][0]
    
    def get_reward(self, idx: int) -> np.float16:
        """Retrieve reward of specific memory."""
        return self[idx][1]
    
    def get_gameover(self, idx: int) -> np.bool_:
        """Retrieve terminal indicator of specific memory."""
        return self[idx][2]
    
    def get_lifelost(self, idx: int) -> np.bool_:
        """Retrieve life-lost indicator of specific memory."""
        return self[idx][3]
    
    def get_frame(self, idx: int) -> np.array:
        """Retrieve state-frame of specific memory."""
        return self[idx][4]
    
    def get_state(self, idx: int) -> np.array:
        """Retrieve state of specific memory.

        State equals a set of state_len frames.
        """
        if idx < self.state_len - 1:
            raise ValueError(f'Index must be larger than state length ({self.state_len - 1})')
        return np.stack([self.get_frame(i) for i in range(idx + 1 - self.state_len, idx + 1)], axis=2)
    
    def get_memory(self, idx, n) -> tuple:
        """Retrieve a full memory.

        Gets the transition sequence:
            S(t), A(t), R[t+1:t+n], game_over(t+n)/life_lost(t+n), S(t+n)
        
        Args:
            idx: Index of memory. This is treated as future time step t+n.
            n: N-step ahead return. Allows us to use a different n from one set in class.

        Note: 
            - A(t) is not stored with S(t), but instead with S(t+1), hence we
              get A(t) from idx-n+1 = t+1.
            - Ditto for rewards.
        """
        return (
            self.get_state(idx - n), self.get_action(idx - n + 1), 
            [self.get_reward(j) for j in range(idx - n + 1, idx + 1)], 
            self.get_gameover(idx), self.get_state(idx), idx
        )  
    
    def get_memories(self, idxs: list, n: int) -> np.array:
        """Get multiple memories from a set of indices."""
        return np.array([self.get_memory(idx, n) for idx in idxs]).T

    def _get_ran_index(self) -> int:
        """Samples a random number uniformly."""
        return random.sample(range(self.N_lower_bound, self.len), 1)[0]

    def get_sample(self, sample_size: int) -> np.array:
        """Samples a random set of memories w/o replacement.

        Note:
            - Sample indices correspond to the next (t+n) state, not current state (t).
            - Enforced is that any sampled transition cannot cross different lives (or episodes.)
        
        Args:
            n: Specified step in multi-step learning.
            sample_size: Desired size of sample.
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
    
    def _effective_replay_len(self) -> int:
        """Returns the effective replay length, i.e. no. of transitions stored."""
        return max(0, self.len - self.N_lower_bound)


class ReplayMemory(UniformReplayMemory):
   """Deprecated: only used for old pickled files."""
   def  __init__(self, *args):
        super().__init__(*args)


class PrioritizedReplayMemory(UniformReplayMemory):
    """Implements a prioritized replay memory, building on uniform replay.

    "Prioritized" refers to the sampling mechanism, i.e. when we 
    request a block of experiences it uses prioritized sampling
    based on a set of priority weights for each memory. Uses a ring 
    buffer as it's underlying data structure for storage.

    Args:
        priorities: Weights for prioritized sampling, structures as sum tree.
        p_maxtree: Max tree to query max priority weight from.
        p_mintree: Min tree to query max priority weight from.
        p_new_mem: Priority assigned to new memories. This value is dynamic.
        alpha: Exponent of priorities, to tune up or down the stength of prioritization.
            Alpha of 0 forces uniform sampling.
        beta: Exponent of importance sampling weights.
            Beta of 0 gives uniform weights.

    Attributes:
        N_lower_bound: Lowest bound of memories we can sample from.
            This bound is determined by len of state and n-step return.

    TODO:
        - Add an epsilon param.
        - Does the below logic ignore that transitions can be seen before?
    """
        
    def __init__(
            self, replay_len: int, state_len: int, n: int = 1, 
            alpha: int = 1, beta: int = 1
        ) -> None:
        UniformReplayMemory.__init__(self, replay_len, state_len, n)  # Inherit replay buffer

        # Make composition with priority trees
        self.priorities = SumTree(replay_len) 
        self.p_maxtree = MaxTree(replay_len)
        self.p_mintree = MinTree(replay_len)
        self.p_new_mem = 1
        self.alpha = alpha 
        self.beta = beta
        self._lower_bound_idxs = np.arange(self.N_lower_bound)

    def get_priority_prob(self, p: float) -> float:
        """Convert priority weight to a priority probability."""
        return p / self.priorities.get_total_sum()
    
    def get_w_max(self) -> float:
        """Get maximum importance sampling weight."""
        prob_min = self.get_priority_prob(self.p_mintree.get_min())
        return self._calc_imps_weight(prob_min) 

    def get_imps_weight(self, buff_idx: int) -> float:
        """Get importance sampling weight.
        
        Args:
            buff_idx: Index in data buffer to query; 0 is first in, -1 is last in.
        """
        idx = self.buffer_idx_to_tree_idx(buff_idx)
        prob = self.get_priority_prob(self.priorities.tree[idx])
        w_normed = self._calc_imps_weight(prob) / self.get_w_max()  # Normalize to [0, 1]
        return w_normed
    
    def get_imps_weights(self, buff_idxs: list) -> np.array:
        """Get batch of importance weigths."""
        return np.array(list(map(self.get_imps_weight, buff_idxs)))
    
    def get_priority(self, buff_idx: int) -> float:
        """Get priority weight.
        
        Args:
            buff_idx: Index in data buffer to query; 0 is first in, -1 is last in.
        """
        return self.priorities.tree[self.buffer_idx_to_tree_idx(buff_idx)]

    def get_priorities(self, buff_idxs: list) -> list:
        """Get batch of priority weights."""
        return list(map(self.get_priority, buff_idxs))

    def _calc_imps_weight(self, prob: float) -> float:
        """Compute importance sampling weight from priority probability."""
        N = self._effective_replay_len()
        return 1 / (N * prob) ** self.beta

    def store_memory(self, memory: tuple) -> None:
        """Store a new memory.

        Overrides method of parent class.
        Adds a new memory to the memory buffer and immediately updates
        the priority trees as well.
        """
        # First check if any of k=(state_len + n - 1) last inserted memories belong to different life/episode.
        # If so set priority to 0 such that this memory will not be sampled.
        # Why: when we sample two transitions, we 
        #   1. don't want the two states to be from different lives/eps,
        #   2. we don't want any transition to be composed of frames from different lives/eps.
        # Hence, (state_len + n - 1) consecutive frames need to be from same life/eps. The last frame can be a game-over,
        # but the last frame cannot be a new life [TODO: currently a bug needs to be fixed]
        crosses_lives = any([self.get_gameover(i) for i in np.arange(max(self.len - self.N_lower_bound, 0), self.len)])
        
        # Store memory data in replay buffer
        super().store_memory(memory)

        # Add priority values
        if crosses_lives:
            self.priorities.append(0)
            self.p_maxtree.append(-np.inf)
            self.p_mintree.append(np.inf)
        else:
            self.priorities.append(self.p_new_mem) # TODO: no need for alpha exponent here [?]
            self.p_maxtree.append(self.p_new_mem)
            self.p_mintree.append(self.p_new_mem)

        # Zero out k=(state_len + n - 1) oldest priorities in the memory buffer
        self._zero_out_lower_memories()

    def _zero_out_lower_memories(self) -> None:
        """Excludes all memories below lowerbound from sampling, by setting prior. to 0."""
        idxs = self.buffer_idx_to_tree_idx(self._lower_bound_idxs)
        for idx in idxs:
            self.priorities.update(idx, 0)
            self.p_maxtree.update(idx, -np.inf)
            self.p_mintree.update(idx, np.inf)

    def _update_priority(self, buff_idx: int, p: float) -> None:
        """Replace a priority weight.
        
        Args:
            buff_idx: Data buffer index to update.
            p: New priority weight.
        """
        # Map buffer index to tree index
        idx = self.buffer_idx_to_tree_idx(buff_idx)

        # Update priority in tree and min/max priority
        p **= self.alpha
        self.priorities.update(idx, p)
        self.p_maxtree.update(idx, p)
        self.p_mintree.update(idx, p)

    def update_priorities(self, idxs: list, ps: float) -> None:
        """Replace a set of new priorities.
        
        Args:
            idxs: Data buffer indices to update.
            ps: New priority weights.
        """
        for idx, p in zip(idxs, ps):
            self._update_priority(idx, p)

        # Update the priority given to new memories
        self.p_new_mem = max(self.p_new_mem, self.p_maxtree.get_max())

    def buffer_idx_to_leaf_idx(self, buff_idx: int) -> int:
        """Convert a data buffer index to a segment tree leaf index."""
        return (buff_idx + self.priorities.app_idx) % self.len

    def leaf_idx_to_buffer_idx(self, leaf_idx: int) -> int:
        """Convert a segment tree leaf index to a data buffer index."""
        return (leaf_idx - self.priorities.app_idx) % self.len
    
    def buffer_idx_to_tree_idx(self, buff_idx: int) -> int:
        """Convert a data buffer index to a segment tree index."""
        idx = self.buffer_idx_to_leaf_idx(buff_idx)
        return self.priorities.leaf_to_tree_index(idx)

    def _get_ran_index(self) -> int:
        """Samples a random index of the data buffer using priority weights.
        
        This method overrides the one from parent class.
        """
        idxs_excl = self._lower_bound_idxs
        leaf_idxs_excl = self.buffer_idx_to_leaf_idx(idxs_excl)  # TODO: should this be used?
        ran_uni = np.random.uniform(low=0, high=self.priorities.get_total_sum())
        ran_leaf_idx = self.priorities.value_index(ran_uni)
        ran_buff_idx = self.leaf_idx_to_buffer_idx(ran_leaf_idx)

        return ran_buff_idx
