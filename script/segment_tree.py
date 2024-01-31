import numpy as np
from abc import ABC, abstractmethod


class SegmentTree(ABC):
    # TODO: maybe write class such that inodes are ignored all together
    #       and a getitem method only focusses on leaves. 
    #       So that we can remove the tree_to_leaf_index() etc.
    def __init__(self, size, dtype):
        self.n_nodes = 2 * size - 1
        self.n_inodes = size - 1
        self.n_leaves = size
        self.app_idx = 0  # Index in leaves sub-array to append/overwrite a new leaf value

    @abstractmethod  # Decorator to define an abstract method
    def _tree_operator(self, x, y):
        pass

    def get_leaves(self):
        return self.tree[self.n_leaves - 1:]
    
    def tree_to_leaf_index(self, idx):
        """Maps tree index to leaf index."""
        return idx - self.n_leaves + 1
    
    def leaf_to_tree_index(self, idx):
        """Maps leaf index to tree index."""
        return idx + self.n_leaves - 1
    
    def append(self, value):
        """Append a new value in the leaves."""
        idx = self.leaf_to_tree_index(0) + self.app_idx 
        self.update(idx, value)
        self.app_idx = (self.app_idx + 1) % self.n_leaves

    def update(self, idx, val):
        """Note: iteration is slightly faster than recursion."""
        self.tree[idx] = val
        while idx != 0:
            idx = (idx - 1) // 2
            val_child_l = self.tree[idx * 2 + 1]
            val_child_r = self.tree[idx * 2 + 2]
            val_parent = self._tree_operator(val_child_l, val_child_r)
            self.tree[idx] = val_parent


class SumTree(SegmentTree):
    def __init__(self, size, dtype=np.float64):
        super().__init__(size, dtype)
        self.tree = np.zeros(self.n_nodes, dtype)
        # self.n_leaves_down = self.n_nodes - 2 ** (np.floor(np.log2(self.n_nodes))) + 1  # Currently unused

    def _tree_operator(self, x, y):
        return x + y

    def _lookup_value(self, idx, val):
        # Exit when reaching a leaf
        if idx >= self.n_leaves - 1:
            return idx

        idx_l = 2 * idx + 1
        val_l = self.tree[idx_l]
        if val <= val_l:
            return self._lookup_value(idx_l, val)
        else: 
            return self._lookup_value(idx_l + 1, val - val_l)

    def value_index(self, val):
        idx = self._lookup_value(0, val)
        return self.tree_to_leaf_index(idx)
    
    def get_total_sum(self):        
        return self.tree[0]


class MaxTree(SegmentTree):
    def __init__(self, size, dtype=np.float64):
        super().__init__(size, dtype)
        self.tree = np.ones(self.n_nodes, dtype) * -np.inf

    def get_max(self):        
        return self.tree[0]
    
    def _tree_operator(self, x, y):
        return max(x, y) 


class MinTree(SegmentTree):
    def __init__(self, size, dtype=np.float64):
        super().__init__(size, dtype)
        self.tree = np.ones(self.n_nodes, dtype) * np.inf

    def get_min(self):        
        return self.tree[0]
    
    def _tree_operator(self, x, y):
        return min(x, y) 