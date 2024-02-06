"""Implements segment trees (also known as a statistic tree) data structures.

Segment trees are binary tree structures and are well-suited for storing 
information about dynamic array intervals as a tree. This allows efficiently 
queries to answer questions such as finding the sum, max or min of an array, 
where search is O(log n) and storage is O(n log n).
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Any


class SegmentTree(ABC):
    """Implements segment tree data structure.
    
    Class is an abstract class, to derive a specific statistic tree from.
    Derived classes specify an operator (e.g. sum, min, max), overriding the 
    method `_tree_operator()`, to define a statistic tree.

    Note:
        - Tree structure is made up of two components
          1. The leaves: are the underlying data to be stored (n nodes).
          2. The inodes: used to keep track of the statistic (n-1 nodes)
        - Both 1. and 2. are stored in a numpy array.
        - Leaves will always be the final n elements in array.
        - Similar to a ring buffer, tree data has fixed length.
          New elements are inserted by replacing old ones (FIFO).

    Args:   
        size: Underlying data size (n).

    Attributes:
        n_leaves: Size of underlying data.
        n_inodes: Size auxilliary nodes to store statistics.
        n_nodes: Total of both 1. and 2.
        app_idx: Index in leaves sub-array to append/overwrite a new leaf value.

    TODO: Maybe write class such that inodes are ignored all together
          and a getitem method only focusses on leaves. So we can remove 
          tree_to_leaf_index() etc.
    """

    def __init__(self, size: int) -> None:
        self.n_nodes = 2 * size - 1
        self.n_inodes = size - 1
        self.n_leaves = size
        self.app_idx = 0

    @abstractmethod
    def _tree_operator(self):
        """Virtual function that specified the tree operator."""
        pass

    def get_leaves(self) -> np.array:
        """Return all underlying data."""
        return self.tree[self.n_leaves - 1:]
    
    def tree_to_leaf_index(self, tree_idx: int) -> int:
        """Map tree index to leaf index, assuming breadth-first."""
        return tree_idx - self.n_leaves + 1
    
    def leaf_to_tree_index(self, leaf_idx: int) -> int:
        """Map leaf index to tree index, assuming breadth-first."""
        return leaf_idx + self.n_leaves - 1
    
    def append(self, value: Any) -> None:
        """Append a new value in the leaves.
        
        Args:
            value: Object to be inserted in the data.
        """
        idx = self.leaf_to_tree_index(0) + self.app_idx 
        self.update(idx, value)
        self.app_idx = (self.app_idx + 1) % self.n_leaves

    def update(self, idx: int, val: Any) -> None:
        """Append a value in the leaves or inodes.

        Args:
            idx: Tree index position to update.
            val: New value to insert.

        Notes: 
            - Iteration is slightly faster than recursion.
        """
        self.tree[idx] = val
        while idx != 0:
            idx = (idx - 1) // 2
            val_child_l = self.tree[idx * 2 + 1]
            val_child_r = self.tree[idx * 2 + 2]
            val_parent = self._tree_operator(val_child_l, val_child_r)
            self.tree[idx] = val_parent


class SumTree(SegmentTree):
    """Implements a segment tree that tracks global and local sum.

    This tree helps answering the following queries
        1. What is the max of (sub-) tree?
        2. For a target value (v) find the first position (i) within the leaves
           such that sum(leaves_{i-1}) < v <= sum(leaves_{i}). 
           This is useful for doing efficient weighted random sampling.

    Attributes:
        tree (np.array): Container to store full tree structure.
    """

    def __init__(self, size: int, dtype: np.dtype = np.float64) -> None:
        super().__init__(size)
        self.tree = np.zeros(self.n_nodes, dtype)

    def _tree_operator(self, x: np.dtype, y: np.dtype) -> np.dtype:
        """Overriding method specifying sum operation."""
        return x + y

    def _lookup_value(self, idx: int, val: Any) -> int:
        """Private method to execute value_index method.
        
        Args:
            idx: Base index to start search (downwards) from.
            val: Lookup value.
        """
        # Exit when reaching a leaf
        if idx >= self.n_leaves - 1:
            return idx

        idx_l = 2 * idx + 1
        val_l = self.tree[idx_l]
        if val <= val_l:
            return self._lookup_value(idx_l, val)
        else: 
            return self._lookup_value(idx_l + 1, val - val_l)

    def value_index(self, val: Any) -> int:
        """Performs lookup of target value as mentioned in class descr.

        Args:
            val: Lookup value.
        """
        idx = self._lookup_value(0, val)
        return self.tree_to_leaf_index(idx)
    
    def get_total_sum(self) -> np.dtype:
        """Return the data total sum."""        
        return self.tree[0]


class MaxTree(SegmentTree):
    """Implements a segment tree that tracks global and local max.
    
    Attributes:
        tree (np.array): Container to store full tree structure.
    """

    def __init__(self, size: int, dtype: np.dtype = np.float64) -> None:
        super().__init__(size)
        self.tree = np.ones(self.n_nodes, dtype) * -np.inf

    def get_max(self) -> np.dtype:
        """Return the data global maximum."""
        return self.tree[0]
    
    def _tree_operator(self, x: np.dtype, y: np.dtype) -> np.dtype:
        """Overriding method specifying max operation.""" 
        return max(x, y) 


class MinTree(SegmentTree):
    """Implements a segment tree that tracks global and local min.
    
    Attributes:
        tree (np.array): Container to store full tree structure.
    """

    def __init__(self, size, dtype: np.dtype = np.float64) -> None:
        super().__init__(size)
        self.tree = np.ones(self.n_nodes, dtype) * np.inf

    def get_min(self) -> np.dtype:
        """Return the data global minimum."""        
        return self.tree[0]
    
    def _tree_operator(self, x: Any, y: Any) -> np.dtype:
        """Overriding method specifying min operation.""" 
        return min(x, y) 