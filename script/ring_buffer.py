"""Implements a ring/circular buffer data structure class.

Ring buffer is well-suited as underlying structure for implementing
for a queue of fixed size. It handles insertions w/o reshuffling, by 
doing overwrites. The current class uses a python list as it's underlying container. 
"""

import random
from typing import Union, Any, List


class RingBuffer:
    """Implements a ring/circular buffer data structure class.

    Ring buffer is well-suited as underlying structure for implementing
    for a queue of fixed size. It handles insertions w/o reshuffling, by 
    doing replacement (FIFO). The current class uses a python list as it's 
    underlying container. 

    Note:
        - Pointers are used to control the tail and head of the buffer.
          Hence, indexing on 0, -1 always return resp. the first (oldest) and last (newest) element.
    """

    def __init__(self, maxlen: int) -> None:
        """Initializes buffer of fixed length.

        Args:
            maxlen: Fixed buffer length
        """
        self.buffer = [None] * maxlen
        self.maxlen = maxlen
        self.len = 0
        self.index = 0

    def __getitem__(self, val: Union[int, slice]) -> Union[Any, list]:
        """Indexing of buffer.
        
        Args:
            val: Indexing key or slice.
        """
        if isinstance(val, slice):
            # User indices
            idx_start = val.start
            idx_stop = val.stop
            
            if val.step is not None:
                raise ValueError('non-None step not implemented')
            
            # Start/end index cannot cross end of buffer
            if idx_start >= self.len:
                raise IndexError('list index out of range')
            if idx_stop > self.len:
                idx_stop = self.len
                
            # Convert user indices to ring indices
            idx_start = (idx_start + self.index - self.len) % self.maxlen
            idx_stop = (idx_stop + self.index - self.len) % self.maxlen
        
            # Account for slice wrapping around tail of ring
            if idx_start >= idx_stop:
                return self.buffer[idx_start : None] + self.buffer[0 : idx_stop]
            return self.buffer[idx_start : idx_stop]
        else:
            if val >= self.len:
                raise IndexError('list index out of range')
            return self.buffer[(val + self.index - self.len) % self.maxlen]

    def __repr__(self) -> str:
        """Returns only first element."""
        return self.buffer[0:1].__str__()
    
    def append(self, obj) -> None:
        """Insert new object in buffer, appending at list tail.
        
        Args:
            obj: New object value to store in buffer.
        """
        self.buffer[self.index] = obj
        self.index = (self.index + 1) % self.maxlen
        self.len = min(self.len + 1, self.maxlen)
        
    def sample(self, sample_size: int) -> List[Any]:
        """Randomly samples using uniform sampling, w/o replacement. 
        
        Args:
            sample_size: Desired sample size.
        """
        if self.len == self.maxlen:
            return random.sample(self.buffer, sample_size)
        else:
            return [self.buffer[i] for i in random.sample(range(self.len), sample_size)]