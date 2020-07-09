import random


class RingBuffer:
    def __init__(self, maxlen):
        self.buffer = [None] * maxlen
        self.maxlen = maxlen
        self.len = 0
        self.index = 0

    def __getitem__(self, val):
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
            
            print(self.index, idx_start, idx_stop)
        
            # Account for slice wrapping around tail of ring
            if idx_start >= idx_stop:
                return self.buffer[idx_start : None] + self.buffer[0 : idx_stop]
            return self.buffer[idx_start : idx_stop]
        else:
            if val >= self.len:
                raise IndexError('list index out of range')
            return self.buffer[(val + self.index - self.len) % self.maxlen]

    def __repr__(self):
        """Return only first element."""
        return self.buffer[0:1].__str__()
    
    def append(self, obj):
        self.buffer[self.index] = obj
        self.index = (self.index + 1) % self.maxlen
        self.len = min(self.len + 1, self.maxlen)
        
    def sample(self, sample_size):
        if self.len == self.maxlen:
            return random.sample(self.buffer, sample_size)
        else:
            return [self.buffer[i] for i in random.sample(range(self.len), sample_size)]