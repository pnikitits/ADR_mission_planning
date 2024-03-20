from collections import deque, namedtuple
import random

# create a tuple subclass that will be used to store transitions
Transition = namedtuple('Transition', 
                        ('state', 'action', 'reward', 'terminal', 'next_state'))


class ReplayBuffer():
    def __init__(self, size, minibatch_size):
        self.minibatch_size = minibatch_size
        self.max_size = size
        self.buffer = deque([], maxlen = self.max_size)

    def append(self, state, action, reward, terminal, next_state):
        self.buffer.append(Transition(state, action, reward, terminal, next_state))

    def sample(self):
        return random.sample(self.buffer, self.minibatch_size)
    
    def size(self):
        return len(self.buffer)