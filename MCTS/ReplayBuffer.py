import random
from collections import namedtuple

Transition = namedtuple("data", ("obs", "pi", "vi"))


class ReplayBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


if __name__ == '__main__':
    mem = ReplayBuffer(10000)
    for i in range(64):
        mem.push(i, i, i)

    batch = mem.sample(32)
    zipped = zip(batch)
    batch = Transition(*zip(*batch))
    print()