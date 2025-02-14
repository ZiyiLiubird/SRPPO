from collections import deque
import random
from copy import deepcopy
import numpy as np
import torch

class BatchFIFO:

    def __init__(self, obses_shape, act_shape, device):
        self.obses = torch.zeros(obses_shape, dtype=torch.float32, device=device)
        self.actions = torch.zeros(act_shape, dtype=torch.float32, device=device)

    def get_sample(self, nbatches):
        if self.num_batches < nbatches:
            return self.process(random.sample(self.buffer_, self.num_batches))

        # sampling w/o replacement. Always include the most recent batch!
        batches = [self.buffer_[-1]] + random.sample(self.buffer_, nbatches-1)
        return self.process(batches)

    @staticmethod
    def process(batches):
        assert len(batches) >= 1
        obs = np.copy(batches[0]['obs'])
        acs = np.copy(batches[0]['acs'])
        for batch in batches[1:]:
            obs = np.concatenate((obs, batch['obs']), axis=0)
            acs = np.concatenate((acs, batch['acs']), axis=0)
        return torch.from_numpy(obs), torch.from_numpy(acs)

    def size(self):
        return self.capacity

    def add(self, obs, acs):
        self.obses = deepcopy(obs)
        self.acs = deepcopy(acs)

    def erase(self):
        self.buffer_ = deque()
        self.num_batches = 0
