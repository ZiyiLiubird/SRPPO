import copy
import torch.nn as nn

class Particle:
    def __init__(self, index) -> None:
        self.index = index
        self.has_model = False

    def update_model(self, model:nn.Module):
        if self.has_model:
            self.model.load_state_dict(model.state_dict())
        else:
            self.model = copy.deepcopy(model)
            self.has_model = True

    def update_traj_buffer(self, traj_buffer):
        # TODO
        # add diversity.
        self.traj_buffer = copy.deepcopy(traj_buffer)