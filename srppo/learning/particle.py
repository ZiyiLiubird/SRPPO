import copy
import torch.nn as nn

class Particle:
    def __init__(self, index) -> None:
        self.index = index
        self.has_model = False
        self.has_traj = False
        self.grad_current_policy = dict()

    def update_gradients(self, index, grads):
        self.grad_current_policy[index] = grads

    def update_model(self, model:nn.Module):
        if self.has_model:
            self.model.load_state_dict(model.state_dict())
        else:
            self.model = copy.deepcopy(model)
            self.has_model = True

    def update_traj(self, obs, acs):
        # TODO
        # add diversity.
        if self.has_traj:
            self.obses[:] = obs[:]
            self.acs[:] = acs[:]
        else:
            self.obses = copy.deepcopy(obs)
            self.acs = copy.deepcopy(acs)
            self.has_traj = True
