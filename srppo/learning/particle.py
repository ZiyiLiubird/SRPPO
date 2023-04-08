import copy
import torch
import torch.nn as nn

class Particle:
    def __init__(self, index) -> None:
        self.index = index
        self.has_model = False
        self.obs_actions = dict()
        self.grad_current_policy = dict()

    def update_gradients(self, index:int, grads:torch.Tensor):
        self.grad_current_policy[index] = grads.clone()

    def update_model(self, model:nn.Module):
        if self.has_model:
            self.model.load_state_dict(model.state_dict())
        else:
            self.model = copy.deepcopy(model)
            self.has_model = True

    def update_traj(self, obs:torch.Tensor, acs:torch.Tensor):
        # TODO
        # add diversity.
        self.obs_actions['obses'] = obs.clone()
        self.obs_actions['actions'] = acs.clone()
