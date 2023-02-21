from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn
import numpy as np


class SRPBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return
    
    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            
            if self.has_cnn:
                if self.permute_input:
                    input_shape = torch_ext.shape_whc_to_cwh(input_shape)
            
            mlp_input_shape = self._calc_input_size(input_shape, self.actor_cnn)
            
            self._build_pdf()
    
        def load(self, params):
            super().load(params)
            
    
        def _build_pdf(self, ob_dim, ac_dim, ):
            pass


    def build(self, name, **kwargs):
        net = SRPBuilder.Network(self.params, **kwargs)
        return net