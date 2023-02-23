from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn
import numpy as np

PDF_LOGIT_INIT_SCALE = 1.0
DR_MIN, DR_MAX = 0.05, 10


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
            
            self._build_pdf(ob_dim=mlp_input_shape, ac_dim=actions_num)

        def load(self, params):
            super().load(params)
            
            self._pdf_units = params['pdf']['units']
            self._pdf_activation = params['pdf']['activation']
            self._pdf_initializer = params['pdf']['initializer']
            self.divergence = params['pdf'].get('divergence,' 'js')
            return

        def eval_pdf(self, normalized_sa):
            with torch.no_grad():
                _pdf_logits = self._pdf_logits(self._pdf_mlp(normalized_sa))
                pdf = self._pdf_modifier(_pdf_logits)
            return pdf

        def _pdf_modifier(self, m):
            m = torch.clamp(m - self._logZ, min=-5., max=5.)
            return m.exp()

        def _dr(numer, denom):
            return torch.div(numer, denom).clamp(min=DR_MIN, max=DR_MAX)

        def mute_param_update(self):
            for p in self.parameters():
                p.requires_grad = False

        def _build_aux_critic(self,):
            pass
        
        def _build_pdf(self, ob_shape, ac_dim):            
            self._pdf_mlp = nn.Sequential()
            
            mlp_args = {
                'input_size' : ob_shape[0]+ac_dim, 
                'units' : self._pdf_units, 
                'activation' : self._pdf_activation, 
                'dense_func' : torch.nn.Linear
            }
            self._pdf_mlp = self._build_mlp(**mlp_args)
            
            mlp_out_size = self._pdf_units[-1]
            self._pdf_logits = torch.nn.Linear(mlp_out_size, 1, bias=False)
            self._logZ = nn.Parameter(torch.ones(1))
            
            mlp_init = self.init_factory.create(**self._pdf_initializer)
            for m in self._pdf_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, 'bias', None) is not None:
                        torch.nn.init.zeros_(m.bias)
            
            torch.nn.init.uniform_(self._pdf_logits.weight, -PDF_LOGIT_INIT_SCALE, PDF_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._pdf_logits.bias) 

            return

    def build(self, name, **kwargs):
        net = SRPBuilder.Network(self.params, **kwargs)
        return net