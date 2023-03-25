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
            actions_num = kwargs['actions_num']
            input_shape = kwargs['input_shape']
            super().__init__(params, **kwargs)
            
            mlp_input_shape = input_shape[0]
            num_particles = kwargs['num_particles']

            self._build_pdf(ob_dim=mlp_input_shape, ac_dim=actions_num)
            self._build_aux_critics(mlp_input_shape=mlp_input_shape, num_particles=num_particles)

        def load(self, params):
            super().load(params)

            self._pdf_units = params['pdf']['units']
            self._pdf_activation = params['pdf']['activation']
            self._pdf_initializer = params['pdf']['initializer']
            self.divergence = params['pdf'].get('divergence,' 'js')
            return

        def eval_pdf(self, normalized_sa):
            _pdf_logits = self._pdf_logits(self._pdf_mlp(normalized_sa))
            pdf = self._get_pdf(_pdf_logits)
            return pdf

        def _get_pdf(self, m):
            m = torch.clamp(m - self._logZ, min=-5., max=5.)
            return m.exp()

        def _dr(numer, denom):
            return torch.div(numer, denom).clamp(min=DR_MIN, max=DR_MAX)

        def mute_param_update(self):
            for p in self.parameters():
                p.requires_grad = False

        def _build_aux_critics(self, mlp_input_shape, num_particles):
            self.aux_critics = nn.ModuleList([
                self._build_aux_critic(mlp_input_shape=mlp_input_shape) for _ in range(num_particles)
            ])

        def _build_pdf(self, ob_dim, ac_dim):            
            self._pdf_mlp = nn.Sequential()

            mlp_args = {
                'input_size' : ob_dim+ac_dim, 
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
            # torch.nn.init.zeros_(self._pdf_logits.bias) 
            return

        def _build_aux_critic(self, mlp_input_shape):

            mlp_args = {
                'input_size' : mlp_input_shape, 
                'units' : self.units, 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer
            }
            if len(self.units) == 0:
                out_size = mlp_input_shape
            else:
                out_size = self.units[-1]
            aux_critic_mlp = self._build_mlp(**mlp_args)
            aux_value = torch.nn.Linear(out_size, self.value_size)
            aux_value_act = self.activations_factory.create(self.value_activation)
            
            aux_critic_network = nn.Sequential(*list(aux_critic_mlp), aux_value, aux_value_act)
            mlp_init = self.init_factory.create(**self.initializer)
            for m in aux_critic_network.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, 'bias', None) is not None:
                        torch.nn.init.zeros_(m.bias)
            return aux_critic_network

    def build(self, name, **kwargs):
        net = SRPBuilder.Network(self.params, **kwargs)
        return net