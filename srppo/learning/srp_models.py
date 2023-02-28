import torch
import torch.nn as nn
from rl_games.algos_torch.models import ModelA2CContinuousLogStd
from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs


class ModelSRPContinuous(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)
        return
    
    def build(self, config):
        net = self.network_builder.build('srp', **config)
        for name, _ in net.named_parameters():
            print(name)

        obs_shape = config['input_shape']
        actions_num = config['actions_num']
        normalize_value = config.get('normalize_value', False)
        normalize_input = config.get('normalize_input', False)
        normalize_pdf_input = config.get('normalized_pdf_input', False)
        value_size = config.get('value_size', 1)
        
        return self.Network(net, normalize_pdf_input=normalize_pdf_input, actions_num=actions_num,
                            obs_shape=obs_shape, normalize_value=normalize_value,
                            normalize_input=normalize_input, value_size=value_size)

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network, normalize_pdf_input, actions_num, **kwargs):
            super().__init__(a2c_network, **kwargs)
            obs_shape = kwargs['obs_shape']
            input_shape = obs_shape + actions_num
            self.normalize_pdf_input = normalize_pdf_input
            if normalize_pdf_input:
                self.pdf_input_mean_std = RunningMeanStd((input_shape,))
            return
        
        def forward(self, input_dict):
            
            
            is_train = input_dict.get('is_train', True)
            result = super().forward(input_dict)
        
        def norm_sa(self, sa):
            with torch.no_grad():
                return self.pdf_input_mean_std(sa) if self.normalize_pdf_input else sa
        
        def infer_pdf(self, input_dict, detach=False):
            obs = input_dict['obs']
            acts = input_dict['prev_actions']
            sa = torch.cat([obs, acts], dim=-1)
            normalized_sa = self.norm_sa(sa)
            if detach:
                result = {
                    'pdf': self.a2c_network.eval_pdf(normalized_sa).detach()
                }
            else:
                result = {
                    'pdf': self.a2c_network.eval_pdf(normalized_sa)
                }
            return result