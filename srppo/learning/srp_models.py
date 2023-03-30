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

        num_particles = config['num_particles']
        obs_shape = config['input_shape']
        actions_num = config['actions_num']
        normalize_value = config.get('normalize_value', False)
        normalize_input = config.get('normalize_input', False)
        normalize_pdf_input = config.get('normalized_pdf_input', False)
        value_size = config.get('value_size', 1)

        return self.Network(net, normalize_pdf_input=normalize_pdf_input, actions_num=actions_num,
                            num_particles=num_particles, obs_shape=obs_shape, normalize_value=normalize_value,
                            normalize_input=normalize_input, value_size=value_size)

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network, normalize_pdf_input, actions_num, num_particles, **kwargs):
            super().__init__(a2c_network, **kwargs)
            obs_shape = kwargs['obs_shape']
            input_shape = obs_shape + actions_num
            self.normalize_pdf_input = normalize_pdf_input
            self.num_particles = num_particles
            if normalize_pdf_input:
                self.pdf_input_mean_std = RunningMeanStd((input_shape,))
            if self.normalize_value:
                self.aux_value_mean_std = [RunningMeanStd((self.value_size,)) for i in range(self.num_particles)]
            return

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            obs = self.norm_obs(input_dict['obs'])
            result = super().forward(input_dict)
            stein_phase = input_dict['stein_phase']
            if stein_phase:
                if is_train:
                    aux_values = [self.a2c_network.aux_critics[i](obs) for i in range(self.num_particles)]
                else:
                    aux_values = [self.unnorm_aux_value(self.a2c_network.aux_critics[i](obs), index=i) for i in range(self.num_particles)]
                result['aux_values'] = aux_values

            return result

        def unnorm_aux_value(self, aux_value, index):
            with torch.no_grad():
                return self.aux_value_mean_std[index](aux_value, unnorm=True) if self.normalize_value else aux_value

        def norm_sa(self, sa):
            with torch.no_grad():
                return self.pdf_input_mean_std(sa) if self.normalize_pdf_input else sa

        def infer_pdf(self, input_dict, detach=False):
            obs = input_dict['obs']
            acts = input_dict['actions']
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