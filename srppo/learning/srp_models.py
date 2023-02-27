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
        normalize_value = config.get('normalize_value', False)
        normalize_input = config.get('normalize_input', False)
        normalize_pdf_input = config.get('normalized_pdf_input', False)
        value_size = config.get('value_size', 1)
        
        return self.Network(net, normalize_pdf_input=normalize_pdf_input, obs_shape=obs_shape,
            normalize_value=normalize_value, normalize_input=normalize_input, value_size=value_size)

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network, normalize_pdf_input, **kwargs):
            super().__init__(a2c_network, **kwargs)
            obs_shape = kwargs['obs_shape']
            if normalize_pdf_input:
                if isinstance(obs_shape, dict):
                    self.pdf_input_mean_std = RunningMeanStdObs(obs_shape)
                else:
                    self.pdf_input_mean_std = RunningMeanStd(obs_shape)
            return