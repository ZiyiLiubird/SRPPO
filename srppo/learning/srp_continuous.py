from rl_games.algos_torch import a2c_continuous
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from rl_games.common import schedulers
from rl_games.common import vecenv

from isaacgym.torch_utils import *

import time
from datetime import datetime
import numpy as np
from torch import optim
import torch 
from torch import nn

import gym

import isaacgymenvs.learning.replay_buffer as replay_buffer
import isaacgymenvs.learning.common_agent as common_agent 

from tensorboardX import SummaryWriter


class SRPAgent(a2c_continuous.A2CAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        
    
    
    def init_tensors(self):
        super().init_tensors()
        self._build_srp_buffers()
        return

    def _build_srp_buffers(self):
        batch_shape = self.experience_buffer.obs_base_shape
        val_space = gym.spaces.Box(low=0, high=1,shape=(self.value_size,))
        past_particle_space = gym.spaces.Box(low=0, high=1,shape=(self.num_particles,))
        self.experience_buffer.tensor_dict['aux_rewards'] = torch.zeros(past_particle_space.shape+batch_shape + val_space.shape,
                                                                        device=self.ppo_device)
        self.experience_buffer.tensor_dict['aux_values'] = torch.zeros(past_particle_space.shape+batch_shape + val_space.shape,
                                                                       device=self.ppo_device)


    def get_stats_weights(self):
        state = super().get_stats_weights()
        if self._normalize_pdf_input:
            state['pdf_input_mean_std'] = self.model.pdf_input_mean_std.state_dict()
        return state

    def set_stats_weights(self, weights):
        super().set_stats_weights(weights)
        if self._normalize_pdf_input:
            self.model.pdf_input_mean_std.load_state_dict(weights['pdf_input_mean_std'])
        return

    def play_steps(self):
        self.set_eval()


    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        self.dataset.values_dict['amp_obs'] = batch_dict['amp_obs']
        self.dataset.values_dict['amp_obs_demo'] = batch_dict['amp_obs_demo']
        self.dataset.values_dict['amp_obs_replay'] = batch_dict['amp_obs_replay']
        return


    def train_epoch(self):
        play_time_start = time.time()


    def calc_gradients(self, input_dict):
        self.set_train()

    def _load_config_params(self, config):
        
        self._task_reward_w = config['task_reward_w']
        self._disc_reward_w = config['disc_reward_w']

        self._amp_observation_space = self.env_info['amp_observation_space']
        self._amp_batch_size = int(config['amp_batch_size'])
        self._amp_minibatch_size = int(config['amp_minibatch_size'])
        assert(self._amp_minibatch_size <= self.minibatch_size)

        self._disc_coef = config['disc_coef']
        self._disc_logit_reg = config['disc_logit_reg']
        self._disc_grad_penalty = config['disc_grad_penalty']
        self._disc_weight_decay = config['disc_weight_decay']
        self._disc_reward_scale = config['disc_reward_scale']
        self._normalize_amp_input = config.get('normalize_amp_input', True)
        return


    def _build_net_config(self):
        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        return config

    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)

        self.writer.add_scalar('losses/disc_loss', torch_ext.mean_list(train_info['disc_loss']).item(), frame)

        self.writer.add_scalar('info/disc_agent_acc', torch_ext.mean_list(train_info['disc_agent_acc']).item(), frame)
        self.writer.add_scalar('info/disc_demo_acc', torch_ext.mean_list(train_info['disc_demo_acc']).item(), frame)
        self.writer.add_scalar('info/disc_agent_logit', torch_ext.mean_list(train_info['disc_agent_logit']).item(), frame)
        self.writer.add_scalar('info/disc_demo_logit', torch_ext.mean_list(train_info['disc_demo_logit']).item(), frame)
        self.writer.add_scalar('info/disc_grad_penalty', torch_ext.mean_list(train_info['disc_grad_penalty']).item(), frame)
        self.writer.add_scalar('info/disc_logit_loss', torch_ext.mean_list(train_info['disc_logit_loss']).item(), frame)

        disc_reward_std, disc_reward_mean = torch.std_mean(train_info['disc_rewards'])
        self.writer.add_scalar('info/disc_reward_mean', disc_reward_mean.item(), frame)
        self.writer.add_scalar('info/disc_reward_std', disc_reward_std.item(), frame)
        return
