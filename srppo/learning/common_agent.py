from rl_games.algos_torch import a2c_continuous
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch.moving_mean_std import MovingMeanStd
from rl_games.algos_torch import torch_ext
from rl_games.common import common_losses
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
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import gym
from tensorboardX import SummaryWriter
from srppo.learning.batch_fifo_pdf import BatchFIFO

EPS = 1e-5
DR_MIN, DR_MAX = 0.05, 10.0

class CommonAgent(a2c_continuous.A2CAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        config = params['config']
        self._load_config_params(config)
        self._setup_action_space()
        self.bounds_loss_coef = config.get('bounds_loss_coef', None)
        self.clip_actions = config.get('clip_actions', True)

        net_config = self._build_net_config()
        self.model = self.network.build(net_config)
        self.model.to(self.ppo_device)

        self.states = None
        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)

        self.actor_params = []
        self.critic_params = []
        self.pdf_params = []

        for n, p in self.model.named_parameters():
            if n.__contains__('actor') or n.__contains__('sigma') or n.__contains__('mu'):
                self.actor_params.append(p)
            elif n.__contains__('critic') or n.__contains__('value'):
                self.critic_params.append(p)
            else:
                self.pdf_params.append(p)
        self.actor_pcnt = sum(p.numel() for p in self.actor_params if p.requires_grad)
        
        self.burnin_epoch = config['burnin_epoch']
        self.burnin_epoch_num = 0
        self.stein_phase = False
        self.particle_update_interval = config['particle_update_interval']
        self.normalize_aux_advantage = config['normalize_aux_advantage']
        self.normalize_aux_rms_advantage = config.get('normalize_aux_rms_advantage', False)

        self.actor_optimizer = optim.Adam(self.actor_params, float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)
        self.critic_optimizer = optim.Adam(self.critic_params, float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)
        self.pdf_optimizer = optim.Adam(self.pdf_params, float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)

        if self.normalize_value:
            self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std
            self.aux_value_mean_std = self.model.aux_value_mean_std
        
        if self.normalize_aux_advantage and self.normalize_aux_rms_advantage:
            momentum = self.config.get('adv_rms_momentum', 0.5) #'0.25'
            self.aux_advantage_mean_std = [MovingMeanStd((1,), momentum=momentum).to(self.ppo_device) for i in range(self.num_particles)]

    def set_eval(self):
        self.model.eval()
        if self.normalize_rms_advantage:
            self.advantage_mean_std.eval()
            for i in range(self.num_particles):
                self.aux_advantage_mean_std[i].eval()

    def set_train(self):
        self.model.train()
        if self.normalize_rms_advantage:
            self.advantage_mean_std.train()
            for i in range(self.num_particles):
                self.aux_advantage_mean_std[i].train()

    def get_stats_weights(self):
        state = super().get_stats_weights()
        if self._normalize_pdf_input:
            state['pdf_input_mean_std'] = self.model.pdf_input_mean_std.state_dict()
        if self.normalize_value:
            state['aux_value_mean_std'] = [self.model.aux_value_mean_std[i].state_dict() for i in range(self.num_particles)]
        if self.normalize_rms_advantage:
            state['aux_advantage_mean_std'] = [self.aux_advantage_mean_std[i].state_dict() for i in range(self.num_particles)]
        return state

    def set_stats_weights(self, weights):
        super().set_stats_weights(weights)
        if self._normalize_pdf_input:
            self.model.pdf_input_mean_std.load_state_dict(weights['pdf_input_mean_std'])
        if self.normalize_value:
            for i in range(self.num_particles):
                self.model.aux_value_mean_std[i].load_state_dict(weights['aux_value_mean_std'][i])
        if self.normalize_rms_advantage:
            for i in range(self.num_particles):
                self.aux_advantage_mean_std[i].load_state_dict(weights['aux_advantage_mean_std'][i])
        return

    def train_epoch(self):
        self.set_eval()
        play_time_start = time.time()

        self.update_particle = False
        if self.epoch_num % self.particle_update_interval == 0:
            self.update_particle = True

        if self.update_particle:
            num_effect_particles = self.epoch_num // self.particle_update_interval - 1
            if num_effect_particles < self.num_particles:
                self.current_index = num_effect_particles
                self.past_particles[self.current_index].update_model(self.model)
            else:
                self.current_index = num_effect_particles % self.num_particles
                self.past_particles[self.current_index].update_model(self.model)

        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        self.set_train()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        kls = []
        train_info = None

        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss = self.train_actor_critic(self.dataset[i], mini_ep, i)
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.dataset.update_mu_sigma(cmu, csigma)
                if self.schedule_type == 'legacy':
                    av_kls = kl
                    if self.multi_gpu:
                        dist.all_reduce(kl, op=dist.ReduceOp.SUM)
                        av_kls /= self.rank_size
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                    self.update_lr(self.last_lr)

            av_kls = torch_ext.mean_list(ep_kls)
            if self.multi_gpu:
                dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                av_kls /= self.rank_size
            if self.schedule_type == 'standard':
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                self.update_lr(self.last_lr)

            kls.append(av_kls)
            self.diagnostics.mini_epoch(self, mini_ep)
            if self.normalize_input:
                self.model.running_mean_std.eval() # don't need to update statstics more than one miniepoch

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul

    def update_epoch(self):
        if not self.stein_phase:
            if self.burnin_epoch_num > self.burnin_epoch:
                self.stein_phase = True
            else:
                self.burnin_epoch_num += 1
        return super().update_epoch()

    def train_actor_critic(self, input_dict, mini_epoch, cur_id):
        self.calc_gradients(input_dict, mini_epoch, cur_id)
        return self.train_result

    def train(self,):
        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.frame = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        if self.multi_gpu:
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])
        
        while True:
            epoch_num = self.update_epoch()
            train_info = self.train_epoch()

    def _dr(numer, denom):
        return torch.div(numer, denom).clamp(min=DR_MIN, max=DR_MAX)

    def trancate_gradients_and_step(self, identity):
        if self.truncate_grads:
            if identity == 'actor':
                self.scaler.unscale_(self.actor_optimizer)
                nn.utils.clip_grad_norm_(self.actor_params, self.grad_norm)
            elif identity == 'critic':
                self.scaler.unscale_(self.critic_optimizer)
                nn.utils.clip_grad_norm_(self.critic_params, self.grad_norm)
            elif identity == 'pdf':
                self.scaler.unscale_(self.pdf_optimizer)
                nn.utils.clip_grad_norm_(self.pdf_params, self.grad_norm)
            else:
                raise NotImplementedError

        if identity == 'actor':
            self.scaler.step(self.actor_optimizer)
        elif identity == 'critic':
            self.scaler.step(self.critic_optimizer)
        elif identity == 'pdf':
            self.scaler.step(self.pdf_optimizer)
        else:
            raise NotImplementedError

        self.scaler.update()

    def update_lr(self, lr):
        if self.multi_gpu:
            lr_tensor = torch.tensor([lr], device=self.device)
            dist.broadcast(lr_tensor, 0)
            lr = lr_tensor.item()

        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.pdf_optimizer.param_groups:
            param_group['lr'] = lr
        if self.has_central_value:
           self.central_value_net.update_lr(lr)

    def _load_config_params(self, config):
        
        self._stein_repulsive_wt = config.get('stein_repulsive_wt', None)
        self._divergence = config.get('divergence', 'kls')
        self._temperature = config.get('temperature', 1.0)
        self._normalize_pdf_input = config.get('normalize_pdf_input', True)
        self.num_particles = config['num_particles']
        # self._traj_buffer_capacity = config['traj_buffer_capacity']
        return

    def _setup_action_space(self):
        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]

        # todo introduce device instead of cuda()
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.ppo_device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.ppo_device)
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
            'normalize_pdf_input': self._normalize_pdf_input,
            'num_particles': self.num_particles
        }
        return config

    def get_full_state_weights(self):
        state = self.get_weights()
        state['epoch'] = self.epoch_num
        state['actor_optimizer'] = self.actor_optimizer.state_dict()
        state['critic_optimizer'] = self.critic_optimizer.state_dict()
        state['pdf_optimizer'] = self.pdf_optimizer.state_dict()
        if self.has_central_value:
            state['assymetric_vf_nets'] = self.central_value_net.state_dict()
        state['frame'] = self.frame

        # This is actually the best reward ever achieved. last_mean_rewards is perhaps not the best variable name
        # We save it to the checkpoint to prevent overriding the "best ever" checkpoint upon experiment restart
        state['last_mean_rewards'] = self.last_mean_rewards

        if self.vec_env is not None:
            env_state = self.vec_env.get_env_state()
            state['env_state'] = env_state

        return state

    def set_full_state_weights(self, weights):
        self.set_weights(weights)
        self.epoch_num = weights['epoch'] # frames as well?
        if self.has_central_value:
            self.central_value_net.load_state_dict(weights['assymetric_vf_nets'])

        self.actor_optimizer.load_state_dict(weights['actor_optimizer'])
        self.critic_optimizer.load_state_dict(weights['critic_optimizer'])
        self.pdf_optimizer.load_state_dict(weights['pdf_optimizer'])

        self.frame = weights.get('frame', 0)
        self.last_mean_rewards = weights.get('last_mean_rewards', -100500)

        env_state = weights.get('env_state', None)

        if self.vec_env is not None:
            self.vec_env.set_env_state(env_state)

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
