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
import torch.distributed as dist

import gym

import isaacgymenvs.learning.replay_buffer as replay_buffer
import isaacgymenvs.learning.common_agent as common_agent 

from tensorboardX import SummaryWriter

from srppo.learning.batch_fifo_pdf import BatchFIFO
from srppo.learning.particle import Particle


class SRPAgent(a2c_continuous.A2CAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        
        config = params['config']
        self._load_config_params(config)
        self.is_discrete = False
        self._setup_action_space()

        self.bounds_loss_coef = config.get('bounds_loss_coef', None)
        self.clip_actions = config.get('clip_actions', True)
        
        net_config = self._build_net_config()
        self.model = self.network.build(net_config)
        self.model.to(self.ppo_device)
        self.states = None
        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)

        self.burnin_epoch = config['burnin_epoch']
        self.burnin_epoch_num = 0
        self.stein_phase = False
        self.particle_update_interval = config['particle_update_interval']

        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)

    def init_tensors(self):
        super().init_tensors()
        self._build_srp_buffers()
        return

    def _setup_action_space(self):
        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]

        # todo introduce device instead of cuda()
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.ppo_device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.ppo_device)
        return

    def _build_srp_buffers(self):
        batch_shape = self.experience_buffer.obs_base_shape
        val_space = gym.spaces.Box(low=0, high=1,shape=(self.value_size,))
        past_particle_space = gym.spaces.Box(low=0, high=1,shape=(self.num_particles,))
        self.experience_buffer.tensor_dict['aux_rewards'] = torch.zeros(past_particle_space.shape + batch_shape + val_space.shape,
                                                                        dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['aux_values'] = torch.zeros(past_particle_space.shape + batch_shape + val_space.shape,
                                                                       dtype=torch.float32, device=self.ppo_device)

        self.current_trajs_buffer = BatchFIFO(capacity=self._traj_buffer_capacity)
        self.past_particles = {}
        for i in range(self.num_particles):
            self.past_particles[i] = Particle(index=i)

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

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)

        # store past particles trajs
        for i in range(self.num_particles):
            self.dataset.values_dict[f'particle_{i}_sa'] = self.past_particles[i].traj_buffer.get_sample(nbatches=1)
        return

    def play_steps(self):        
        update_list = self.update_list

        step_time = 0.0

        for n in range(self.horizon_length):
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs) # contain aux_values

            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)
            
            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards)

            if self.stein_phase:
                for idx, aux_value in enumerate(res_dict['aux_values']):
                    self.experience_buffer.tensor_dict['aux_values'][idx][n,:] = aux_value

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]
     
            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        last_values_dict = self.get_values(self.obs)
        last_values = last_values_dict['value']

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values
        
        if self.stein_phase:
            ## calculate aux_critic advantages and returns.
            mb_obses = self.experience_buffer.tensor_dict['obses']
            mb_actions = self.experience_buffer.tensor_dict['actions']
            last_aux_values = last_values_dict['aux_values']
            aux_mb_values = self.experience_buffer.tensor_dict['aux_values']
            self._calc_stein_force_rewards()

            

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time

        return batch_dict

    def train_epoch(self):
        self.set_eval()
        play_time_start = time.time()
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
        entropies = []
        kls = []
        if self.has_central_value:
            self.train_central_value()

        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                a_loss, c_loss, entropy, kl, last_lr, lr_mul = self.train_actor_critic(self.dataset[i])
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)

            av_kls = torch_ext.mean_list(ep_kls)
            if self.multi_gpu:
                dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                av_kls /= self.rank_size

            self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
            self.update_lr(self.last_lr)
            kls.append(av_kls)
            self.diagnostics.mini_epoch(self, mini_ep)
            if self.normalize_input:
                self.model.running_mean_std.eval() # don't need to update statstics more than one miniepoch

        if self.epoch_num % self.particle_update_interval == 0:
            pass

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul

    def update_epoch(self):
        if not self.stein_phase:
            if self.burnin_epoch_num > self.burnin_epoch:
                self.stein_phase = True
            else:
                self.burnin_epoch_num += 1
        return super().update_epoch()
        

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



    def get_action_values(self, obs):
        processed_obs = self._preproc_obs(obs['obs'])
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
            'rnn_states' : self.rnn_states,
            'stein_phase': self.stein_phase,
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_central_value:
                states = obs['states']
                input_dict = {
                    'is_train': False,
                    'states' : states,
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value
        return res_dict

    def get_values(self, obs):
        with torch.no_grad():
            if self.has_central_value:
                states = obs['states']
                self.central_value_net.eval()
                input_dict = {
                    'is_train': False,
                    'states' : states,
                    'actions' : None,
                    'is_done': self.dones,
                    'stein_phase': self.stein_phase,
                }
                value = self.get_central_value(input_dict)
            else:
                self.model.eval()
                processed_obs = self._preproc_obs(obs['obs'])
                input_dict = {
                    'is_train': False,
                    'prev_actions': None, 
                    'obs' : processed_obs,
                    'rnn_states' : self.rnn_states,
                    'stein_phase': self.stein_phase,
                }
                result = self.model(input_dict)
                values = {
                    'value': result['values']
                }
                values['aux_values'] = result['aux_values'] if self.stein_phase else None                
            return values

    def _calc_stein_force_rewards(self, mb_obses, mb_actions):
        input_dict = {
            'obs': mb_obses,
            'actions': mb_actions
        }
        m_pdfs = self.model.infer_pdf(input_dict, detach=True)
        for i in range(self.num_particles):
            pass

    def calc_gradients(self, input_dict):
        self.set_train()

    def _load_config_params(self, config):
        
        self._stein_repulsive_wt = config.get('stein_repulsive_wt', None)
        self._divergence = config.get('divergence', 'kls')
        self._temperature = config.get('temperature', 1.0)
        self._normalize_pdf_input = config.get('normalize_pdf_input', True)
        self.num_particles = config['num_particles']
        self._traj_buffer_capacity = config['traj_buffer_capacity']

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
