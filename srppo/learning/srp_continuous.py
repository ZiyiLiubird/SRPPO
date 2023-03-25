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

import isaacgymenvs.learning.replay_buffer as replay_buffer
import isaacgymenvs.learning.common_agent as common_agent 

from tensorboardX import SummaryWriter

from srppo.learning.batch_fifo_pdf import BatchFIFO
from srppo.learning.particle import Particle

EPS = 1e-5
DR_MIN, DR_MAX = 0.05, 10.0


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

        for n, p in self.model.named_parameters():
            if n.__contains__('actor') or n.__contains__('sigma') or n.__contains__('mu'):
                self.actor_params.append(p)
            elif n.__contains__('critic') or n.__contains__('value'):
                self.critic_params.append(p)
            else:
                self.pdf_params.append(p)

        self.states = None
        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)

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

        self.algo_observer.after_init(self)

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

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)

        if self.stein_phase:
            aux_returns = batch_dict['aux_returns']
            aux_values = batch_dict['aux_values']

            aux_advantages = aux_returns - aux_values

            if self.normalize_value:
                for i in range(self.num_particles):
                    self.aux_value_mean_std[i].train()
                    aux_values[i] = self.aux_value_mean_std[i](aux_values[i])
                    aux_returns[i] = self.aux_value_mean_std[i](aux_returns[i])
                    self.aux_value_mean_std[i].eval()
            
            for i in range(self.num_particles):
                aux_advantages[i] = torch.sum(aux_advantages[i], axis=1)

            if self.normalize_aux_advantage:
                if self.normalize_aux_rms_advantage:
                    for i in range(self.num_particles):
                        aux_advantages[i] = self.aux_advantage_mean_std(aux_advantages[i])
                else:
                    for i in range(self.num_particles):
                        aux_advantages[i] = (aux_advantages[i] - aux_advantages[i].mean()) / (aux_advantages[i].std() + 1e-8)

            self.dataset.values_dict['old_aux_values'] = aux_values
            self.dataset.values_dict['aux_advantages'] = aux_advantages
            self.dataset.values_dict['aux_returns'] = aux_returns

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

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time

        mb_obses = self.experience_buffer.tensor_dict['obses']
        mb_actions = self.experience_buffer.tensor_dict['actions']
        # TODO
        self.current_trajs_buffer.add(mb_obses, mb_actions)

        if self.stein_phase:
            ## calculate aux_critic advantages and returns.
            last_aux_values = last_values_dict['aux_values']
            aux_nb_values = self.experience_buffer.tensor_dict['aux_values']
            self._calc_stein_force_rewards(mb_obses, mb_actions)
            aux_nb_rewards = self.experience_buffer.tensor_dict['aux_rewards']
            aux_nb_returns = []
            for i in range(self.num_particles):
                nb_advs = self.discount_values(fdones, last_aux_values[i],
                                               mb_fdones, aux_nb_values[i],
                                               aux_nb_rewards[i])
                nb_returns = nb_advs + aux_nb_values[i]
                nb_returns = a2c_common.swap_and_flatten01(nb_returns)
                aux_nb_returns.append(nb_returns)

            batch_dict['aux_returns'] = torch.stack(aux_nb_returns, axis=0)
            batch_dict['aux_values'] = aux_nb_values

        return batch_dict

    def train_epoch(self):
        self.set_eval()
        play_time_start = time.time()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps() 

        if self.epoch_num % self.particle_update_interval == 0:
            num_effect_particles = self.epoch_num // self.particle_update_interval - 1
            if num_effect_particles < self.num_particles:
                index = num_effect_particles
                self.past_particles[index].update_model(self.model)
                self.past_particles[index].update_traj_buffer(self.current_trajs_buffer)
            else:
                index = num_effect_particles % self.num_particles
                self.past_particles[index].update_model(self.model)
                self.past_particles[index].update_traj_buffer(self.current_trajs_buffer)
                print('Add particle')
                print('iteration ', self.epoch_num)
                print('---------------------------------------------')

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
                a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss = self.train_actor_critic(self.dataset[i])
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
        m_pdf = self.model.infer_pdf(input_dict, detach=True)
        horizon_length = mb_obses.shape[0]
        for i in range(self.num_particles):
            n_pdf = self.past_particles[i].model.infer_pdf(input_dict, detach=True)
            if self._divergence == "js":
                ratio = self._dr(m_pdf, n_pdf)
                rewards = -torch.log(1. / (1. + ratio) + EPS)
            elif self._divergence == "kls":
                ratio = self._dr(n_pdf, m_pdf)
                rewards = -ratio - torch.log(ratio + EPS)
            else:
                raise ValueError("Unknown divergence")

            rewards = rewards.view(horizon_length, -1, 1)
            self.experience_buffer.tensor_dict['aux_rewards'][i].copy_(rewards)

    def calc_gradients(self, input_dict):
        # self.set_train()
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        if self.stein_phase:
            aux_value_preds_batch = input_dict['old_aux_values']
            aux_advantages = input_dict['aux_advantages']
            aux_return_batch = input_dict['aux_returns']

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
            'stein_phase': self.stein_phase,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len

            if self.zero_rnn_on_done:
                batch_dict['dones'] = input_dict['dones']            

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            if self.stein_phase:
                aux_values = res_dict['aux_values']

            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)
            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss , entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            # loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
            actor_loss = a_loss - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
            critic_loss = 0.5 * c_loss * self.critic_coef
            pdf_loss = None
            
            if self.multi_gpu:
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                self.pdf_optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(actor_loss).backward()
        self.scaler.scale(critic_loss).backward()
        self.scaler.scale(pdf_loss).backward()
        #TODO: Refactor this ugliest code of they year
        self.trancate_gradients_and_step(identity='actor')
        self.trancate_gradients_and_step(identity='critic')
        self.trancate_gradients_and_step(identity='pdf')

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask

        self.diagnostics.mini_batch(self,
        {
            'values' : value_preds_batch,
            'returns' : return_batch,
            'new_neglogp' : action_log_probs,
            'old_neglogp' : old_action_log_probs_batch,
            'masks' : rnn_masks
        }, curr_e_clip, 0)      

        self.train_result = (a_loss, c_loss, entropy, \
            kl_dist, self.last_lr, lr_mul, \
            mu.detach(), sigma.detach(), b_loss)

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
