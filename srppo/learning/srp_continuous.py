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

from srppo.learning.common_agent import CommonAgent
from srppo.learning.batch_fifo_pdf import BatchFIFO
from srppo.learning.particle import Particle

EPS = 1e-5
DR_MIN, DR_MAX = 0.05, 10.0


class SRPAgent(CommonAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        self.is_discrete = False
        self.algo_observer.after_init(self)

    def init_tensors(self):
        super().init_tensors()
        self._build_srp_buffers()
        return

    def _build_srp_buffers(self):
        batch_shape = self.experience_buffer.obs_base_shape
        val_space = gym.spaces.Box(low=0, high=1,shape=(self.value_size,))
        past_particle_space = gym.spaces.Box(low=0, high=1,shape=(self.num_particles,))
        self.experience_buffer.tensor_dict['aux_rewards'] = torch.zeros(past_particle_space.shape + batch_shape + val_space.shape,
                                                                        dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['aux_values'] = torch.zeros(past_particle_space.shape + batch_shape + val_space.shape,
                                                                       dtype=torch.float32, device=self.ppo_device)
        obses_shape = self.experience_buffer.tensor_dict['obses'].shape
        act_shape = self.experience_buffer.tensor_dict['actions'].shape
        # self.current_trajs_buffer = BatchFIFO()
        self.past_particles = {}
        for i in range(self.num_particles):
            self.past_particles[i] = Particle(index=i)

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

        if self.update_particle:
            self.past_particles[self.current_index].update_traj(obs=batch_dict['obses'],
                                                                acs=batch_dict['actions'])
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

    def calc_gradients(self, input_dict, mini_epoch, cur_id):
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

            # calculate policy gradients.
            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)

            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss , entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            actor_loss = a_loss - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
            critic_loss = 0.5 * c_loss * self.critic_coef
            # TODO: this
            pdf_loss = None

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            self.pdf_optimizer.zero_grad()


            # calculate stein repulsive force
            if self.stein_phase:
                self.scaler.scale(actor_loss).backward(retain_graph=True)
                grad_current_policy = parameters_to_vector((p.grad for p in self.actor_params)).detach()
                assert grad_current_policy.size(0) == self.actor_pcnt
                if self.update_particle and mini_epoch == 0:
                    self.past_particles[self.current_index].update_gradients(index=cur_id, grads=grad_current_policy)
                self.actor_optimizer.zero_grad()

                grad_exploration = torch.zeros_like(grad_current_policy, device=self.ppo_device)
                grad_exploitation = grad_current_policy.clone()

                for i in range(self.num_particles):
                    nb_a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, aux_advantages[i], self.ppo, curr_e_clip)
                    nb_a_loss = nb_a_loss.mean()
                    self.actor_optimizer.zero_grad()
                    nb_a_loss.backward(retain_graph=(i!=self.num_particles-1))
                    grad_stein_force = parameters_to_vector((p.grad for p in self.actor_params)).detach()
                    self.actor_optimizer.zero_grad()
                    grad_exploration += kernel_vals[i] * grad_stein_force

                    grad_nb = self.past_particles[i].grad_current_policy[cur_id]
                    assert grad_nb.size(0) == self.actor_pcnt
                    grad_exploitation += kernel_vals[i] * grad_nb
                
                # gradient averaging
                # TODO  
                omega = anneal_coef if self.svpg_expl_wt is None else self.svpg_expl_wt
                grad_avg = grad_exploration * omega + grad_exploitation * (1 - omega)
                grad_avg /= (sum(kernel_vals.values()) + 1)

            else:
                self.scaler.scale(actor_loss).backward()
                self.scaler.scale(critic_loss).backward()
                self.scaler.scale(pdf_loss).backward()
                grad_current_policy = parameters_to_vector((p.grad for p in self.actor_params)).detach()
                if self.update_particle and mini_epoch == 0:
                    self.past_particles[self.current_index].update_gradients(index=cur_id, grads=grad_current_policy)


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

    def eval_neg_divergence(self, m_obses, m_actions, nb_obses, nb_actions, nb_index):
        with torch.no_grad():
            m_input_dict = {
                'obs': m_obses,
                'actions': m_actions,
            }
