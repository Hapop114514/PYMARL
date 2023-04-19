import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.coma import COMACritic
from utils.rl_utils import build_td_lambda_targets
import torch as th
import numpy as np
#from torch.optim import RMSprop
from torch.optim import RAdam
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class COMALearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger
        self.entropy = -1
        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_alpha = th.tensor(np.log(0.001), dtype=th.float, requires_grad=True)
        self.log_alpha.requires_grad = True
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.target_entropy = -np.log(args.n_actions)

        self.tau = 0.01
        self.critic = COMACritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.target_mac = copy.deepcopy(self.mac)
        self.agent_params = list(mac.parameters())
        self.target_agent_params = list(self.target_mac.parameters())
        self.critic_params = list(self.critic.parameters())
        self.target_critic_params = list(self.target_critic.parameters())
        self.params = self.agent_params + self.critic_params
        self.agent_optimiser = RAdam(params=self.agent_params, lr=args.lr, betas=[args.optim_alpha,0.999], eps=args.optim_eps)
        self.critic_optimiser = RAdam(params=self.critic_params, lr=args.critic_lr, betas=[args.optim_alpha,0.999], eps=args.optim_eps)
        self.log_alpha_optimiser = RAdam(params=[self.log_alpha], lr=args.critic_lr, betas=[args.optim_alpha,0.999], eps=args.optim_eps)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # th.autograd.set_detect_anomaly(True)
        # Get the relevant quantities
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]

        critic_mask = mask.clone()

        mask = mask.repeat(1, 1, self.n_agents).view(-1)
        new_actions = actions
        actions = actions[:,:-1]
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
        target_mac_out = th.stack(target_mac_out, dim=1)  # Concat over time
        target_mac_out[avail_actions == 0] = 0
        target_mac_out = target_mac_out/target_mac_out.sum(dim=-1, keepdim=True)
        target_mac_out[avail_actions == 0] = 0
        # Calculated baseline
        target_pi = target_mac_out.view(-1, self.n_actions)
        target_pi_taken = th.gather(target_pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        target_pi_taken[mask == 0] = 1.0
        target_pi_taken = target_pi_taken.view(batch.batch_size, batch.max_seq_length - 1, self.n_agents)
        # 终止状态补全
        ones_0 = th.ones((batch.batch_size, 1, self.n_agents), dtype=target_pi_taken.dtype, device=target_pi_taken.device)
        target_pi_taken = th.cat([target_pi_taken, ones_0], dim=1) # 终止状态时所有智能体的行动概率均为1
        # Calculate policy grad with mask
        actions = new_actions
        q_vals, critic_train_stats = self._train_critic(batch, rewards, terminated, actions, avail_actions,
                                                        critic_mask, bs, max_t,target_pi_taken)

        actions = actions[:,:-1]

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0

        # Calculated baseline
        q_vals = q_vals.reshape(-1, self.n_actions)
        pi = mac_out.view(-1, self.n_actions)
        baseline = (pi * q_vals).sum(-1).detach()

        # Calculate policy grad with mask
        q_taken = th.gather(q_vals, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken = th.gather(pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken[mask == 0] = 1.0
        log_pi_taken = th.log(pi_taken)


        advantages = (q_taken - baseline).detach()
        log_pi = th.log(pi + 1e-10)               # 加上一个极小数以避免取对数时出现NaN
        entropy = - (pi * log_pi).sum(dim=1) + 1e-10     # 计算每个样本的策略熵
        entropy = (mask.view(-1) * entropy).mean()# 按样本数加权求和
        coma_loss = - (((advantages - self.log_alpha.exp() * log_pi_taken )* log_pi_taken ) * mask).sum() / mask.sum()

        # Optimise agents
        self.agent_optimiser.zero_grad()
        coma_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        # 更新温度系数alpha
        alpha_loss = th.mean((entropy-self.target_entropy).detach() * self.log_alpha.exp())
        # 梯度更新
        self.log_alpha_optimiser.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimiser.step()
        # if (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval >= 1.0:
        #     self._update_targets()
        #     self.last_target_update_step = self.critic_training_steps
        self._update_soft()
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean", "target_mean"]:
                self.logger.log_stat(key, sum(critic_train_stats[key])/ts_logged, t_env)

            self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("coma_loss", coma_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env
    # th.autograd.set_detect_anomaly(False)

    def _train_critic(self, batch, rewards, terminated, actions, avail_actions, mask, bs, max_t,target_pi):
        # Optimise critic
        target_q_vals = self.target_critic(batch)[:, :]
        targets_taken = th.gather(target_q_vals, dim=3, index=actions).squeeze(3)
        th.autograd.set_detect_anomaly(True)
        # Calculate td-lambda targets
        targets = build_td_lambda_targets(rewards, terminated, mask, targets_taken , self.n_agents, self.args.gamma, self.n_actions,target_pi,self.log_alpha)

        q_vals = th.zeros_like(target_q_vals)[:, :-1]

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        for t in reversed(range(rewards.size(1))):
            mask_t = mask[:, t].expand(-1, self.n_agents)
            if mask_t.sum() == 0:
                continue

            q_t = self.critic(batch, t)
            q_vals[:, t] = q_t.view(bs, self.n_agents, self.n_actions)
            q_taken = th.gather(q_t, dim=3, index=actions[:, t:t+1]).squeeze(3).squeeze(1)
            targets_t = targets[:, t]

            td_error = (q_taken - targets_t.detach())

            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask_t

            # Normal L2 loss, take mean over actual data
            loss = (masked_td_error ** 2).sum() / mask_t.sum()
            # v_n = []
            # v_v = []
            # v_g = []
            # for name, parameter in self.critic.named_parameters():
            #     v_n.append(name)
            #     v_v.append(parameter.detach().cpu().numpy() if parameter is not None else [0])
            #     v_g.append(parameter.grad.detach().cpu().numpy() if parameter.grad is not None else [0])

            # for i in range(len(v_n)):
            #     if np.max(v_v[i]).item() - np.min(v_v[i]).item() < 1e-6:
            #         color = bcolors.FAIL + '*'
            #     else:
            #         color = bcolors.OKGREEN + ' '
            #     print('%svalue %s: %.3e ~ %.3e' % (color, v_n[i], np.min(v_v[i]).item(), np.max(v_v[i]).item()))
            #     print('%sgrad  %s: %.3e ~ %.3e' % (color, v_n[i], np.min(v_g[i]).item(), np.max(v_g[i]).item()))
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
            self.critic_optimiser.step()
            self.critic_training_steps += 1
            th.autograd.set_detect_anomaly(False)
            running_log["critic_loss"].append(loss.item())
            running_log["critic_grad_norm"].append(grad_norm)
            mask_elems = mask_t.sum().item()
            running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
            running_log["q_taken_mean"].append((q_taken * mask_t).sum().item() / mask_elems)
            running_log["target_mean"].append((targets_t * mask_t).sum().item() / mask_elems)

        return q_vals, running_log

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.logger.console_logger.info("Updated target network")
    def _update_soft(self):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param_2, param_2 in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param_2.data.copy_(target_param_2.data * (1.0 - self.tau) + param_2.data * self.tau)
  
    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()
        self.target_mac.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
