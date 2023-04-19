import torch as th
import numpy as np

def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, target_pi,alpha):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    # ret = target_qs.new_zeros(*target_qs.shape)
    # ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
    # # Backwards  recursive  update  of the "forward  view"
    # for t in range(ret.shape[1] - 2, -1,  -1):
    #     ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
    #                 * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
    # # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    ret = target_qs.new_zeros((rewards.shape[0], rewards.shape[1], n_agents))
    expected_qs = th.mean(target_qs, dim=-1)  # 计算期望Q值
    target_probs = target_pi.reshape(-1, n_actions)
    expected_probs = th.mean(target_probs, dim=-1).reshape(-1, n_agents)  # 计算期望动作概率
    for t in range(ret.shape[1] - 1, -1, -1):
        ret[:, t, :] = rewards[:, t, :] + gamma * (expected_qs[:, t+1] * (1 - terminated[:, t, :]) - alpha.exp() * th.log(expected_probs)[:, t, :] * (1 - terminated[:, t, :]))
    return ret




