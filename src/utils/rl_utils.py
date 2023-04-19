import torch as th

def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, n_actions,target_pi,alpha):
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
    for t in range(ret.shape[1] - 2, - 1, - 1):
        ret[:, t] = rewards[:, t] + gamma * (target_qs[:, t + 1] * (1 - terminated[:, t])-alpha.exp() * target_pi[:,t+1]* (1 - terminated[:, t]))
    return ret[:, 0:-1]


