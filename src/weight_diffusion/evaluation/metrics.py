import numpy as np
from einops import repeat


def calculate_prompt_alignment(prompt_actual, prompt_targets):
    """
    Calculate the prompt alignment based on the R2 statistic
    :param evaluation_dict: dictionary containing actual statistics achieved by a sampled checkpoint
    :param targets: dictionary containing actual statistics in prompt
    :return:
    R2 of the values and targetds
    """
    rsq = []
    for k in prompt_targets.keys():
        print('Calculating R2 for key:', k)
        rsq += calculate_r2_prompt_alignment(prompt_actual[k], prompt_targets[k])
        print('r2', rsq)
    mean_rsq = np.mean(rsq)
    return mean_rsq

def calculate_r2_prompt_alignment(y,t):
    """
    y: prediction
    t: target
    
    """
    # get t_mean: tensor with mean values of t, same shape as t
    t_mean = repeat(t.mean(dim=0), "d -> n d", n=t.shape[0])
    # compute error to t_mean
    e_mean = t-t_mean
    # compute mse(t,t_mean)
    l_mean = torch.einsum('ij,ij ->',e_mean,e_mean)/e_mean.numel()
    # compute prediction error
    e = t-y
    # compute prediction mse
    loss = torch.einsum('ij,ij ->',e,e)/e.numel()
    # compute rsq
    rsq = 1 - loss.item()/l_mean.item()
    return rsq
