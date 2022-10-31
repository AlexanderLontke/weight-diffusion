import torch


def get_flat_params(state_dict):
    parameters = []
    for parameter in state_dict.values():
        parameters.append(parameter.flatten())
    return torch.cat(parameters)


def get_param_sizes(state_dict):
    return torch.tensor([p.numel() for p in state_dict.values()], dtype=torch.long)
