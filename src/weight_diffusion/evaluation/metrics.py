import numpy as np


def calculate_ldm_prompt_alignment(evaluation_dict, targets):
    """
    Calculate the prompt alignment based on the root mean squared errors of metrics
    :param evaluation_dict: dictionary containing actual statistics achieved by a sampled checkpoint
    :param targets: dictionary containing actual statistics in prompt
    :return:
    Root mean squared error of metrics' differences
    """
    squared_errors = []
    for k in evaluation_dict.keys():
        squared_errors += [(evaluation_dict[k] - targets[k]) ** 2]
    mse = np.mean(squared_errors)
    return np.sqrt(mse)
