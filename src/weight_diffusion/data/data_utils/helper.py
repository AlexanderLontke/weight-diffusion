import torch
from typing import List, Any, Collection
import json
from pathlib import Path
from ghrp.model_definitions.def_net import NNmodule
import copy


def get_flat_params(state_dict):
    parameters = []
    for parameter in state_dict.values():
        parameters.append(parameter.flatten())
    return torch.cat(parameters)


def generate_checkpoints_from_weights(weights, model_config, layer_lst):
    # init model
    base_model = NNmodule(model_config)
    checkpoint_base = base_model.model.state_dict()
    # iterate over samples
    for idx in range(weights.shape[0]):
        # slice
        weight_vector = weights[idx, :].clone()
        # checkpoint
        chkpt = vector_to_checkpoint(
            checkpoint=checkpoint_base,
            vector=weight_vector,
            layer_lst=layer_lst,
            use_bias=True,
        )

        return chkpt


def vector_to_checkpoint(checkpoint, vector, layer_lst, use_bias=False):
    # assert checkpoints and vector size match
    checkpoint = copy.deepcopy(checkpoint)
    testvector = vectorize_checkpoint(checkpoint, layer_lst, use_bias=use_bias)
    assert len(testvector) == len(
        vector
    ), f"checkpoint and test vector lengths dont match - {len(testvector)} vs {len(vector)} "

    # transformation
    idx_start = 0
    for layer, _ in layer_lst:
        # weights
        # load old/sample weights
        weight = checkpoint.get(f"module_list.{layer}.weight", torch.empty(0))
        # flatten sample weights to get dimension
        tmp = weight.flatten()
        # get end index
        idx_end = idx_start + tmp.shape[0]
        #         print(f"idx_start = {idx_start} - {idx_end}")
        #         print("old weight")
        #         print(weight)
        # slice incoming vector and press it in corresponding shape
        weight_new = vector[idx_start:idx_end].view(weight.shape)
        #         print("new weight")
        #         print(weight_new)
        # update dictionary
        checkpoint[f"module_list.{layer}.weight"] = weight_new.clone()
        # update index
        idx_start = idx_end

        # bias

        if use_bias:
            # bias
            # load old/sample bias
            bias = checkpoint.get(f"module_list.{layer}.bias", torch.empty(0))
            # flatten sample bias to get dimension
            tmp = bias.flatten()
            # get end index
            idx_end = idx_start + tmp.shape[0]
            #             print(f"idx_start = {idx_start} - {idx_end}")
            #             print(bias)
            # slice incoming vector and press it in corresponding shape
            bias_new = vector[idx_start:idx_end].view(bias.shape)
            #             print(bias_new)
            # update dictionary
            checkpoint[f"module_list.{layer}.bias"] = bias_new.clone()
            # update index
            idx_start = idx_end

    return checkpoint


def vectorize_checkpoint(checkpoint, layer_lst, use_bias=False):
    # initialize data list
    ddx = []
    # loop over all layers in layer_lst
    for _, (layer, _) in enumerate(layer_lst):
        # load input data and label from data
        weight = checkpoint.get(f"module_list.{layer}.weight", torch.empty(0))
        bias = checkpoint.get(f"module_list.{layer}.bias", torch.empty(0))

        # put weights to be considered in a list for post-processing
        ddx.append(weight)
        if use_bias:
            ddx.append(bias)

    vec = torch.Tensor()
    for idx in ddx:
        vec = torch.cat((vec, idx.view(-1)))
    ddx = vec

    return ddx


def get_param_sizes(state_dict):
    return torch.tensor([p.numel() for p in state_dict.values()], dtype=torch.long)


def perform_train_test_validation_split(
    list_to_split: List[Any], dataset_split_ratios: Collection[float], split: str
):
    # Perform Train-Test(-Validation) Split
    assert sum(dataset_split_ratios) == 10, "Dataset split ratios do not add up to 10"

    # Convert to absolute amounts
    n = len(list_to_split)
    dataset_split_ratios = [int(ratio / 10.0 * n) for ratio in dataset_split_ratios]

    # Check if validation split is included or not
    if len(dataset_split_ratios) == 2:
        number_of_training_items, number_of_testing_items = dataset_split_ratios
        number_of_validation_items = None
    elif len(dataset_split_ratios) == 3:
        (
            number_of_training_items,
            number_of_validation_items,
            number_of_testing_items,
        ) = dataset_split_ratios
    else:
        print(
            f"List giving dataset split ratios must have length 2 or three but"
            f"given ratios had length {len(dataset_split_ratios)}. "
            f"Loading 100% of dataset"
        )
        return list_to_split

    # Perform split
    if split == "train":
        list_to_split = list_to_split[:number_of_training_items]
    elif split == "test":
        list_to_split = list_to_split[-number_of_testing_items:]
    elif split == "val":
        if number_of_validation_items:
            list_to_split = list_to_split[
                number_of_training_items : number_of_training_items
                + number_of_validation_items
            ]
        else:
            raise ValueError(
                "validation split requested, but only two split ratios provided."
            )
    return list_to_split
