import copy
import itertools
import random
from math import factorial
from typing import List

from weight_diffusion.data.data_utils.helper import get_flat_params

import torch


def permute_checkpoint(checkpoint, layer_lst, permute_layers, permutation_idxs_dct):
    """
    Permutes kernels / neurons in model given as torch checkpoint.
    All layers to be considered have to be indicated in layer_lst, with entries as tuple (index, "type"), e.g. (3,"conv2d")
    Layers to be permuted have to be indicated in permute_layers, a list with layer index.
    The new order/index per layer is given in permutation_idx_dct, with one entry per layer
    """
    # compute kernel size
    layer_kernels = []
    for kdx in permute_layers:
        layer_type = [y for (x, y) in layer_lst if x == kdx][0]
        if layer_type == "conv2d":
            weights = checkpoint.get(f"module_list.{kdx}.weight", torch.empty(0))
            kernels = weights.shape[0]
            layer_kernels.append(kernels)
        elif layer_type == "fc":
            weights = checkpoint.get(f"module_list.{kdx}.weight", torch.empty(0))
            kernels = weights.shape[0]
            layer_kernels.append(kernels)
        else:
            print(
                f"permutations for layers of type {layer_type} are not yet implemented"
            )
            raise NotImplementedError

    # apply permutation
    for ldx, (layer, type) in enumerate(layer_lst):
        # load input data and label from self.data
        weight = checkpoint.get(f"module_list.{layer}.weight", torch.empty(0))
        bias = checkpoint.get(f"module_list.{layer}.bias", None)

        # check if permutations are applied data
        if layer in permute_layers:
            # get type of permuted layer
            # layer_type = [y for (x, y) in layer_lst if x == layer]
            kdx = permute_layers.index(layer)
            # create index 0,...,n-1
            index_old = list(range(layer_kernels[kdx]))

            # get type and data of following layer
            try:
                (layer_next, layer_next_type) = layer_lst[ldx + 1]
            except Exception as e:
                print(e)
                print(
                    f"permuting layer {ldx}, "
                    f"there was en error loading the following layer. "
                    f"your probably trying to permute the last layer, which doesn't work."
                )
                continue

            # load next layers weights
            weight_next = checkpoint.get(
                f"module_list.{layer_next}.weight", torch.empty(0)
            )

            # permute current layer
            # get new index of layer and get permutation
            index_new = permutation_idxs_dct[f"layer_{layer}"]
            if type == "conv2d":
                # create new input cnn_layer
                weight_new = torch.zeros_like(weight)
                weight_new[index_old, :, :, :] = weight[index_new, :, :, :]
                if bias is not None:
                    bias_new = bias[index_new]
                # create new output cnn_layer
            elif type == "fc":
                # permute by first axis
                # input
                weight_new = weight[index_new]
                if bias is not None:
                    bias_new = bias[index_new]
                # output
            else:
                raise NotImplementedError

            # permute following layer with transposed
            # permute followup layer also by input channels. (weights only, bias only affects output channels...)
            if layer_next_type == "conv2d":
                # permute followup layer 2nd axis
                # input
                weight_next_new = torch.zeros_like(weight_next)
                weight_next_new[:, index_old, :, :] = weight_next[:, index_new, :, :]
            elif layer_next_type == "fc":
                # fc input dimensions correspond to channels
                if weight_next.shape[1] == layer_kernels[kdx]:
                    weight_next_new = torch.zeros_like(weight_next)
                    # permute by second axis
                    weight_next_new[:, index_old] = weight_next[:, index_new]

                else:
                    weight_next_new = torch.zeros_like(weight_next)
                    # assume: output of conv2d is flattened.
                    # flatting happens within channels first.
                    # channel outputs must be devider of fc dim[1]
                    assert (
                        int(weight_next.shape[1]) % int(layer_kernels[kdx]) == 0
                    ), "divider must be of type integer, dimensions don't add up"

                    fc_block_length = int(
                        int(weight_next.shape[1]) / int(layer_kernels[kdx])
                    )
                    # iterate over blocks and change indices accordingly
                    for idx_old, idx_new in zip(index_old, index_new):
                        for fcdx in range(fc_block_length):
                            offset_old = idx_old * fc_block_length + fcdx
                            offset_new = idx_new * fc_block_length + fcdx
                            weight_next_new[:, offset_old] = weight_next[:, offset_new]

                # input
            else:
                raise NotImplementedError

            # update weights in checkpoint
            checkpoint[f"module_list.{layer}.weight"] = weight_new
            if bias is not None:
                checkpoint[f"module_list.{layer}.bias"] = bias_new
            # overwrite next layers weights
            checkpoint[f"module_list.{layer_next}.weight"] = weight_next_new
    return checkpoint


def create_random_permutation_index(
    layers_to_permute: List[int], layer_kernels: List[int], number_of_permutations: int
):
    permutations_dct = {}
    for kdx, layer in enumerate(layers_to_permute):
        # initialize empty list
        permutations_dct[f"layer_{layer}"] = []
        old_layer_index = list(range(layer_kernels[kdx]))

        # figure out layer size
        theoretical_permutations = factorial(len(old_layer_index))
        no_perms_this_layer = min(
            theoretical_permutations, number_of_permutations
        ) // len(layers_to_permute)
        print(
            f"compute {no_perms_this_layer} random permutations for layer {kdx} - {layer}"
        )
        for pdx in range(no_perms_this_layer):
            index_new = copy.deepcopy(old_layer_index)
            random.shuffle(index_new)
            # append list of new index to list per layer
            permutations_dct[f"layer_{layer}"].append(list(index_new))
    return permutations_dct


def create_complete_permutation_index(
    layers_to_permute: List[int], layer_kernels: List[int]
):
    permutations_dct = {}
    for kdx, layer in enumerate(layers_to_permute):
        # initialize empty list
        permutations_dct[f"layer_{layer}"] = []
        old_layer_index = list(range(layer_kernels[kdx]))

        # iterate over all complete combinations of index_old
        for index_new in itertools.permutations(old_layer_index, len(old_layer_index)):
            # append list of new index to list per layer
            permutations_dct[f"layer_{layer}"].append(list(index_new))
    return permutations_dct


class Permutation:
    def __init__(
        self,
        layers_to_permute,
        checkpoint_sample,
        layer_lst,
        number_of_permutations: int,
        permutation_mode: str,
    ):
        """
        This function creates self.permutations_dct, a dictionary with mappings for all permutations.
        it contains keys for all layers, with lists as values. the lists contain one mapping per permutation.
        """
        self.layers_to_permute = layers_to_permute
        self.layer_lst = layer_lst
        self.number_of_permutations = number_of_permutations
        self.checkpoint_sample = checkpoint_sample
        self.permutation_mode = permutation_mode

        # check # of kernels for first data entry
        self.layer_kernels = self.get_permute_layer_sizes(checkpoint_sample)

        # dict of list for every layer, with lists of index permutations
        self.permutations_dct = {}
        self.permutations_dct = self.create_permutation_index(
            permutation_mode=permutation_mode,
        )
        self.permutations_index = self.get_permutation_map()

        print("prepare permutation dicts")
        self.permutations_dct_lst = self.prepare_permutations_dct_list()

    def get_permute_layer_sizes(self, checkpoint_sample):
        """
        Computes the size of each layer
        :param checkpoint_sample: State dict sample out of the data set
        :return:
        """
        layer_kernels = []
        for kdx in self.layers_to_permute:
            layer_type = [y for (x, y) in self.layer_lst if x == kdx][0]
            assert layer_type in [
                "conv2d",
                "fc",
            ], f"permutations for layers of type {layer_type} are not yet implemented"
            weights = checkpoint_sample.get(f"module_list.{kdx}.weight", torch.empty(0))
            kernels = weights.shape[0]
            layer_kernels.append(kernels)
        return layer_kernels

    def create_permutation_index(self, permutation_mode: str):
        """
        Creates the permutation mapping for each layer based on the designated mode
        :return:
        """
        assert permutation_mode in ["complete", "random"]

        # Mode 1: precompute all permutations
        if permutation_mode == "complete":
            return create_complete_permutation_index(
                layers_to_permute=self.layers_to_permute,
                layer_kernels=self.layer_kernels,
            )
        elif permutation_mode == "random":
            return create_random_permutation_index(
                layers_to_permute=self.layers_to_permute,
                layer_kernels=self.layer_kernels,
                number_of_permutations=self.number_of_permutations,
            )

    def get_permutation_map(self):
        combinations = []
        # Mode 1: precompute all permutations
        if self.permutation_mode == "complete":
            # get #of permutations per layer
            for kdx, layer in enumerate(self.layers_to_permute):
                n_perms = len(self.permutations_dct[f"layer_{layer}"])
                index_kdx = list(range(n_perms))
                combinations.append(index_kdx)
            # get all combinations of permutation indices
            combinations = list(itertools.product(*combinations))
            # random shuffle combinations around
            random.shuffle(combinations)
        # pick only random permutations from indices prepared
        elif self.permutation_mode == "random":
            for pdx in range(self.number_of_permutations):
                # random pick index to permutation in perm_dict for each layer
                combination_single = []
                for kdx, layer in enumerate(self.layers_to_permute):
                    # pick random index list for that layer
                    n_perms = len(self.permutations_dct[f"layer_{layer}"])
                    index_kdx = random.choice(list(range(n_perms)))
                    combination_single.append(index_kdx)
                # append tuple to list
                combinations.append(tuple(combination_single))
        else:
            raise NotImplementedError
        print(f"prepared {len(combinations)} permutations")
        return combinations

    def prepare_permutations_dct_list(self):
        """
        re-order the index in one stand-alone dict per permutation, so that the dicts don't have to be put together at runtime.
        the list get's an index and returns a dict with all necessary indices.
        """
        permutations_dct_lst = []
        # compute one dict for the number of wanted permutations
        for pdx in range(self.number_of_permutations):
            prmt_dct = {}
            for kdx, layer in enumerate(self.layers_to_permute):
                # get permutation index for permutation pdx and layer kdx
                permutation_idx = self.permutations_index[pdx][kdx]
                prmt_dct[f"layer_{layer}"] = self.permutations_dct[f"layer_{layer}"][
                    permutation_idx
                ]
            permutations_dct_lst.append(copy.deepcopy(prmt_dct))

        return permutations_dct_lst

    def permute_checkpoint(self, checkpoint):
        if self.number_of_permutations > 0:
            # get permutation index -> pick random number from available perms

            pdx = random.randint(0, self.number_of_permutations - 1)
            # perform actual permutation

            # get perm dict
            prmt_dct = self.permutations_dct_lst[pdx]

            # apply permutation on input data
            checkpoint = permute_checkpoint(
                copy.deepcopy(checkpoint),
                self.layer_lst,
                self.layers_to_permute,
                prmt_dct,
            )

        # append data to permuted list
        return checkpoint

    def permute_checkpoint(self, checkpoint, pdx):
        # pdx = -1 returns original checkpoint
        if self.number_of_permutations > 0 and pdx > -1:
            # get perm dict
            prmt_dct = self.permutations_dct_lst[pdx]

            # apply permutation on input data
            checkpoint = permute_checkpoint(
                copy.deepcopy(checkpoint),
                self.layer_lst,
                self.layers_to_permute,
                prmt_dct,
            )

        # append data to permuted list
        return checkpoint

    def get_all_permutations_for_checkpoint(self, checkpoint):
        permuted_checkpoints = [get_flat_params(checkpoint)]
        for prmt_dct in self.permutations_dct_lst:
            # apply permutation on input data
            a = permute_checkpoint(
                    copy.deepcopy(checkpoint),
                    self.layer_lst,
                    self.layers_to_permute,
                    prmt_dct,
                )

            b = get_flat_params(a)

            permuted_checkpoints.append(b)

        # append data to permuted list
        return permuted_checkpoints
