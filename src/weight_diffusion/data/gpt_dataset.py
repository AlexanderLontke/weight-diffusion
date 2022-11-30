import os

import torch
from pathlib import Path
from tqdm import tqdm
from weight_diffusion.data.modelzoo_dataset import ModelZooDataset
from weight_diffusion.data.data_utils.helper import get_param_sizes, get_flat_params
from weight_diffusion.data.data_utils.permutation import Permutation


class GptDataset(ModelZooDataset):
    def __init__(
            self, data_dir: Path, checkpoint_property_of_interest: str, split: str, **kwargs
    ):
        super().__init__(
            data_dir=data_dir,
            checkpoint_property_of_interest=checkpoint_property_of_interest,
            split=split,
            **kwargs,
        )

        # TODO remove these two?
        # self.parameter_sizes = get_param_sizes(self.data_sample).long().tolist()
        # self.parameter_names = list(self.data_sample.keys())

        # Create dataset index
        self.index_dict = {}  # Dict[index nr.] = (model Nr., (Checkpoint Nr. i, Checkpoint Nr.j), where i < j
        self.count = 0
        for model_key, checkpoints_dict in tqdm(
                self.checkpoints_dict.items(), desc="Indexing G.pt dataset"
        ):
            n_checkpoints = len(checkpoints_dict.keys())

            # Index not only for a single checkpoint but all possible
            # combinations of start and end checkpoint pairs
            for checkpoint_i_key in range(n_checkpoints):
                for permutation_i_key in range(self.number_of_permutations):
                    for checkpoint_j_key in range(checkpoint_i_key + 1, n_checkpoints):
                        for permutation_j_key in range(self.number_of_permutations):
                            self.index_dict[self.count] = (model_key,
                                                           checkpoint_i_key,
                                                           permutation_i_key,
                                                           checkpoint_j_key,
                                                           permutation_j_key)
                            self.count += 1

    def __getitem__(self, index):
        # TODO would a list be faster than a dict?
        model_key, checkpoint_i_key, permutation_i_key, checkpoint_j_key, permutation_j_key = self.index_dict[index]

        checkpoint_i_dict = self.checkpoints_dict[model_key][checkpoint_i_key]
        checkpoint_i = checkpoint_i_dict[0][permutation_i_key]
        loss_i = checkpoint_i_dict[1]

        checkpoint_j_dict = self.checkpoints_dict[model_key][checkpoint_j_key]
        checkpoint_j = checkpoint_j_dict[0][permutation_j_key]
        loss_j = checkpoint_j_dict[1]

        return {
            "parameters_0": self.normalize(get_flat_params(checkpoint_i)),
            "parameters_1": self.normalize(get_flat_params(checkpoint_j)),
            f"{self.checkpoint_property_of_interest}_0": loss_i,
            f"{self.checkpoint_property_of_interest}_1": loss_j,
        }

    def normalize(self, weights):
        return self.normalizer.normalize(weights)

    def unnormalize(self, normalized_weights):
        return self.normalizer.unnormalize(normalized_weights)

    def get_run_losses(self, i: int):
        return torch.Tensor(
            [
                self.checkpoint_metrics_dict[i][checkpoint_key][
                    self.checkpoint_property_of_interest
                ]
                for checkpoint_key, _ in self.checkpoint_metrics_dict[i].items()
            ]
        )

    def get_run_network(self, model_key: int, epoch: int = 0):
        return get_flat_params(self.checkpoints_dict[model_key][epoch])

    def get_range(self, normalize: bool):
        min_val, max_val = self.min_parameter_value, self.max_parameter_value
        if normalize:
            # If normalize=True, this returns the range of normalized parameter values
            assert hasattr(
                self, "normalizer"
            ), "normalizer hasn't been instantiated yet"
            min_val, max_val = self.normalizer.get_range(min_val, max_val)
        return min_val, max_val

    def _get_prompt(self, checkpoint_progress, prompt_path):
        return checkpoint_progress[self.checkpoint_property_of_interest]

    def _parse_checkpoint_directory(self, checkpoint_directory, model_directory):
        checkpoint_path = os.path.join(
            self.data_dir, model_directory, checkpoint_directory, "checkpoints"
        )

        checkpoint = torch.load(checkpoint_path)
        self._update_min_max_param_value(checkpoint)

        permuted_checkpoints = self.permutation.get_all_permutations_for_checkpoint(
            checkpoint
        )

        return permuted_checkpoints
