import torch
from pathlib import Path
from tqdm import tqdm
from weight_diffusion.data.modelzoo_dataset import ModelZooDataset
from weight_diffusion.data.data_utils.helper import get_param_sizes, get_flat_params


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
        self.parameter_sizes = get_param_sizes(self.data_sample).long().tolist()
        self.parameter_names = list(self.data_sample.keys())

        # Create dataset index
        self.index_dict = (
            {}
        )  # Dict[index nr.] = (model Nr., (Checkpoint Nr. i, Checkpoint Nr.j), where i < j
        self.count = 0
        for model_key, checkpoints_dict in tqdm(
            self.checkpoints_dict.items(), desc="Indexing G.pt dataset"
        ):
            checkpoints_list = list(checkpoints_dict.keys())
            n_checkpoints = len(checkpoints_list)
            # Index not only for a single checkpoint but all possible
            # combinations of start and end checkpoint pairs
            for i in range(n_checkpoints):
                for j in range(i + 1, n_checkpoints):
                    self.index_dict[self.count] = (model_key, (i, j))
                    self.count += 1

    def __getitem__(self, index):
        model_key, (checkpoint_i, checkpoint_j) = self.index_dict[index]
        checkpoint_0 = self.checkpoints_dict[model_key][checkpoint_i]
        loss_0 = self.checkpoint_metrics_dict[model_key][checkpoint_i][
            self.checkpoint_property_of_interest
        ]
        checkpoint_1 = self.checkpoints_dict[model_key][checkpoint_j]
        loss_1 = self.checkpoint_metrics_dict[model_key][checkpoint_j][
            self.checkpoint_property_of_interest
        ]

        return {
            "parameters_0": self.normalize(get_flat_params(checkpoint_0)),
            "parameters_1": self.normalize(get_flat_params(checkpoint_1)),
            f"{self.checkpoint_property_of_interest}_0": loss_0,
            f"{self.checkpoint_property_of_interest}_1": loss_1,
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
