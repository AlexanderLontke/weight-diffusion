from pathlib import Path
import torch
from weight_diffusion.data.modelzoo_dataset import ModelZooDataset

# TODO Permutation
# TODO Normalization


def get_flat_params(state_dict):
    parameters = []
    for parameter in state_dict.values():
        parameters.append(parameter.flatten())
    return torch.cat(parameters)


class GptDataset(ModelZooDataset):
    def __init__(self, data_dir: Path, checkpoint_property_of_interest: str):
        super().__init__(
            data_dir=data_dir,
            checkpoint_property_of_interest=checkpoint_property_of_interest
        )

    def __getitem__(self, index):
        (checkpoint0, loss0), (checkpoint1, loss1) = super().__getitem__(index)
        return {
            "parameters_0": get_flat_params(checkpoint0),
            "parameters_1": get_flat_params(checkpoint1),
            f"{self.checkpoint_property_of_interest}_0": loss0,
            f"{self.checkpoint_property_of_interest}_1": loss1,
        }
