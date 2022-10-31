from pathlib import Path

from weight_diffusion.data.modelzoo_dataset import ModelZooDataset
from weight_diffusion.data.data_utils.normalization import get_normalizer
from weight_diffusion.data.data_utils.helper import get_param_sizes, get_flat_params

# TODO Permutation
# TODO Normalization


class GptDataset(ModelZooDataset):
    def __init__(self, data_dir: Path, checkpoint_property_of_interest: str, openai_coeff: float,
                 normalizer_name="openai"):
        super().__init__(
            data_dir=data_dir,
            checkpoint_property_of_interest=checkpoint_property_of_interest
        )
        (sample, _), (_, _) = super().__getitem__(0)
        self.parameter_sizes = get_param_sizes(sample).long().tolist()
        self.parameter_names = list(sample.keys())

        self.normalizer_name = normalizer_name
        self.openai_coeff = openai_coeff
        self.normalizer = get_normalizer(
            self.normalizer_name,
            openai_coeff=self.openai_coeff
        )

    def __getitem__(self, index):
        (checkpoint0, loss0), (checkpoint1, loss1) = super().__getitem__(index)
        return {
            "parameters_0": get_flat_params(checkpoint0),
            "parameters_1": get_flat_params(checkpoint1),
            f"{self.checkpoint_property_of_interest}_0": loss0,
            f"{self.checkpoint_property_of_interest}_1": loss1,
        }

    def normalize(self, weights):
        return self.normalizer.normalize(weights)

    def unnormalize(self, normalized_weights):
        return self.normalizer.unnormalize(normalized_weights)