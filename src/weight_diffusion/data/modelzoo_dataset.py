import os
import json
import csv
from typing import Tuple, List, Dict, Union, Callable
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from torch.utils.data.dataset import T_co

from weight_diffusion.data.data_utils.normalization import get_normalizer
from weight_diffusion.data.data_utils.helper import get_flat_params
from weight_diffusion.data.data_utils.permutation import Permutation
from weight_diffusion.data.data_utils.helper import perform_train_test_validation_split


def get_all_directories_for_a_path(
    path: Path,
    return_only_directories: bool = True,
    return_no_hidden_directories: bool = True,
):
    result = os.listdir(path)
    if return_only_directories:
        result = [
            directory
            for directory in result
            if os.path.isdir(os.path.join(path, directory))
        ]
    if return_no_hidden_directories:
        result = [
            directory for directory in result if not str(directory).startswith(".")
        ]
    return result


def _guess_dtype(x):
    try:
        return float(x)
    except ValueError:
        try:
            return bool(x)
        except ValueError:
            return x


def parse_progress_csv(
    path_to_progress_csv: Path,
) -> Dict[int, Dict[str, Union[str, bool, float]]]:
    """
    To know what the training/test loss/other metrics looked like at each checkpoint
    we need to parse the progress CSV file which is included for each model
    :return: List containing a dictionary containing metrics for each checkpoint
    """
    with path_to_progress_csv.open("r") as in_file:
        reader = csv.reader(in_file)
        header = reader.__next__()
        progress_dict = {}
        count = 0
        for row in reader:
            progress_dict[count] = {k: _guess_dtype(v) for k, v in zip(header, row)}
            count += 1
    return progress_dict


class ModelZooDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        checkpoint_property_of_interest: str,
        split: str,
        dataset_split_ratios: List[float] = None,
        openai_coefficient: float = 4.185,
        normalizer_name="openai",
        use_permutation: bool = True,
        permute_layers: Union[List[int], str] = "all",
        number_of_permutations: int = 100,
        permutation_mode="random",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.use_permutation = use_permutation
        self.split = split

        # Initialize dataset attributes
        self.min_parameter_value = np.Inf
        self.max_parameter_value = -np.Inf

        # Initialize dataset split ratios
        if not dataset_split_ratios:
            dataset_split_ratios = [10, 0]
        self.dataset_split_ratios = dataset_split_ratios

        # Load model architecture
        with open(data_dir.joinpath("index_dict.json")) as model_json:
            self.model_config = json.load(model_json)
        self.layer_list = self.model_config["layer"]

        self.number_of_permutations = number_of_permutations

        # Prepare permutation
        if permute_layers == "all":
            permute_layers = [
                layer_id
                for layer_id, layer_type in self.layer_list[
                    :-1
                ]  # Can't permute last layer
            ]
        self.permute_layers = permute_layers

        # Define which model metric should be used for training
        self.checkpoint_property_of_interest = checkpoint_property_of_interest

        # Get all model directories and perform train_val_test split
        model_directory_paths = perform_train_test_validation_split(
            # TODO Remove [:100]
            list_to_split=get_all_directories_for_a_path(data_dir),
            dataset_split_ratios=self.dataset_split_ratios,
            split=self.split,
        )

        # Load model checkpoints
        self.checkpoints_dict = {}  # Dict[Model Nr.][Checkpoint Nr.] = model_state_dict
        self.checkpoint_metrics_dict = {}
        self.model_count = 0
        self.first_checkpoint = True
        for model_directory in tqdm(model_directory_paths, desc="Loading Models"):
            # TODO Fix
            if model_directory == 'MNIST':
                    continue
            (
                self.checkpoints_dict[self.model_count],
                self.checkpoint_metrics_dict[self.model_count],
            ) = self._parse_model_directory(model_directory)
            self.model_count += 1

        # Initialize dataset attributes
        # TODO Do we need?
        self.optimal_loss = self._reduce_metrics_data(
            metric=self.checkpoint_property_of_interest,
            aggregation_function=min,
        )

        # Define normalizer
        self.normalizer_name = normalizer_name
        self.openai_coefficient = openai_coefficient
        self.normalizer = get_normalizer(
            self.normalizer_name, openai_coeff=self.openai_coefficient
        )

    def __getitem__(self, index) -> T_co:
        raise NotImplementedError

    def __len__(self):
        return self.count

    def _get_prompt(self, checkpoint_progress, prompt_path):
        raise NotImplementedError

    # parsing methods
    def _parse_checkpoint_directory(
        self, checkpoint_directory, model_directory
    ) -> torch.Tensor:
        raise NotImplementedError

    def _parse_model_directory(
        self, model_directory
    ) -> Dict[int, Tuple[torch.Tensor, any]]:
        model_directory_dict = {}
        model_progress_dict = parse_progress_csv(
            path_to_progress_csv=self.data_dir.joinpath(model_directory).joinpath(
                "progress.csv"
            )
        )

        for checkpoint_directory in get_all_directories_for_a_path(
            self.data_dir.joinpath(model_directory)
        ):
            if self.first_checkpoint:
                self._initialise_permutations(model_directory, checkpoint_directory)
                self.first_checkpoint = False

            checkpoint_key = int(checkpoint_directory[-6:])
            if checkpoint_key in model_progress_dict.keys():
                checkpoint_progress = model_progress_dict[checkpoint_key]

                permutation_dict = self._parse_checkpoint_directory(
                    checkpoint_directory, model_directory
                )

                prompt_path = os.path.join(
                    self.data_dir, model_directory, checkpoint_directory, "prompt"
                )
                prompt = self._get_prompt(checkpoint_progress, prompt_path)

                model_directory_dict[checkpoint_key] = (permutation_dict, prompt)
            else:
                continue
        return model_directory_dict, model_progress_dict

    def _reduce_metrics_data(self, metric: str, aggregation_function: Callable):
        return aggregation_function(
            [
                aggregation_function(
                    [
                        metrics_dict[metric]
                        for _, metrics_dict in checkpoint_dict.items()
                    ]
                )
                for _, checkpoint_dict in self.checkpoint_metrics_dict.items()
            ]
        )

    def _update_min_max_param_value(self, checkpoint):
        flattened_checkpoint = get_flat_params(checkpoint)
        # Store Min and Max parameter value for later
        min_parameter_in_cp, max_parameter_in_cp = (
            getattr(flattened_checkpoint, f)().item() for f in ["min", "max"]
        )
        if self.min_parameter_value > min_parameter_in_cp:
            self.min_parameter_value = min_parameter_in_cp
        if self.max_parameter_value < max_parameter_in_cp:
            self.max_parameter_value = max_parameter_in_cp

    def _initialise_permutations(self, model_directory, checkpoint_directory):
        checkpoint_path = self.data_dir.joinpath(
            model_directory, checkpoint_directory, "checkpoints"
        )
        sample_checkpoint = torch.load(checkpoint_path)
        self.data_sample = sample_checkpoint
        self.permutation = Permutation(
            checkpoint_sample=sample_checkpoint,
            layers_to_permute=self.permute_layers,
            layer_lst=self.layer_list,
            number_of_permutations=self.number_of_permutations,
            permutation_mode="random",
        )
