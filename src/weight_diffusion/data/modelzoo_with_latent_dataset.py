import csv
import json
import os
from pathlib import Path
from typing import Tuple, List, Dict, Union, Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torch import nn
from tqdm import tqdm

from weight_diffusion.data.data_utils.helper import get_flat_params
from weight_diffusion.data.data_utils.helper import perform_train_test_validation_split
from weight_diffusion.data.data_utils.normalization import get_normalizer
from weight_diffusion.data.data_utils.permutation import Permutation


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
            encoder: nn.Module,
            device: torch.device,
            dataset_split_ratios: List[float] = None,
            openai_coefficient: float = 4.185,
            normalizer_name="openai",
            use_permutation: bool = True,
            permute_layers: Union[List[int], str] = "all",
            number_of_permutations: int = 10,
            permutation_mode="random"
    ):
        super().__init__()
        self.data_dir = data_dir
        self.use_permutation = use_permutation
        self.split = split

        self.device = device
        self.encoder = encoder

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

        # Get all model directories and perform train_val_test split
        model_directory_paths = perform_train_test_validation_split(
            # TODO Remove [:15]
            list_to_split=get_all_directories_for_a_path(data_dir)[:15],
            dataset_split_ratios=self.dataset_split_ratios,
            split=self.split,
        )

        # Load model checkpoints
        self.checkpoints_dict = {}  # Dict[Model Nr.][Checkpoint Nr.] = model_state_dict
        self.checkpoint_metrics_dict = {}
        self.model_count = 0
        for model_directory in tqdm(model_directory_paths, desc="Loading Models"):
            (
                self.checkpoints_dict[self.model_count],
                self.checkpoint_metrics_dict[self.model_count],
            ) = self._parse_model_directory(model_directory)
            self.model_count += 1

        # Create dataset index
        self.index_dict = {}  # Dict[index nr.] = (model Nr., Checkpoint Nr.)
        self.count = 0
        for model_key, checkpoints_dict in tqdm(
                self.checkpoints_dict.items(), desc="Indexing dataset"
        ):
            n_checkpoints = len(checkpoints_dict.keys())
            for i in range(n_checkpoints):
                self.index_dict[self.count] = (model_key, i)
                self.count += 1

        # Define which model metric should be used for training
        self.checkpoint_property_of_interest = checkpoint_property_of_interest

        # Initialize dataset attributes
        model_key, checkpoint_key = self.index_dict[0]
        self.data_sample = self.checkpoints_dict[model_key][checkpoint_key]
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

        # Prepare permutation
        if permute_layers == "all":
            permute_layers = [
                layer_id
                for layer_id, layer_type in self.layer_list[:-1]  # Can't permute last layer
            ]
        self.permute_layers = permute_layers

        # Initialize permutation
        if use_permutation:
            self.permutation = Permutation(
                checkpoint_sample=self.data_sample[0],
                layers_to_permute=self.permute_layers,
                layer_lst=self.layer_list,
                number_of_permutations=number_of_permutations,
                permutation_mode=permutation_mode,
            )

        # TODO del self.encoder

    def __getitem__(self, index) -> T_co:
        model_key, checkpoint_key = self.index_dict[index]
        checkpoint = self.checkpoints_dict[model_key][checkpoint_key]
        if self.use_permutation:
            checkpoint = self.permutation.permute_checkpoint(checkpoint=checkpoint)
        return checkpoint

    def __len__(self):
        return self.count

    # parsing methods
    def _parse_checkpoint_directory(
            self, checkpoint_directory, model_directory
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:

        checkpoint_path = os.path.join(
            self.data_dir, model_directory, checkpoint_directory, "checkpoints"
        )
        checkpoint = torch.load(checkpoint_path)
        # TODO Add permutations self.permutation.permute_checkpoint(checkpoint)
        flattened_checkpoint = get_flat_params(checkpoint)

        # Store Min and Max parameter value for later
        min_parameter_in_cp, max_parameter_in_cp = (
            getattr(flattened_checkpoint, f)().item() for f in ["min", "max"]
        )
        if self.min_parameter_value > min_parameter_in_cp:
            self.min_parameter_value = min_parameter_in_cp
        if self.max_parameter_value < max_parameter_in_cp:
            self.max_parameter_value = max_parameter_in_cp

        checkpoint_latent_rep_path = os.path.join(
            self.data_dir, model_directory, checkpoint_directory, "checkpoints_latent_rep"
        )

        # Fetch latent representation from storage or generate a new one and save it
        if os.path.isfile(checkpoint_latent_rep_path):
            checkpoint_latent_rep = torch.load(checkpoint_latent_rep_path)
        else:
            # Need to convert from checkpoint to a list of checkpoints
            # TODO torch.unsqueeze(x, dim=0)
            flattened_checkpoint = torch.tensor([flattened_checkpoint.tolist()])
            with torch.no_grad():
                checkpoint_latent_rep, _ = self.encoder.forward(flattened_checkpoint.to(self.device))
            torch.save(checkpoint_latent_rep, checkpoint_latent_rep_path)

        return int(checkpoint_directory[-6:]), checkpoint, checkpoint_latent_rep

    def _parse_model_directory(
            self, model_directory
    ) -> Tuple[Dict[int, Tuple[torch.Tensor, torch.Tensor]], Dict[int, Dict[str, Union[str, bool, float]]]]:
        model_directory_dict = {}
        model_progress_dict = parse_progress_csv(
            path_to_progress_csv=self.data_dir.joinpath(model_directory).joinpath(
                "progress.csv"
            )
        )

        for checkpoint_directory in get_all_directories_for_a_path(
                self.data_dir.joinpath(model_directory)
        ):
            checkpoint_key, checkpoint, checkpoint_latent_rep = self._parse_checkpoint_directory(
                checkpoint_directory,
                model_directory,
            )
            if checkpoint_key in model_progress_dict.keys():
                model_directory_dict[checkpoint_key] = (checkpoint, checkpoint_latent_rep)
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
