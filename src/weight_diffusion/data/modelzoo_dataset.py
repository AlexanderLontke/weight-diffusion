import os
import json
import csv
from typing import Any, Tuple, List, Dict, Union
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from torch.utils.data.dataset import T_co

from weight_diffusion.data.data_utils.normalization import get_normalizer
from weight_diffusion.data.data_utils.helper import get_flat_params
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
        result = [directory for directory in result if not directory.startswith(".")]
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
) -> Dict[int, Dict[str, Union[str, int, float]]]:
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
        openai_coeff: float,
        normalizer_name="openai",
        use_permutation: bool = True,
        permute_layers: Union[List[int], str] = "all",
        number_of_permutations: int = 10,
        permutation_mode="random",
    ):
        super().__init__()
        self.min_parameter_value = np.Inf
        self.max_parameter_value = -np.Inf
        self.data_dir = data_dir
        self.use_permutation = use_permutation

        # Load model architecture
        with open(data_dir.joinpath("index_dict.json")) as model_json:
            self.model_config = json.load(model_json)
        self.layer_list = self.model_config["layer"]

        # Define which model metric should be used for training
        self.checkpoint_property_of_interest = checkpoint_property_of_interest

        # Define normalizer
        self.normalizer_name = normalizer_name
        self.openai_coeff = openai_coeff
        self.normalizer = get_normalizer(
            self.normalizer_name, openai_coeff=self.openai_coeff
        )

        # Load model checkpoints
        self.checkpoints_dict = {}  # Dict[Model Nr.][Checkpoint Nr.] = model_state_dict
        for model_directory in tqdm(
            get_all_directories_for_a_path(data_dir)[:15], desc="Loading Models"
        ):
            self.checkpoints_dict[model_directory] = self._parse_model_directory(
                model_directory
            )

        # Create dataset index
        self.index_dict = (
            {}
        )  # Dict[index nr.] = (model Nr., (Checkpoint Nr. i, Checkpoint Nr.j), where i < j
        self.count = 0
        for model_key, checkpoints_dict in tqdm(
            self.checkpoints_dict.items(), desc="Indexing dataset"
        ):
            checkpoints_list = list(checkpoints_dict.keys())
            n_checkpoints = len(checkpoints_list)
            # Index not only for a single checkpoint but all possible
            # combinations of start and end checkpoint pairs
            for i in range(n_checkpoints):
                for j in range(i + 1, n_checkpoints):
                    self.index_dict[self.count] = (model_key, (i, j))
                    self.count += 1

        # Prepare permutation
        if permute_layers == "all":
            permute_layers = [
                layer_id for layer_id, layer_type in self.layer_list[:-1]
            ]  # Can't permute last layer
        self.permute_layers = permute_layers

        model_key, (checkpoint_key, _) = self.index_dict[0]
        (self.data_sample, _) = self.checkpoints_dict[model_key][checkpoint_key]
        if use_permutation:
            self.permutation = Permutation(
                checkpoint_sample=self.data_sample,
                layers_to_permute=self.permute_layers,
                layer_lst=self.layer_list,
                number_of_permutations=number_of_permutations,
                permutation_mode=permutation_mode,
            )

    def __getitem__(self, index) -> T_co:
        model_key, checkpoint_keys_01 = self.index_dict[index]
        checkpoints_01 = tuple(
            self.checkpoints_dict[model_key][checkpoint]
            for checkpoint in checkpoint_keys_01
        )

        if self.use_permutation:
            checkpoints_01 = (
                (self.permutation.permute_checkpoint(checkpoint=checkpoint), loss)
                for checkpoint, loss in checkpoints_01
            )

        return tuple(checkpoints_01)

    def __len__(self):
        return self.count

    # parsing methods
    def _parse_checkpoint_directory(
        self, checkpoint_directory, model_directory
    ) -> Tuple[int, torch.Tensor]:
        checkpoint_path = os.path.join(
            self.data_dir, model_directory, checkpoint_directory, "checkpoints"
        )
        checkpoint = torch.load(checkpoint_path)
        flattened_checkpoint = get_flat_params(checkpoint)

        # Store Min and Max parameter value for later
        min_parameter_in_cp, max_parameter_in_cp = (
            getattr(flattened_checkpoint, f)().item() for f in ["min", "max"]
        )
        if self.min_parameter_value > min_parameter_in_cp:
            self.min_parameter_value = min_parameter_in_cp
        if self.max_parameter_value < max_parameter_in_cp:
            self.max_parameter_value = max_parameter_in_cp

        return int(checkpoint_directory[-6:]), checkpoint

    def _parse_model_directory(
        self, model_directory
    ) -> Dict[int, Tuple[torch.Tensor, Union[str, int, float]]]:
        model_directory_dict = {}
        model_progress_dict = parse_progress_csv(
            path_to_progress_csv=self.data_dir.joinpath(model_directory).joinpath(
                "progress.csv"
            )
        )
        for checkpoint_directory in get_all_directories_for_a_path(
            self.data_dir.joinpath(model_directory)
        ):
            checkpoint_key, checkpoint = self._parse_checkpoint_directory(
                checkpoint_directory,
                model_directory,
            )
            if checkpoint_key in model_progress_dict.keys():
                checkpoint_metrics = model_progress_dict[checkpoint_key]
                model_directory_dict[checkpoint_key] = (
                    checkpoint,
                    checkpoint_metrics[self.checkpoint_property_of_interest],
                )
            else:
                continue
        return model_directory_dict
