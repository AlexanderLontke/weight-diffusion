import csv
import json
import os
from pathlib import Path
from typing import Tuple, List, Dict, Union, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
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
        count = 1
        for row in reader:
            progress_dict[count] = {k: _guess_dtype(v) for k, v in zip(header, row)}
            count += 1
    return progress_dict


class ModelZooWithLatentDataset(Dataset):
    def __init__(
            self,
            data_dir: Path,
            checkpoint_property_of_interest: str,
            split: str,
            encoder: nn.Module,
            device: torch.device,
            tokenizer,
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
        self.tokenizer = tokenizer

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
            list_to_split=get_all_directories_for_a_path(data_dir)[15:30],
            dataset_split_ratios=self.dataset_split_ratios,
            split=self.split,
        )

        # Prepare permutation
        if permute_layers == "all":
            permute_layers = [
                layer_id
                for layer_id, layer_type in self.layer_list[:-1]  # Can't permute last layer
            ]
        self.permute_layers = permute_layers

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

        # Delete encoder from memory to free up space
        del self.encoder
        del self.tokenizer

    def __getitem__(self, index) -> T_co:
        model_key, checkpoint_key = self.index_dict[index]
        checkpoint_latent_rep = self.checkpoints_dict[model_key][checkpoint_key][0]
        prompt_latent_rep = self.checkpoints_dict[model_key][checkpoint_key][1]

        return {"checkpoint_latent": checkpoint_latent_rep, "prompt_latent": prompt_latent_rep}

    def __len__(self):
        return self.count

    # parsing methods
    def _parse_checkpoint_directory(
            self, checkpoint_directory, model_directory, model_progress_dict
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:

        checkpoint_path = os.path.join(
            self.data_dir, model_directory, checkpoint_directory, "checkpoints"
        )
        checkpoint = torch.load(checkpoint_path)
        flattened_checkpoint = get_flat_params(checkpoint)
        checkpoint_key = int(checkpoint_directory[-6:])
        if checkpoint_key == 0:
            return 0, torch.Tensor(), torch.Tensor()
        checkpoint_progress = model_progress_dict[checkpoint_key]

        prompt_latent_rep_path = os.path.join(
            self.data_dir, model_directory, checkpoint_directory, "prompt_latent_rep"
        )

        # Fetch latent representation from storage or generate a new one and save it
        if os.path.isfile(prompt_latent_rep_path):
            prompt_latent_rep = torch.load(prompt_latent_rep_path)
        else:
            prompt = f"The training loss is {checkpoint_progress['train_loss']:.4g}. " \
                     f"The training accuracy is {checkpoint_progress['train_acc']:.4g}. " \
                     f"The validation loss is {checkpoint_progress['validation_loss']:.4g}. " \
                     f"The validation accuracy is {checkpoint_progress['validation_acc']:.4g}. " \
                     f"The test loss is {checkpoint_progress['test_loss']:.4g}. " \
                     f"The test accuracy is {checkpoint_progress['test_acc']:.4g}. "

            prompt_latent_rep = self.tokenizer(prompt, return_tensors='pt')["input_ids"]
            torch.save(prompt_latent_rep, prompt_latent_rep_path)

        # Store Min and Max parameter value for later
        min_parameter_in_cp, max_parameter_in_cp = (
            getattr(flattened_checkpoint, f)().item() for f in ["min", "max"]
        )
        if self.min_parameter_value > min_parameter_in_cp:
            self.min_parameter_value = min_parameter_in_cp
        if self.max_parameter_value < max_parameter_in_cp:
            self.max_parameter_value = max_parameter_in_cp

        permuted_checkpoints = self.permutation.get_all_permutations_for_checkpoint(checkpoint)

        for i in range(len(permuted_checkpoints)):

            permuted_checkpoint = permuted_checkpoints[i]
            permuted_checkpoint_path = os.path.join(checkpoint_path + "_p" + str(i))

            if os.path.isfile(permuted_checkpoint_path):
                permuted_checkpoint = torch.load(permuted_checkpoint_path)
            else:
                torch.save(permuted_checkpoint, permuted_checkpoint_path)

            checkpoint_latent_rep_path = os.path.join(
                self.data_dir, model_directory, checkpoint_directory, "checkpoints_latent_rep" + "_p" + str(i)
            )

            # Fetch latent representation from storage or generate a new one and save it
            if os.path.isfile(checkpoint_latent_rep_path):
                checkpoint_latent_rep = torch.load(checkpoint_latent_rep_path)
            else:
                # Need to convert from checkpoint to a list of checkpoints
                with torch.no_grad():
                    checkpoint_latent_rep, _ = self.encoder.forward(
                        torch.unsqueeze(flattened_checkpoint, dim=0).to(self.device))
                torch.save(checkpoint_latent_rep, checkpoint_latent_rep_path)

        return checkpoint_key, checkpoint_latent_rep, prompt_latent_rep

    def _parse_model_directory(
            self, model_directory
    ) -> Tuple[Dict[int, Tuple[torch.Tensor, torch.Tensor]], Dict[int, Dict[str, Union[str, bool, float]]]]:
        model_directory_dict = {}
        model_progress_dict = parse_progress_csv(
            path_to_progress_csv=self.data_dir.joinpath(model_directory).joinpath(
                "progress.csv"
            )
        )

        first = True

        for checkpoint_directory in get_all_directories_for_a_path(
                self.data_dir.joinpath(model_directory)
        ):

            if first:
                checkpoint_path = os.path.join(
                    self.data_dir, model_directory, checkpoint_directory, "checkpoints"
                )
                sample_checkpoint = torch.load(checkpoint_path)

                self.permutation = Permutation(
                    checkpoint_sample=sample_checkpoint,
                    layers_to_permute=self.permute_layers,
                    layer_lst=self.layer_list,
                    number_of_permutations=10,
                    permutation_mode="random",
                )

                first = False

            checkpoint_key, checkpoint_latent_rep, prompt_latent_rep = self._parse_checkpoint_directory(
                checkpoint_directory,
                model_directory,
                model_progress_dict
            )

            if checkpoint_key in model_progress_dict.keys():
                model_directory_dict[checkpoint_key] = (checkpoint_latent_rep, prompt_latent_rep)
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
