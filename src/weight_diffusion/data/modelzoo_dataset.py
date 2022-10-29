from typing import Any, Tuple, List, Dict, Union
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
import random
import copy

import itertools
from math import factorial
import glob

from torch.utils.data.dataset import T_co

from checkpoints_to_datasets.permute_checkpoint import permute_checkpoint
from checkpoints_to_datasets.map_to_canonical import sort_layers_checkpoint
from checkpoints_to_datasets.dataset_auxiliaries import (
    vectorize_checkpoint,
    add_noise_to_checkpoint,
    printProgressBar,
    vector_to_checkpoint,
)
from checkpoints_to_datasets.random_erasing import RandomErasingVector
import ray
from checkpoints_to_datasets.progress_bar import ProgressBar

import os
import json
import csv


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
            progress_dict[count] = {k: v for k, v in zip(header, row)}
            count += 1
    return progress_dict


class ModelZooDataset(Dataset):
    def __init__(self, data_dir: Path, checkpoint_property_of_interest: str):
        super().__init__()
        self.data_dir = data_dir
        self.checkpoint_property_of_interest = checkpoint_property_of_interest
        with open(data_dir.joinpath("index_dict.json")) as model_json:
            self.model_config = json.load(model_json)
        # Load model checkpoints
        self.checkpoints_dict = {}
        for model_directory in tqdm(
                get_all_directories_for_a_path(data_dir)[:15], desc="Loading Models"
        ):
            self.checkpoints_dict[model_directory] = self._parse_model_directory(
                model_directory
            )

        # Create dataset index
        self.index_dict = {}
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

    def __getitem__(self, index) -> T_co:
        model_key, checkpoint_step_tuple = self.index_dict[index]
        return (
            self.checkpoints_dict[model_key][step] for step in checkpoint_step_tuple
        )

    def __len__(self):
        return self.count

    # parsing methods
    def _parse_checkpoint_directory(
            self, checkpoint_directory, model_directory
    ) -> Tuple[int, torch.Tensor]:
        checkpoint_path = os.path.join(
            self.data_dir, model_directory, checkpoint_directory, "checkpoints"
        )
        return int(checkpoint_directory[-6:]), torch.load(checkpoint_path)

    def _parse_model_directory(
            self, model_directory
    ) -> Dict[int, Tuple[torch.Tensor, Union[str, int, float]]]:
        model_directory_dict = {}
        model_progress_dict = parse_progress_csv(
            path_to_progress_csv=self.data_dir.joinpath(model_directory).joinpath("progress.csv")
        )
        for checkpoint_directory in get_all_directories_for_a_path(
                self.data_dir.joinpath(model_directory)
        ):
            checkpoint_key, checkpoint = self._parse_checkpoint_directory(
                checkpoint_directory, model_directory,
            )
            if checkpoint_key in model_progress_dict.keys():
                checkpoint_metrics = model_progress_dict[checkpoint_key]
                model_directory_dict[checkpoint_key] = (
                    checkpoint, checkpoint_metrics[self.checkpoint_property_of_interest]
                )
            else:
                continue
        return model_directory_dict
