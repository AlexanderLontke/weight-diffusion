import glob
import torch
import hydra
import omegaconf
from pathlib import Path
from weight_diffusion.data.modelzoo_dataset import ModelZooDataset

from Gpt.utils import (
    construct_loader,
)
from Gpt.meters import TrainMeter

from wrapper_model import WrapperModel


def train(config):
    # TODO needed? Set manual seed
    torch.manual_seed(1337)

    # TODO define training data transformations
    training_data_transformations = []

    # TODO load training set and load it into a custom dataset
    training_dataset = ModelZooDataset(
        data_dir=Path(config.dataset.path),
        checkpoint_property_of_interest=config.dataset.train_metric,
        openai_coeff=config.dataset.openai_coeff,
    )
    print("number of training samples", len(training_dataset))

    # TODO do we need their loader?
    train_loader = construct_loader(
        training_dataset,
        config.train.mb_size,
        config.num_gpus,
        shuffle=True,
        drop_last=True,
        num_workers=config.dataset.num_workers,
    )
    # TODO examine data

    # Construct meters
    train_meter = TrainMeter(len(train_loader), config.train.num_ep)

    # TODO GPU training
    # TODO construct required sub-models and pass them on
    model = WrapperModel()


@hydra.main(config_path="../../../configs/train", config_name="config.yaml")
def main(config: omegaconf.DictConfig):
    # TODO Implement multi thread training
    train(config)
