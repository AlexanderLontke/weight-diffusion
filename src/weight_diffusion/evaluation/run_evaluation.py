import json
import re
from pathlib import Path
from typing import Dict, Tuple

import torch
import hydra
import omegaconf
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import wandb
from tqdm import tqdm
from ghrp.model_definitions.def_net import NNmodule
from pytorch_lightning import seed_everything
from torch.utils.data import random_split, DataLoader

from ldm.util import instantiate_from_config
from weight_diffusion.execution.util import load_model_from_config
from weight_diffusion.ofga.sampling import sample_from_prompt
from weight_diffusion.data.data_utils.helper import generate_checkpoints_from_weights


def _sample_checkpoints_from_ldm(
    sampling_config, model_config, layer_list, ldm, encoder, tokenizer,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, float]]]:
    sampled_mnist_model_checkpoints_dict = {}
    targets_dict = {}
    for prompt_statistics in [
        v for _, v in sampling_config.evaluation_prompt_statistics.items()
    ]:
        prompt = _prompt_from_results_dict(prompt_statistics)
        prompt_latent_rep = tokenizer(
            prompt,
            max_length=sampling_config.prompt_embedding_max_length,
            return_tensors="pt",
            padding="max_length",
        )["input_ids"]
        sampled_weights_latent = sample_from_prompt(
            prompt=prompt_latent_rep,
            model=ldm,
            sampling_steps=sampling_config.sampling_steps,
            shape=tuple(sampling_config.shape),
            guidance_scale=1.0,
        )
        sampled_weights = encoder.forward_decoder(sampled_weights_latent)
        sampled_checkpoint = generate_checkpoints_from_weights(
            sampled_weights, model_config, layer_list
        )
        sampled_mnist_model_checkpoints_dict[prompt] = sampled_checkpoint
        # Return dictionary containing target metrics for each prompt
        targets_dict[prompt] = prompt_statistics

    return sampled_mnist_model_checkpoints_dict, targets_dict


def _instantiate_encoder(encoder_config):
    encoder = instantiate_from_config(encoder_config)
    hyper_representations_path = Path(encoder_config["encoder_checkpoint_path"])
    encoder_checkpoint_path = hyper_representations_path.joinpath("checkpoint_ae.pt")
    encoder_checkpoint = torch.load(encoder_checkpoint_path)
    encoder.model.load_state_dict(encoder_checkpoint)
    return encoder


def _initiate_tokenizer(tokenizer_config):
    tokenizer = instantiate_from_config(tokenizer_config)
    return tokenizer


def _instantiate_ldm(ldm_config):
    ldm_checkpoint_path = Path(ldm_config.ldm_checkpoint_path)
    ldm = load_model_from_config(ldm_config, ldm_checkpoint_path)
    return ldm


def _instantiate_MNIST_CNN(mnist_cnn_config, checkpoint, device):
    mnist_cnn = NNmodule(mnist_cnn_config,cuda=(device == "cuda"), verbosity=0)
    mnist_cnn.model.load_state_dict(checkpoint)
    mnist_cnn = mnist_cnn.to(torch.device(device))
    return mnist_cnn


def _get_evaluation_datasets(evaluation_dataset_config):
    # Same seed as in
    # https://github.com/ModelZoos/ModelZooDataset/blob/main/code/zoo_generators/train_zoo_mnist_uniform.py
    dataset_seed = 42
    evaluation_datasets = {}
    # load raw dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    val_and_trainset_raw = datasets.MNIST(
        evaluation_dataset_config.data_dir, train=True, download=True, transform=transform)
    testset_raw = datasets.MNIST(
        evaluation_dataset_config.data_dir, train=False, download=True, transform=transform)
    trainset_raw, valset_raw = torch.utils.data.random_split(
        val_and_trainset_raw, [50000, 10000], generator=torch.Generator().manual_seed(dataset_seed))

    # temp dataloaders
    evaluation_datasets["train"] = torch.utils.data.DataLoader(
        dataset=trainset_raw, batch_size=len(trainset_raw), shuffle=True
    )
    evaluation_datasets["validation"] = torch.utils.data.DataLoader(
        dataset=valset_raw, batch_size=len(valset_raw), shuffle=True
    )
    evaluation_datasets["test"] = torch.utils.data.DataLoader(
        dataset=testset_raw, batch_size=len(testset_raw), shuffle=True
    )

    return evaluation_datasets


def _finetune_MNIST_CNN(model: NNmodule, epochs, train_dataloader, prompt):
    for epoch in range(epochs):
        loss_runing, accuracy = model.train_epoch(train_dataloader, epoch)
        wandb.log(
            {
                f"{prompt}/train_loss_running": loss_runing,
                f"{prompt}/train_accuracy": accuracy
            }
        )
    return model


def _evaluate_MNIST_CNN(model: NNmodule, evaluation_datasets):
    evaluation_dict = {}

    for key, dataloader in evaluation_datasets.items():
        overall_loss, overall_accuracy = model.test_epoch(dataloader, epoch=-1)
        evaluation_dict[f"{key}_loss"] = overall_loss
        evaluation_dict[f"{key}_acc"] = overall_accuracy

    return evaluation_dict


def _calculate_ldm_prompt_alignment(evaluation_dict, targets):
    """
    Calculate the prompt alignment based on the root mean squared errors of metrics
    :param evaluation_dict: dictionary containing actual statistics achieved by a sampled checkpoint
    :param targets: dictionary containing actual statistics in prompt
    :return:
    Root mean squared error of metrics' differences
    """
    squared_errors = []
    for k in evaluation_dict.keys():
        squared_errors += [(evaluation_dict[k] - targets[k]) ** 2]
    mse = np.mean(squared_errors)
    return np.sqrt(mse)


def _prompt_from_results_dict(results_dict):
    print(results_dict)
    prompt = (
        f"The training loss is {results_dict['train_loss']:.4g}. "
        f"The training accuracy is {results_dict['train_acc']:.4g}. "
        f"The validation loss is {results_dict['validation_loss']:.4g}. "
        f"The validation accuracy is {results_dict['validation_acc']:.4g}. "
        f"The test loss is {results_dict['test_loss']:.4g}. "
        f"The test accuracy is {results_dict['test_acc']:.4g}. "
    )
    return prompt


def evaluate(config: omegaconf.DictConfig, models_to_evaluate: Dict[str, Tuple[NNmodule, Dict[str, float]]]):
    evaluation_datasets = _get_evaluation_datasets(config.evaluation_dataset_config)
    
    log_dict = {}
    current_epoch = 0
    for finetune_epoch in tqdm(config.finetune_config.finetune_epochs, desc="Evaluating sampled weights"):
        epochs_to_train = finetune_epoch - current_epoch
        current_epoch = finetune_epoch
        for prompt, (model, targets) in models_to_evaluate.items():
            if epochs_to_train > 0:
                _finetune_MNIST_CNN(
                    model, epochs_to_train, evaluation_datasets["train"], prompt
                )

            evaluation_dict = _evaluate_MNIST_CNN(
                model, evaluation_datasets,
            )
            
            log_dict[prompt][finetune_epoch] = evaluation_dict
            # if first epoch then calculate prompt alignment
            if finetune_epoch == 0:
                prompt_alignment = _calculate_ldm_prompt_alignment(
                    evaluation_dict=evaluation_dict,
                    targets=targets
                )
                log_dict[prompt]["prompt_alignment"] = prompt_alignment
    wandb.log(
        log_dict,
    )

@hydra.main(config_path="../configs/evaluate", config_name="config.yaml")
def main(config: omegaconf.DictConfig):
    # initiate wandb logging of evaluation
    wandb.init(**config.wandb_config)

    # Set global seed
    seed_everything(config.seed)

    # Load model architecture
    with open(Path(config.data_dir).joinpath("config.json")) as model_json:
        model_config = json.load(model_json)

    with open(Path(config.data_dir).joinpath("index_dict.json")) as index_json:
        index_dict = json.load(index_json)
    layer_list = index_dict["layer"]

    # Sampling checkpoints based on method under evaluation
    if config.sampling_method == "Gpt":
        # TODO sample from Gpt
        raise NotImplementedError
    elif config.sampling_method == "ldm":
        ldm = _instantiate_ldm(config.ldm_config)
        encoder = _instantiate_encoder(config.encoder_config)
        tokenizer = _initiate_tokenizer(config.tokenizer_config)
        sampled_mnist_model_checkpoints_dict, targets_dict = _sample_checkpoints_from_ldm(
            sampling_config=config.sampling_config,
            model_config=model_config,
            layer_list=layer_list,
            ldm=ldm,
            encoder=encoder,
            tokenizer=tokenizer,
        )
    else:
        raise NotImplementedError

    models_to_evaluate = {}
    for prompt, sampled_checkpoint in sampled_mnist_model_checkpoints_dict.items():
        model = _instantiate_MNIST_CNN(model_config, sampled_checkpoint, config.device)
        models_to_evaluate[prompt] = (model, targets_dict[prompt])

    evaluate(config, models_to_evaluate)


if __name__ == "__main__":
    main()
