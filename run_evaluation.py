import json
import re
from pathlib import Path
from typing import Dict, Tuple

import torch
import hydra
import omegaconf
import wandb
from tqdm import tqdm
from ghrp.model_definitions.def_net import NNmodule
from pytorch_lightning import seed_everything

from weight_diffusion.evaluation.eval_task import (
    get_evaluation_datasets,
    finetune_MNIST_CNN,
    evaluate_MNIST_CNN,
    instantiate_MNIST_CNN,
)
from weight_diffusion.evaluation.metrics import calculate_ldm_prompt_alignment
from weight_diffusion.evaluation.util import (
    sample_checkpoints_from_ldm,
    instantiate_gpt,
    instantiate_ldm,
    instantiate_encoder,
    initiate_tokenizer,
    log_dictionary_locally
)


def evaluate(
    config: omegaconf.DictConfig,
    models_to_evaluate: Dict[str, Tuple[NNmodule, Dict[str, float]]],
):
    evaluation_datasets = get_evaluation_datasets(config.evaluation_dataset_config)

    log_dict = {}
    current_epoch = 0
    for finetune_epoch in tqdm(
        config.finetune_config.finetune_epochs, desc="Evaluating sampled weights"
    ):
        epochs_to_train = finetune_epoch - current_epoch
        current_epoch = finetune_epoch
        for prompt, (model, targets) in models_to_evaluate.items():
            if finetune_epoch == 0:
                # Instantiate logging dict
                log_dict[prompt] = {}
                # Instantiate tracking of training
                log_dict[prompt]["train_running_loss"] = []
                log_dict[prompt]["train_running_accuracy"] = []
                log_dict[prompt]["targets"] = dict(targets)
            else:
                _, progress_dict = finetune_MNIST_CNN(
                    model, epochs_to_train, evaluation_datasets["train"], prompt
                )
                for k in ["train_running_loss", "train_running_accuracy"]:
                    log_dict[prompt][k] += progress_dict[k]

            evaluation_dict = evaluate_MNIST_CNN(model, evaluation_datasets)
            log_dict[prompt][f"epoch_{finetune_epoch}"] = evaluation_dict

            # if first epoch then calculate prompt alignment
            if finetune_epoch == 0:
                prompt_alignment = calculate_ldm_prompt_alignment(
                    evaluation_dict=evaluation_dict, targets=targets
                )
                log_dict[prompt]["prompt_alignment"] = prompt_alignment
                
    log_dictionary_locally(
        logging_dict=log_dict,
        logging_path="./logs.json"
    )
    wandb.log(
        log_dict,
    )


@hydra.main(config_path="./configs/evaluate", config_name="config.yaml")
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
        diffusion, gpt_model, train_dataset = instantiate_gpt(config.gpt_config)
        # TODO Test both togehther? 

    elif config.sampling_method == "ldm":
        ldm = instantiate_ldm(config.ldm_config)
        encoder = instantiate_encoder(config.encoder_config)
        tokenizer = initiate_tokenizer(config.tokenizer_config)
        (
            sampled_mnist_model_checkpoints_dict,
            targets_dict,
        ) = sample_checkpoints_from_ldm(
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
        model = instantiate_MNIST_CNN(model_config, sampled_checkpoint, config.device)
        models_to_evaluate[prompt] = (model, targets_dict[prompt])

    evaluate(config, models_to_evaluate)


if __name__ == "__main__":
    main()
