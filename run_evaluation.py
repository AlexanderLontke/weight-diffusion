import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pickle
import hydra
import omegaconf
import wandb
import torch
from tqdm import tqdm
from ghrp.model_definitions.def_net import NNmodule
from pytorch_lightning import seed_everything

from weight_diffusion.evaluation.eval_task import (
    get_evaluation_datasets,
    finetune_MNIST_CNN,
    evaluate_MNIST_CNN,
    instantiate_MNIST_CNN,
)
from weight_diffusion.evaluation.util import (
    sample_checkpoints_from_ldm,
    sample_checkpoint_from_gpt,
    instantiate_gpt,
    instantiate_ldm,
    instantiate_encoder,
    initiate_tokenizer,
    log_dictionary_locally
)


def evaluate(
    config: omegaconf.DictConfig,
    models_to_evaluate: Dict[str, Tuple[NNmodule, Dict[str, float]]],
    model_config: Dict[str, Any],
):
    evaluation_datasets = get_evaluation_datasets(
        evaluation_dataset_config=config.evaluation_dataset_config,
        model_config=model_config,
    )

    log_dict = {}
    prompt_targets = {}
    prompt_actual = {}
    for prompt, (model, targets) in  tqdm(models_to_evaluate.items(), desc="Evaluating sampled weights"):
        # Instantiate logging dict
        log_dict[prompt] = {}
        # Instantiate tracking of training
        log_dict[prompt]["train_running_loss"] = []
        log_dict[prompt]["train_running_accuracy"] = []
        log_dict[prompt]["targets"] = dict(targets)

        current_epoch = 0
        evaluation_dict = evaluate_MNIST_CNN(model, evaluation_datasets, current_epoch)
        log_dict[prompt][f"epoch_{current_epoch}"] = evaluation_dict

        for k in targets.keys():
            if prompt == "baseline":
                continue
            if k not in prompt_targets.keys():
                prompt_targets[k] = []
                prompt_actual[k] = []
            prompt_targets[k].append(targets[k]) 
            prompt_actual[k].append(evaluation_dict[k]) 

        if prompt in list(config.finetune_config.prompts_to_finetune) or prompt == "baseline":
            for finetune_epoch in tqdm(
                config.finetune_config.finetune_epochs, desc="Fine-tuning sampled weights"
            ):
                epochs_to_train = finetune_epoch - current_epoch
                current_epoch = finetune_epoch

                _, progress_dict = finetune_MNIST_CNN(
                    model, epochs_to_train, evaluation_datasets["train"]
                )
                for k in ["train_running_loss", "train_running_accuracy"]:
                    log_dict[prompt][k] += progress_dict[k]
                
                evaluation_dict = evaluate_MNIST_CNN(model, evaluation_datasets, finetune_epoch)
            
                log_dict[prompt][f"epoch_{finetune_epoch}"] = evaluation_dict

    log_dict["prompt_alignment_prompt_actual"] = prompt_actual
    log_dict["prompt_alignment_prompt_targets"] = prompt_targets

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
    wandb.init(**config.wandb_config, config=config.__dict__)

    # Set global seed
    seed_everything(config.seed)

    # Load model architecture
    with open(Path(config.data_dir).joinpath("config.json")) as model_json:
        model_config = json.load(model_json)

    with open(Path(config.data_dir).joinpath("index_dict.json")) as index_json:
        index_dict = json.load(index_json)
    layer_list = index_dict["layer"]

    # Sample checkpoints
    targets_dict = {}
    sampled_checkpoints_path = config.get("pickled_sampled_checkpoints_dir", None)
    if sampled_checkpoints_path:
        with open(f"{sampled_checkpoints_path}/sampled_checkpoints_path.pkl", "rb") as file:
            sampled_mnist_model_checkpoints_dict = pickle.load(file=file)
        with open(f"{sampled_checkpoints_path}/targets_dict.pkl", "rb") as file:
            targets_dict = pickle.load(file=file)
    else:
        # Sampling checkpoints based on method under evaluation
        if config.sampling_method == "Gpt":
            diffusion, gpt_model, dataset = instantiate_gpt(config.gpt_config)
            cur_device = torch.cuda.current_device()
            gpt_model = gpt_model.cuda(device=cur_device)
            (
                sampled_mnist_model_checkpoints_dict,
                targets_dict,
            ) = sample_checkpoint_from_gpt(config.sampling_config, model_config, layer_list, diffusion, gpt_model, dataset)
        elif config.sampling_method == "ldm":
            ldm = instantiate_ldm(config.ldm_config)
            encoder = instantiate_encoder(config.encoder_config)
            # tokenizer = initiate_tokenizer(config.tokenizer_config)
            (
                sampled_mnist_model_checkpoints_dict,
                targets_dict,
            ) = sample_checkpoints_from_ldm(
                sampling_config=config.sampling_config,
                model_config=model_config,
                layer_list=layer_list,
                ldm=ldm,
                encoder=encoder,
                # tokenizer=tokenizer,
                device=config.device
            )
        else:
            raise NotImplementedError

        with open("./sampled_checkpoints_path", "wb") as file:
            pickle.dump(sampled_mnist_model_checkpoints_dict, file=file)
        with open("./targets_dict", "wb") as file:
            pickle.dump(targets_dict, file=file)

    models_to_evaluate = {}
    first = True
    for prompt, sampled_checkpoint in sampled_mnist_model_checkpoints_dict.items():
        if first:
            randomly_initialiased_model = instantiate_MNIST_CNN(model_config, device=config.device)
            models_to_evaluate['baseline'] = (randomly_initialiased_model, targets_dict[prompt])
        model = instantiate_MNIST_CNN(model_config, sampled_checkpoint, config.device)
        models_to_evaluate[prompt] = (model, targets_dict[prompt])
        first = False

    evaluate(config, models_to_evaluate, model_config=model_config)


if __name__ == "__main__":
    main()
