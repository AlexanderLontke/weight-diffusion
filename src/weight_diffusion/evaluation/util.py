from typing import Dict, Tuple
from pathlib import Path
import torch
import json

from ldm.util import instantiate_from_config
from weight_diffusion.execution.util import load_model_from_config
from weight_diffusion.data.data_utils.helper import generate_checkpoints_from_weights
from weight_diffusion.ofga.sampling import sample_from_prompt


def prompt_from_results_dict(results_dict):
    prompt = (
        f"The training loss is {results_dict['train_loss']:.4g}. "
        f"The training accuracy is {results_dict['train_acc']:.4g}. "
        f"The validation loss is {results_dict['validation_loss']:.4g}. "
        f"The validation accuracy is {results_dict['validation_acc']:.4g}. "
        f"The test loss is {results_dict['test_loss']:.4g}. "
        f"The test accuracy is {results_dict['test_acc']:.4g}. "
    )
    return prompt


def sample_checkpoints_from_ldm(
    sampling_config,
    model_config,
    layer_list,
    ldm,
    encoder,
    tokenizer,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, float]]]:
    sampled_mnist_model_checkpoints_dict = {}
    targets_dict = {}
    for prompt_statistics in [
        v for _, v in sampling_config.evaluation_prompt_statistics.items()
    ]:
        prompt = prompt_from_results_dict(prompt_statistics)
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


def instantiate_encoder(encoder_config):
    encoder = instantiate_from_config(encoder_config)
    hyper_representations_path = Path(encoder_config["encoder_checkpoint_path"])
    encoder_checkpoint_path = hyper_representations_path.joinpath("checkpoint_ae.pt")
    encoder_checkpoint = torch.load(encoder_checkpoint_path)
    encoder.model.load_state_dict(encoder_checkpoint)
    return encoder


def initiate_tokenizer(tokenizer_config):
    tokenizer = instantiate_from_config(tokenizer_config)
    return tokenizer


def instantiate_ldm(ldm_config):
    ldm_checkpoint_path = Path(ldm_config.ldm_checkpoint_path)
    ldm = load_model_from_config(ldm_config, ldm_checkpoint_path)
    return ldm


def log_dictionary_locally(logging_dict: Dict, logging_path: str):
    logging_path: Path = Path(logging_path)
    logging_path.parent.mkdir(parents=True, exist_ok=True)
    with logging_path.open('w') as convert_file:
        convert_file.write(json.dumps(logging_dict))

def load_logging_dict(logging_path: str):
    logging_path: Path = Path(logging_path)
    with logging_path.open('r') as convert_file:
        logging_dict = convert_file.read()
    return json.loads(logging_dict)
