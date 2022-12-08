import hydra
import omegaconf
import torch
from ldm.util import instantiate_from_config
from pathlib import Path
from typing import Tuple, List, Dict, Union, Callable
from weight_diffusion.zoo import mnist_task
from weight_diffusion.ofga.evaluation import *

def evaluate(evaluation_config: Dict, models_to_evaluate: List[mnist_task.CNN]):




@hydra.main(config_path="../../../configs/evaluate", config_name="config.yaml")
def main(cfg: omegaconf.DictConfig):

    # Sampling checkpoints based on method under evaluation
    if cfg.sampling_method == 'Gpt':
        # TODO sample from Gpt
        raise NotImplementedError
    elif cfg.sampling_method == 'ldm':
        ldm = _instantiate_ldm(cfg.ldm_config)
        encoder = _instantiate_encoder(cfg.encoder_config)
        sampled_mnist_model_checkpoints_dict = _sample_checkpoints_from_ldm(cfg.evaluation_prompts, ldm, encoder)
    else:
        raise NotImplementedError

    models_to_evaluate = {}
    for prompt, sampled_checkpoint in sampled_mnist_model_checkpoints_dict:
        model = _instantiate_MNIST_CNN(cfg.mnist_cnn_config, sampled_checkpoint)
        models_to_evaluate[prompt] = model

    evaluate(cfg.evaluation_config, models_to_evaluate)


if __name__ == "__main__":
    main()

def _sample_checkpoints_from_ldm(evaluation_prompts, ldm, encoder):
    noise = ...  # TODO Generate from shape, (see ldm?)

    sampled_mnist_model_checkpoints_dict = {}
    for prompt in evaluation_prompts:
        sampled_checkpoint_latent = ldm(noise, prompt)
        sampled_checkpoint = encoder.forward_decoder(sampled_checkpoint_latent)
        sampled_mnist_model_checkpoints_dict[prompt] = sampled_checkpoint

    return sampled_mnist_model_checkpoints_dict


def _instantiate_encoder(encoder_config):
    encoder = instantiate_from_config(encoder_config)
    hyper_representations_path = Path(encoder_config['encoder_checkpoint_path'])
    encoder_checkpoint_path = hyper_representations_path.joinpath('checkpoint_ae.pt')
    encoder_checkpoint = torch.load(encoder_checkpoint_path)
    encoder.model.load_state_dict(encoder_checkpoint)
    return encoder

def _instantiate_ldm(ldm_config):
    ldm = instantiate_from_config(ldm_config)
    ldm_checkpoint_path = Path(ldm_config['ldm_checkpoint_path'])
    ldm_checkpoint = torch.load(ldm_checkpoint_path)
    ldm.model.load_state_dict(ldm_checkpoint)
    return ldm

def _instantiate_MNIST_CNN(mnist_cnn_config, checkpoint):
    mnist_cnn = instantiate_from_config(mnist_cnn_config)
    mnist_cnn.model.load_state_dict(checkpoint)
    return mnist_cnn

def _get_evaluation_dataset