from typing import List
from weight_diffusion.zoo import mnist_task
from pytorch_lightning.callbacks import Callback
import torch
import ray
from ray import tune
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLogger
import ghrp.sampling_auxiliaries.sample_finetune_auxiliaries
from ldm.util import instantiate_from_config
import wandb


def evaluate_generated_models(epochs: int, prompt: str, models: mnist_task.CNN, input, target):
    k = len(models)
    wandb.init()

    for model in models:
        _finetune_model(model, epochs)
        _evaluate_model(model, input, target)
        # TODO record evaluation metrics 
        # TODO save models?

def _finetune_model(model, epochs):
    # TODO finetune for requiested epochs, our code or Konstantin's? 
    tune()

def _evaluate_model(model: mnist_task.CNN, input, target):
    loss, accuracy = mnist_task.test_step(input, target, model)
    # TODO save evaluation, can't always rely on lighting?

def _calculate_prompt_alignment(prompt: str, results_dict: dict):
    # TODO calculate prompt alignment


# TODO add PromptAlignmentCallback to trainer
class PromptAlignmentCallback(Callback):
    def __init__(self, list_of_prompts, checkpoints_shape, model_config, dataloader, encoder_config):
        self.list_of_prompts = list_of_prompts
        self.checkpoints_shape = checkpoints_shape
        self.model_config = model_config
        self.dataloader = dataloader

        # load decoder
        self.encoder = instantiate_from_config(encoder_config)
        self.hyper_representations_path = Path(encoder_config['encoder_checkpoint_path'])
        encoder_checkpoint_path = self.hyper_representations_path.joinpath('checkpoint_ae.pt')
        encoder_checkpoint = torch.load(encoder_checkpoint_path)
        self.encoder.model.load_state_dict(encoder_checkpoint)

    # TODO which function to overwrite 
    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        noise = ...  # Generate from shape
        mnist_model_checkpoints_encoded = [pl_module(noise, prompt) for prompt in self.list_of_prompts]
        mnist_model_checkpoints = self.encoder.forward_decoder(mnist_model_checkpoints_encoded)
        results = {}
        for checkpoint in mnist_model_checkpoints:
            current_model = mnist_task.CNN(**self.model_config).load_state_dict(state_dict=checkpoint)

            # TODO evaluate current_model on MNIST dataloader
            # TODO store results in dictionary and save them
            # TODO use lightning module to log
        pass
