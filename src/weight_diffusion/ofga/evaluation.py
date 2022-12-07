from typing import List
from weight_diffusion.zoo import mnist_task
from pytorch_lightning.callbacks import Callback
import torch
import ray
from ray import tune
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLogger
from ghrp.sampling_auxiliaries.sample_finetune_auxiliaries import *
from ldm.util import instantiate_from_config

# TODO add PromptAlignmentCallback to trainer
def evaluate_generated_models(n: int, prompt: str, models: List[mnist_task.CNN]):
    k = len(models)
    
    if ray.is_initialized():
        ray.shutdown()

    finetune(
        project=f"{source}_to_{target}",
        population_path=population_path,
        path_to_samples=path_to_samples,
        path_target_zoo=source_zoo_path,
        model_config_path=model_config_path,
        model_config=take_config_from,
        no_samples=no_samples,
        training_epochs=n,
        cpus=cpus,
        skip=["uniform", "train", "kde_z_train"],
    )

    # TODO Fine tune them for n steps
    # TODO Evaluate their performance during training on MNIST
    # TODO Return dict of final classification results
    pass


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

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        noise = ...  # Generate from shape
        mnist_model_checkpoints_encoded = [pl_module(noise, prompt) for prompt in self.list_of_prompts]
        mnist_model_checkpoints = self.encoder.forward_decoder(mnist_model_checkpoints_encoded)
        results = {}
        for checkpoint in mnist_model_checkpoints:
            current_model = mnist_task.CNN(**self.model_config).load_state_dict(state_dict=checkpoint)
            # TODO evaluate current_model on MNIST dataloader
            # TODO store results in dictionary
        pass
