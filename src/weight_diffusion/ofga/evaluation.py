from typing import List
from weight_diffusion.zoo import mnist_task
from pytorch_lightning.callbacks import Callback


# TODO add PromptAlignmentCallback to trainer
def evaluate_generated_models(n: int, prompt: str, models: List[mnist_task.CNN]):
    k = len(models)
    # TODO Generate list of k model given prompt
    # TODO Fine tune them for n steps
    # TODO Evaluate their performance during training on MNIST
    # TODO Return dict of final classification results
    pass


class PromptAlignmentCallback(Callback):
    def __init__(self, list_of_prompts, checkpoints_shape, model_config, dataloader):
        self.list_of_prompts = list_of_prompts
        self.checkpoints_shape = checkpoints_shape
        self.model_config = model_config
        self.dataloader = dataloader

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        noise = ...  # Generate from shape
        mnist_model_checkpoints = [pl_module(noise, prompt) for prompt in self.list_of_prompts]
        results = {}
        for checkpoint in mnist_model_checkpoints:
            current_model = mnist_task.CNN(**self.model_config).load_state_dict(state_dict=checkpoint)
            # TODO evaluate current_model on MNIST dataloader
            # TODO store results in dictionary
        pass
