import hydra
import omegaconf
import torch
from ldm.util import instantiate_from_config
from pathlib import Path
from typing import Tuple, List, Dict, Union, Callable
from weight_diffusion.zoo import mnist_task
from weight_diffusion.ofga.evaluation import *
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything


def evaluate(evaluation_config: omegaconf.DictConfig, models_to_evaluate: Dict[str, mnist_task.CNN]):
    evaluation_datasets = _get_evaluation_datasets(evaluation_config.evaluation_dataset_config)

    current_epoch = 0
    for finetune_epoch in evaluation_config.finetune_epochs:
        epochs_to_train = finetune_epoch - current_epoch
        if epochs_to_train > 0:
            for model in models_to_evaluate:
                _finetune_MNIST_CNN(model,
                                    epochs_to_train,
                                    evaluation_datasets['train'],
                                    evaluation_config.finetune_config)

        # TODO put model into test mode?


@hydra.main(config_path="../../../configs/evaluate", config_name="config.yaml")
def main(cfg: omegaconf.DictConfig):
    # TODO set up wandb logging

    # Set global seed
    seed_everything(cfg.seed)

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


def _get_evaluation_datasets(evaluation_dataset_config):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    evaluation_datasets = {'test': torchvision.datasets.MNIST(root=evaluation_dataset_config.data_dir,
                                                              train=False,
                                                              download=True,
                                                              transform=transform)}

    training_data = torchvision.datasets.MNIST(root=evaluation_dataset_config.data_dir,
                                               train=True,
                                               download=True,
                                               transform=transform)
    training_data = torch.utils.data.random_split(training_data, evaluation_dataset_config.split)

    evaluation_datasets['train'] = training_data[0]
    evaluation_datasets['validation'] = training_data[1]

    return evaluation_datasets


def _finetune_MNIST_CNN(model, epochs, train_dataloader, finetune_config):
    loss_func = finetune_config.loss_func
    optimiser = finetune_config.optimiser

    for epoch in range(epochs):
        for (i, (images, labels)) in enumerate(train_dataloader):
            (images, labels) = (images.to(finetune_config.DEVICE, dtype=torch.float), labels.to(finetune_config.DEVICE))
            # run forward pass through the model
            pred = model(images)
            batch_predictions = torch.argmax(pred, dim=1)

            # calculating loss
            loss = loss_func(pred, labels)

            # resetting gradients
            optimiser.zero_grad()

            # Backward Propagation
            loss.backward()

            # Updating the weights
            optimiser.step()

    return model
