import json
import re
from pathlib import Path
from typing import Dict

import hydra
import omegaconf
import torchvision
import torchvision.transforms as transforms
import wandb
from ghrp.model_definitions.def_net import NNmodule
from pytorch_lightning import seed_everything

from weight_diffusion.data.data_utils.helper import generate_checkpoints_from_weights
from weight_diffusion.ofga.evaluation import *


def evaluate(config: omegaconf.DictConfig, models_to_evaluate: Dict[str, NNmodule]):
    evaluation_datasets = _get_evaluation_datasets(config.evaluation_dataset_config)

    current_epoch = 0
    for finetune_epoch in config.finetune_config.finetune_epochs:
        epochs_to_train = finetune_epoch - current_epoch
        current_epoch = finetune_epoch
        for prompt, model in models_to_evaluate:
            if epochs_to_train > 0:
                _finetune_MNIST_CNN(model,
                                    epochs_to_train,
                                    evaluation_datasets['train'],
                                    config.finetune_config)

            # if first epoch then calculate prompt alignment
            if finetune_epoch == 0:
                evaluation_dict = _evaluate_MNIST_CNN(model,
                                                      evaluation_datasets,
                                                      config.evaluation_config,
                                                      prompt)
            else:
                evaluation_dict = _evaluate_MNIST_CNN(model,
                                                      evaluation_datasets,
                                                      config.evaluation_config)

            log_dict = {"model_prompt": prompt,
                        "finetune_epoch": finetune_epoch,
                        "evaluation": evaluation_dict}

            wandb.log(log_dict)

        # TODO put model into test mode?


@hydra.main(config_path="../../../configs/evaluate", config_name="config.yaml")
def main(config: omegaconf.DictConfig):
    # initiate wandb logging of evaluation
    wandb.init(config=config.wandb_config)

    # Set global seed
    seed_everything(config.seed)

    device = config.device

    # Load model architecture
    with open(config.data_dir.joinpath("config.json")) as model_json:
        model_config = json.load(model_json)

    with open(config.data_dir.joinpath("index_dict.json")) as index_json:
        index_dict = json.load(index_json)
    layer_list = index_dict["layer"]

    # Sampling checkpoints based on method under evaluation
    if config.sampling_method == 'Gpt':
        # TODO sample from Gpt
        raise NotImplementedError
    elif config.sampling_method == 'ldm':
        ldm = _instantiate_ldm(config.ldm_config)
        encoder = _instantiate_encoder(config.encoder_config)
        tokenizer = _initiate_tokenizer(config.tokenizer_config)
        sampled_mnist_model_checkpoints_dict = _sample_checkpoints_from_ldm(config.sampling_config,
                                                                            model_config,
                                                                            layer_list,
                                                                            ldm,
                                                                            encoder,
                                                                            tokenizer,
                                                                            device)
    else:
        raise NotImplementedError

    models_to_evaluate = {}
    for prompt, sampled_checkpoint in sampled_mnist_model_checkpoints_dict:
        model = _instantiate_MNIST_CNN(model_config, sampled_checkpoint)
        models_to_evaluate[prompt] = model

    evaluate(config, models_to_evaluate)


if __name__ == "__main__":
    main()


def _sample_checkpoints_from_ldm(sampling_config, model_config, layer_list, ldm, encoder, tokenizer, device):
    noise = torch.randn(sampling_config.shape, device=device)

    sampled_mnist_model_checkpoints_dict = {}
    for prompt_statistics in sampling_config.evaluation_prompt_statistics:
        prompt = _prompt_from_results_dict(prompt_statistics)
        prompt_latent_rep = tokenizer(
            prompt,
            max_length=sampling_config.prompt_embedding_max_length,
            return_tensors="pt",
            padding="max_length",
        )["input_ids"]
        sampled_weights_latent = ldm(noise, prompt_latent_rep)
        sampled_weights = encoder.forward_decoder(sampled_weights_latent)
        sampled_checkpoint = generate_checkpoints_from_weights(sampled_weights,
                                                               model_config,
                                                               layer_list)
        sampled_mnist_model_checkpoints_dict[prompt] = sampled_checkpoint

    return sampled_mnist_model_checkpoints_dict


def _instantiate_encoder(encoder_config):
    encoder = instantiate_from_config(encoder_config)
    hyper_representations_path = Path(encoder_config['encoder_checkpoint_path'])
    encoder_checkpoint_path = hyper_representations_path.joinpath('checkpoint_ae.pt')
    encoder_checkpoint = torch.load(encoder_checkpoint_path)
    encoder.model.load_state_dict(encoder_checkpoint)
    return encoder


def _initiate_tokenizer(tokenizer_config):
    tokenizer = instantiate_from_config(tokenizer_config)
    return tokenizer


def _instantiate_ldm(ldm_config):
    ldm = instantiate_from_config(ldm_config)
    ldm_checkpoint_path = Path(ldm_config['ldm_checkpoint_path'])
    ldm_checkpoint = torch.load(ldm_checkpoint_path)
    ldm.model.load_state_dict(ldm_checkpoint)
    return ldm


def _instantiate_MNIST_CNN(mnist_cnn_config, checkpoint):
    mnist_cnn = NNmodule(mnist_cnn_config, verbosity=0)
    mnist_cnn.model.load_state_dict(checkpoint)
    return mnist_cnn


def _get_evaluation_datasets(evaluation_dataset_config):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    evaluation_datasets = {
        'test': torch.utils.data.DataLoader(torchvision.datasets.MNIST(root=evaluation_dataset_config.data_dir,
                                                                       train=False,
                                                                       download=True,
                                                                       transform=transform),
                                            batch_size=10000,
                                            shuffle=False)}

    training_data = torchvision.datasets.MNIST(root=evaluation_dataset_config.data_dir,
                                               train=True,
                                               download=True,
                                               transform=transform)

    training_data = torch.utils.data.random_split(training_data, evaluation_dataset_config.split)

    evaluation_datasets['train'] = torch.utils.data.DataLoader(training_data[0],
                                                               batch_size=10000,
                                                               shuffle=False)
    evaluation_datasets['validation'] = torch.utils.data.DataLoader(training_data[1],
                                                                    batch_size=10000,
                                                                    shuffle=False)

    return evaluation_datasets


def _finetune_MNIST_CNN(model, epochs, train_dataloader, finetune_config):
    loss_func = finetune_config.loss_func
    optimiser = finetune_config.optimiser

    for epoch in range(epochs):
        for (i, (images, labels)) in enumerate(train_dataloader):
            (images, labels) = (images.to(finetune_config.device, dtype=torch.float), labels.to(finetune_config.DEVICE))
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


def _evaluate_MNIST_CNN(model, evaluation_datasets, evaluation_config, prompt=None):
    evaluation_dict = {}

    for key, dataloader in evaluation_datasets:
        # init collection of mini-batch losses
        cumulative_batch_losses = 0
        cumulative_batch_correct = 0
        count = 0

        # iterate over all-mini batches
        for i, (images, labels) in enumerate(dataloader):
            loss, correct = mnist_task.test_step(images, labels, model, evaluation_config.loss_function)
            cumulative_batch_losses += loss
            cumulative_batch_correct += correct
            count += len(images)

        # TODO is this the way to get number of batches?
        number_of_batches = len(dataloader)
        overall_loss = cumulative_batch_losses / number_of_batches
        overall_accuracy = cumulative_batch_correct / count

        evaluation_dict[key] = {'loss': overall_loss, 'accuracy': overall_accuracy}

    if prompt is not None:
        evaluation_dict['prompt_alignment'] = _calculate_ldm_prompt_alignment(prompt, evaluation_dict)

    return evaluation_dict


def _calculate_ldm_prompt_alignment(prompt, evaluation_dict):
    prompted_statistics = re.findall(r"[-+]?(?:\d*\.*\d+)", prompt)
    diff = ...  # TODO what formula?
    return diff


def _prompt_from_results_dict(results_dict):
    prompt = (
        f"The training loss is {results_dict['train_loss']:.4g}. "
        f"The training accuracy is {results_dict['train_acc']:.4g}. "
        f"The validation loss is {results_dict['validation_loss']:.4g}. "
        f"The validation accuracy is {results_dict['validation_acc']:.4g}. "
        f"The test loss is {results_dict['test_loss']:.4g}. "
        f"The test accuracy is {results_dict['test_acc']:.4g}. "
    )
    return prompt
