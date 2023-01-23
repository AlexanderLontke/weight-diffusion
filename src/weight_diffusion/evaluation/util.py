from typing import Dict, Tuple
from pathlib import Path
import torch
import json

from copy import deepcopy

from Gpt.diffusion import create_diffusion
from Gpt.models.transformer import Gpt
from Gpt.vis import synth
from Gpt.utils import (
    requires_grad
)

from ldm.util import instantiate_from_config
from weight_diffusion.execution.util import load_model_from_config
from weight_diffusion.data.data_utils.helper import generate_checkpoints_from_weights
from weight_diffusion.ofga.sampling import sample_from_prompt
from weight_diffusion.data.gpt_dataset import GptDataset


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

def new_prompt_from_results_dict(results_dict, device):
    return torch.Tensor([[
        results_dict['train_acc'], 
        results_dict['validation_acc'], 
        results_dict['test_acc']
        ]]).to(device)
        
        
def sample_checkpoint_from_gpt(sampling_config, model_config, layer_list, diffusion, model, dataset):
    sampled_mnist_model_checkpoints_dict = {}
    targets_dict = {}
    prev_checkpoint = "hello"
    params0 = dataset[0]['parameters_0']
    print(dataset[0]['validation_loss_0'])
    params0_loss = dataset[0]['validation_loss_0']

    for prompt_statistics in [
        v for _, v in sampling_config.evaluation_prompt_statistics.items()
    ]:
        loss_prev = [params0_loss]
        target = prompt_statistics['validation_loss']
        loss_targets = [target]
        w_prev = [params0]
        print('loss_prev', loss_prev)
        print('loss_targets', loss_targets)

        targets_dict[str(target)] = prompt_statistics
        sampled_weights = synth(diffusion, model, torch.Tensor(loss_targets).view(-1, 1).cuda(),  torch.Tensor(loss_prev).view(-1, 1).cuda(), torch.stack(w_prev).cuda())

        unnormalised_sampled_weights = dataset.unnormalize(sampled_weights)
        #print('un', unnormalised_sampled_weights)
        print('normalised same', torch.all(sampled_weights.eq(unnormalised_sampled_weights)))
        
        sampled_checkpoint = generate_checkpoints_from_weights(
            unnormalised_sampled_weights, model_config, layer_list
        )
        #print('sampled', sampled_checkpoint)
        sampled_mnist_model_checkpoints_dict[str(target)] = sampled_checkpoint
        #print('sampled_checkpoint', sampled_checkpoint)

        if not isinstance(prev_checkpoint, str):
            print('same weights', torch.all(prev_weights.eq(unnormalised_sampled_weights)))
            #print('same checkpoint', torch.all(prev_checkpoint.eq(sampled_checkpoint)))

        prev_weights = unnormalised_sampled_weights
        prev_checkpoint = sampled_checkpoint

    return sampled_mnist_model_checkpoints_dict, targets_dict


def sample_checkpoints_from_ldm(
    sampling_config,
    model_config,
    layer_list,
    ldm,
    encoder,
    # tokenizer,
    device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, float]]]:
    sampled_mnist_model_checkpoints_dict = {}
    targets_dict = {}
    for steps in sampling_config.sampling_steps:
        for prompt_key, prompt_statistics in sampling_config.evaluation_prompt_statistics_ldm.items():
            prompt = new_prompt_from_results_dict(prompt_statistics, device)
            prompt_latent_rep = prompt
            uc = torch.Tensor([
                [0.1, 0.1, 0.1]
            ]
            ).to(device)

            print("################    PROMPT       ###############")
            print(prompt_latent_rep)
            print("################################################")
            sampled_weights_latent = sample_from_prompt(
                prompt=prompt_latent_rep,
                model=ldm,
                sampling_steps=steps,
                shape=tuple(sampling_config.shape),
                guidance_scale=1.0,
                uc=uc,
            )
            print("################    LATENT       ###############")
            print(sampled_weights_latent)
            print("################################################")

            sampled_weights = encoder.forward_decoder(sampled_weights_latent)
            print("################    WEIGHTS       ###############")
            print(sampled_weights)
            print("################################################")
            sampled_checkpoint = generate_checkpoints_from_weights(
                sampled_weights, model_config, layer_list
            )
            sampling_steps_string = f"_sampling_steps_{steps}"
            sampled_mnist_model_checkpoints_dict[prompt_key + sampling_steps_string] = sampled_checkpoint
            # Return dictionary containing target metrics for each prompt
            targets_dict[prompt_key + sampling_steps_string] = prompt_statistics

    return sampled_mnist_model_checkpoints_dict, targets_dict


def instantiate_encoder(encoder_config):
    encoder = instantiate_from_config(encoder_config)
    hyper_representations_path = Path(encoder_config["encoder_checkpoint_path"])
    encoder_checkpoint_path = hyper_representations_path.joinpath("checkpoint_ae.pt")
    encoder_checkpoint = torch.load(encoder_checkpoint_path, map_location=encoder_config["device"])
    encoder.model.load_state_dict(encoder_checkpoint)
    return encoder


def initiate_tokenizer(tokenizer_config):
    tokenizer = instantiate_from_config(tokenizer_config)
    return tokenizer


def instantiate_ldm(ldm_config):
    ldm_checkpoint_path = Path(ldm_config.ldm_checkpoint_path)
    ldm = load_model_from_config(ldm_config, ldm_checkpoint_path)
    return ldm


def instantiate_gpt(gpt_config):

    # Diffusion objects
    diffusion = create_diffusion(
        learn_sigma=False,
        predict_xstart=gpt_config.transformer.predict_xstart,
        noise_schedule="linear",
        steps=1000,
    )
    
    # Construct datasets
    dataset_split_ratios = [7, 3]
    test_dataset = GptDataset(
        data_dir=Path(gpt_config.dataset.path),
        checkpoint_property_of_interest=gpt_config.dataset.train_metric,
        openai_coefficient=gpt_config.dataset.openai_coefficient,
        split="test",
        dataset_split_ratios=dataset_split_ratios,
        use_permutation=False,
    )

    # Construct the model and optimizer
    model = Gpt(
        parameter_sizes=test_dataset.parameter_sizes,
        parameter_names=test_dataset.parameter_names,
        predict_xstart=gpt_config.transformer.predict_xstart,
        absolute_loss_conditioning=gpt_config.transformer.absolute_loss_conditioning,
        chunk_size=gpt_config.transformer.chunk_size,
        split_policy=gpt_config.transformer.split_policy,
        max_freq_log2=gpt_config.transformer.max_freq_log2,
        num_frequencies=gpt_config.transformer.num_frequencies,
        n_embd=gpt_config.transformer.n_embd,
        encoder_depth=gpt_config.transformer.encoder_depth,
        decoder_depth=gpt_config.transformer.decoder_depth,
        n_layer=gpt_config.transformer.n_layer,
        n_head=gpt_config.transformer.n_head,
        attn_pdrop=gpt_config.transformer.dropout_prob,
        resid_pdrop=gpt_config.transformer.dropout_prob,
        embd_pdrop=gpt_config.transformer.dropout_prob,
    )

    # Resume from checkpoint
    resume_checkpoint = torch.load('/netscratch2/kreynisson/weight-diffusion/src/weight_diffusion/baselines/results/dgx2_zoo_mnist/checkpoints/best.pt', 
    map_location=lambda storage, loc: storage)
    model.load_state_dict(resume_checkpoint['G'])

    return diffusion, model, test_dataset

def log_dictionary_locally(logging_dict: Dict, logging_path: str):
    logging_path: Path = Path(logging_path)
    logging_path.parent.mkdir(parents=True, exist_ok=True)
    with logging_path.open("w") as convert_file:
        convert_file.write(json.dumps(logging_dict))


def load_logging_dict(logging_path: str):
    logging_path: Path = Path(logging_path)
    with logging_path.open("r") as convert_file:
        logging_dict = convert_file.read()
    return json.loads(logging_dict)
