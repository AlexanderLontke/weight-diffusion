from typing import Dict, Tuple
from pathlib import Path
import torch
import json

from Gpt.diffusion import create_diffusion
from Gpt.models.transformer import Gpt
from Gpt.latent_walk_helpers import create_latent_walk_for_cnn, slerpify

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


def ddim_synth(
    diffusion,
    G,
    loss_target,        # The prompted loss/error/return: shape (N, 1)
    loss_prev,          # The starting loss/error/return: shape (N, 1)
    w_prev,             # The starting parameter vector: shape (N, D)
    **ddim_sample_loop_kwargs
):
    """
    Samples from G.pt via the reverse diffusion process using DDIM sampling.
    Specifically, this function draws a sample from p(theta^*|prompt_loss,starting_loss,starting_theta).
    """
    assert loss_target.size(0) == loss_prev.size(0) == w_prev.size(0)

    model_kwargs = {
        'loss_target': loss_target,
        'loss_prev': loss_prev,
        'x_prev': w_prev
    }

    shape = w_prev.shape
    sample = diffusion.ddim_sample_loop(
        G,
        shape,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        device='cuda',
        **ddim_sample_loop_kwargs
    )

    return sample



def sample_checkpoint_from_gpt(sampling_config, diffusion, model, dataset, **ddim_sample_loop_kwargs):
    loss_targets = []
    loss_prev = []
    use_fixed_input_theta = True
    n_samples = len(sampling_config.evaluation_prompt_statistics.items())
    # TODO change to config value
    n_steps = 120
    seed = 1337
    dim = dataset.get_run_network(run_index=0, iter=0, normalize=True, augment=False).size(0)
    run_indices = list(range(seed * n_samples, (seed + 1) * n_samples))
    noise = slerpify(torch.randn(1, n_samples, dim, device="cuda"), n_steps)  # Create looping noise
    nets, lps = [], []  # nets = input (starting) parameters, lps = starting losses/errors
    for run_index in run_indices:
        net = dataset.get_run_network(run_index=run_index, iter=0, normalize=True, augment=False).unsqueeze(0).cuda()
        lp = dataset.get_run_losses(run_index=run_index)[0].view(1).to("cuda")
        nets.append(net)
        lps.append(lp)
    nets = slerpify(torch.cat(nets, 0).view(1, n_samples, dim), n_steps)  # (n_videos, n_samples, n_steps, D)
    if use_fixed_input_theta:
        # Use the same starting parameters for every frame in the video:
        nets = nets[0, 0, 0].view(1, 1, 1, -1).repeat(1, n_samples, n_steps, 1)  # (n_videos, n_samples, n_steps, D)
    # Use a constant starting loss/error to better isolate the effect of sampling noise:

    sampled_mnist_model_checkpoints_dict = {}
    targets_dict = {}
    prev = -1

    for prompt_statistics in [
        v for _, v in sampling_config.evaluation_prompt_statistics.items()
    ]:
        if prev < 0:
            prev = prompt_statistics['test_loss']
            continue
        
        loss_prev.append(prev)
        prev = prompt_statistics['test_loss']
        loss_targets.append(prev)
    
    return ddim_synth(diffusion, model, torch.Tensor(loss_targets).view(-1, 1),  torch.Tensor(loss_prev).view(-1, 1), nets.view(-1, dim), noise=noise.view(-1, dim), progress=True)


def sample_checkpoints_from_ldm(
    sampling_config,
    model_config,
    layer_list,
    ldm,
    encoder,
    tokenizer,
    device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, float]]]:
    sampled_mnist_model_checkpoints_dict = {}
    targets_dict = {}
    for prompt_statistics in [
        v for _, v in sampling_config.evaluation_prompt_statistics.items()
    ]:
        prompt = new_prompt_from_results_dict(prompt_statistics, device)
        prompt_latent_rep = prompt
        uc = torch.Tensor([
            [0.1, 0.1, 0.1]
        ]
        ).to(device)

        sampled_weights_latent = sample_from_prompt(
            prompt=prompt_latent_rep,
            model=ldm,
            sampling_steps=sampling_config.sampling_steps,
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
        sampled_mnist_model_checkpoints_dict[str(prompt)] = sampled_checkpoint
        # Return dictionary containing target metrics for each prompt
        targets_dict[str(prompt)] = prompt_statistics

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
    train_dataset = GptDataset(
        data_dir=Path(gpt_config.dataset.path),
        checkpoint_property_of_interest=gpt_config.dataset.train_metric,
        openai_coefficient=gpt_config.dataset.openai_coefficient,
        split="train",
        dataset_split_ratios=dataset_split_ratios,
    )



    # Construct the model and optimizer
    model = Gpt(
        parameter_sizes=train_dataset.parameter_sizes,
        parameter_names=train_dataset.parameter_names,
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

    return diffusion, model, train_dataset

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
