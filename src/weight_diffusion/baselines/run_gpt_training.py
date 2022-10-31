#!/usr/bin/env python3
"""
Script for training and evaluating G.pt models.
"""

import hydra
import omegaconf
from pathlib import Path
import os
import random
from copy import deepcopy
import torch
import torch.utils.data

from Gpt.diffusion import create_diffusion
from Gpt.diffusion.timestep_sampler import UniformSampler

from Gpt.distributed import scaled_all_reduce
from Gpt.models.transformer import Gpt
from Gpt.meters import TrainMeter, TestMeter
from Gpt.utils import (
    setup_env,
    construct_loader,
    shuffle,
    update_lr,
    spread_losses,
    accumulate,
    requires_grad,
)
from Gpt.distributed import get_rank, get_world_size, is_main_proc, synchronize

# from Gpt.vis import VisMonitor
# from Gpt.tasks import get
from Gpt.download import find_model

from weight_diffusion.data.gpt_dataset import GptDataset


def run_diffusion_vlb(cfg, diffusion, model, timestep_sampler, batch_dict):
    """
    Computes the diffusion training loss for a batch of inputs.
    """
    w_t, w_t1 = batch_dict["parameters_0"].cuda(), batch_dict["parameters_1"].cuda()
    loss_t, loss_t1 = (
        batch_dict[f"{cfg.dataset.train_metric}_0"].cuda(),
        batch_dict[f"{cfg.dataset.train_metric}_1"].cuda(),
    )
    t, vlb_weights = timestep_sampler.sample(w_t.shape[0], w_t.device)
    with torch.cuda.amp.autocast(enabled=cfg.amp):
        model_kwargs = {"loss_target": loss_t1, "loss_prev": loss_t, "x_prev": w_t}
        losses = diffusion.training_losses(model, w_t1, t, model_kwargs=model_kwargs)
    loss = (losses["loss"] * vlb_weights).mean()
    return loss, losses


def train_epoch(
    cfg,
    diffusion,
    model,
    model_module,
    ema,
    train_loader,
    timestep_sampler,
    optimizer,
    scaler,
    meter,
    epoch,
):
    """
    Performs one epoch of G.pt training.
    """
    shuffle(train_loader, epoch)
    model.train()
    meter.reset()
    meter.iter_tic()
    epoch_iters = len(train_loader)
    for batch_ind, batch_dict in enumerate(train_loader):
        lr = update_lr(cfg, optimizer, epoch + (batch_ind / epoch_iters))
        loss, loss_dict = run_diffusion_vlb(
            cfg, diffusion, model, timestep_sampler, batch_dict
        )
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.train.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if cfg.transformer.ema:
            accumulate(ema, model_module, cfg.train.ema_decay)
        loss_dict["loss"] = loss.view(1)
        loss_dict = {
            k: scaled_all_reduce(cfg, [v.mean()])[0].item()
            for k, v in loss_dict.items()
        }
        meter.iter_toc()
        meter.record_stats(loss_dict, lr)
        meter.log_iter_stats(epoch, batch_ind + 1)
        meter.iter_tic()
    meter.log_epoch_stats(epoch + 1)


def checkpoint_model(
    cfg, is_best_model, epoch, G_module, ema_module, optimizer, **save_dict
):
    """
    Save a G.pt checkpoint.
    """
    periodic_checkpoint = epoch % cfg.train.checkpoint_freq == 0
    if is_best_model or periodic_checkpoint:
        if is_main_proc():
            base_path = f"{cfg.out_dir}/{cfg.exp_name}/checkpoints"
            save_dict.update(
                {"G": G_module.state_dict(), "optim": optimizer.state_dict()}
            )
            if cfg.transformer.ema:
                save_dict.update({"G_ema": ema_module.state_dict()})
            if is_best_model:
                torch.save(save_dict, f"{base_path}/best.pt")
            if periodic_checkpoint:
                torch.save(save_dict, f"{base_path}/{epoch:04}.pt")
        synchronize()


@torch.inference_mode()
def test_epoch(cfg, diffusion, model, test_loader, timestep_sampler, meter, epoch):
    """
    Evaluate G.pt on test set (unseen) neural networks.
    """
    if (epoch + 1) % cfg.test.freq == 0:
        model.eval()
        meter.reset()
        for batch_ind, batch_dict in enumerate(test_loader):
            loss, loss_dict = run_diffusion_vlb(
                cfg, diffusion, model, timestep_sampler, batch_dict
            )
            loss_dict["loss"] = loss.view(1)
            loss_dict = {
                k: scaled_all_reduce(cfg, [v.mean()])[0].item()
                for k, v in loss_dict.items()
            }
            meter.record_stats(loss_dict)
        meter.log_epoch_stats(epoch + 1)


def train(cfg):
    """Performs the full training loop."""

    # Instantiate visualization objects (they will be fully-initialized later on)

    # Construct datasets
    train_dataset = GptDataset(
        data_dir=Path(cfg.dataset.path),
        checkpoint_property_of_interest=cfg.dataset.train_metric,
        openai_coeff=cfg.dataset.openai_coeff,
    )

    # Construct data loaders
    train_loader = construct_loader(
        train_dataset,
        cfg.train.mb_size,
        cfg.num_gpus,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.dataset.num_workers,
    )

    # Construct meters
    train_meter = TrainMeter(len(train_loader), cfg.train.num_ep)

    # Construct the model and optimizer
    model = Gpt(
        parameter_sizes=train_dataset.parameter_sizes,
        parameter_names=train_dataset.parameter_names,
        predict_xstart=cfg.transformer.predict_xstart,
        absolute_loss_conditioning=cfg.transformer.absolute_loss_conditioning,
        chunk_size=cfg.transformer.chunk_size,
        split_policy=cfg.transformer.split_policy,
        max_freq_log2=cfg.transformer.max_freq_log2,
        num_frequencies=cfg.transformer.num_frequencies,
        n_embd=cfg.transformer.n_embd,
        encoder_depth=cfg.transformer.encoder_depth,
        decoder_depth=cfg.transformer.decoder_depth,
        n_layer=cfg.transformer.n_layer,
        n_head=cfg.transformer.n_head,
        attn_pdrop=cfg.transformer.dropout_prob,
        resid_pdrop=cfg.transformer.dropout_prob,
        embd_pdrop=cfg.transformer.dropout_prob,
    )

    # Create an exponential moving average (EMA) of G.pt
    if cfg.transformer.ema:
        ema = deepcopy(model)
        requires_grad(ema, False)
    else:
        ema = None

    # Diffusion objects
    diffusion = create_diffusion(
        learn_sigma=False,
        predict_xstart=cfg.transformer.predict_xstart,
        noise_schedule="linear",
        steps=1000,
    )
    timestep_sampler = UniformSampler(diffusion)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    # Transfer model to GPU
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)
    if cfg.transformer.ema:
        ema = ema.cuda(device=cur_device)

    # Use DDP for multi-gpu training
    if (not cfg.test_only) and (cfg.num_gpus > 1):
        model = torch.nn.parallel.DistributedDataParallel(
            module=model,
            device_ids=[cur_device],
            output_device=cur_device,
            static_graph=True,
        )
        model.configure_optimizers = model.module.configure_optimizers
        module = model.module
    else:
        module = model

    # Initialize the EMA model with an exact copy of weights
    if cfg.transformer.ema and cfg.resume_path is None:
        accumulate(ema, module, 0)

    # Construct the optimizer
    optimizer = model.configure_optimizers(
        lr=cfg.train.base_lr, wd=cfg.train.wd, betas=(0.9, cfg.train.beta2)
    )

    # Resume from checkpoint
    if cfg.resume_path is not None:
        resume_checkpoint = find_model(cfg.resume_path)
        module.load_state_dict(resume_checkpoint["G"])
        if cfg.transformer.ema:
            ema.load_state_dict(resume_checkpoint["G_ema"])
        if not cfg.test_only:
            optimizer.load_state_dict(resume_checkpoint["optim"])
        try:
            start_epoch = int(os.path.basename(cfg.resume_path).split(".")[0])
        except ValueError:
            start_epoch = 0
        print(
            f"Resumed G.pt from checkpoint {cfg.resume_path}, using start_epoch={start_epoch}"
        )
    else:
        start_epoch = 0
        print("Training from scratch")

    print("Beginning training...")
    best_test_metric = float("-inf")
    for epoch in range(start_epoch, cfg.train.num_ep):
        train_epoch(
            cfg,
            diffusion,
            model,
            module,
            ema,
            train_loader,
            timestep_sampler,
            optimizer,
            scaler,
            train_meter,
            epoch,
        )

        # new_best_model = (test_metric is not None) and (test_metric > best_test_metric)
        # if new_best_model:
        #     best_test_metric = test_metric
        # checkpoint_model(cfg, new_best_model, epoch + 1, module, ema, optimizer)


def single_proc_train(local_rank, port, world_size, cfg):
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="tcp://localhost:{}".format(port),
        world_size=world_size,
        rank=local_rank,
    )
    torch.cuda.set_device(local_rank)
    train(cfg)
    torch.distributed.destroy_process_group()
    exit()


@hydra.main(config_path="../../../configs/train", config_name="config.yaml")
def main(cfg: omegaconf.DictConfig):

    # Multi-gpu training
    if cfg.num_gpus > 1:
        # Select a port for proc group init randomly
        port_range = [10000, 65000]
        port = random.randint(port_range[0], port_range[1])
        # Start a process per-GPU:
        torch.multiprocessing.start_processes(
            single_proc_train,
            args=(port, cfg.num_gpus, cfg),
            nprocs=cfg.num_gpus,
            start_method="spawn",
        )
    else:
        train(cfg)


if __name__ == "__main__":
    main()
