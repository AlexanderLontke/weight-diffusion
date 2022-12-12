"""
Implements sampling via Denoising Diffusion Implicit Model Sampler
"""
import time
import torch
from typing import Tuple
from tqdm import trange
from torch import autocast
from omegaconf import OmegaConf
from contextlib import nullcontext

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from weight_diffusion.execution.util import load_model_from_config


def sample_from_prompt(
    prompt: str,
    model,
    sampling_steps: int,
    shape: Tuple,
    uc,
    sampler_type: str = "DDIM",
    guidance_scale: float = 7.5,
    use_autocast_precision: bool = True,
    ddim_eta: float = 0.0,  # DDIM ETA 0.0 corresponds to deterministic sampling
):
    # Instantiate Sampler
    if sampler_type == "DPM":
        sampler = DPMSolverSampler(model)
    elif sampler_type == "PLMS":
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    precision_scope = autocast if use_autocast_precision else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                if guidance_scale == 1.0:
                    uc = None
                c = model.get_learned_conditioning(prompt)
                samples_ddim, _ = sampler.sample(
                    S=sampling_steps,
                    conditioning=c,
                    batch_size=1,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=guidance_scale,
                    unconditional_conditioning=uc,
                    eta=ddim_eta,
                )
    return samples_ddim
