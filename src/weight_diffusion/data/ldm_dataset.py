import torch
from pathlib import Path
from tqdm import tqdm
from weight_diffusion.data.modelzoo_dataset import ModelZooDataset
from weight_diffusion.data.data_utils.helper import get_param_sizes, get_flat_params

class LDMDataset(ModelZooDataset):
    def __init__(
        self, data_dir: Path, checkpoint_property_of_interest: str, split: str, **kwargs
    ):
        super().__init__(
            data_dir=data_dir,
            checkpoint_property_of_interest=checkpoint_property_of_interest,
            split=split,
            **kwargs,
        )
