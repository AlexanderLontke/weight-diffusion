import os
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data.dataset import T_co
from tqdm import tqdm

from weight_diffusion.data.data_utils.helper import get_flat_params
from weight_diffusion.data.modelzoo_dataset import ModelZooDataset

from ldm.util import instantiate_from_config


class ModelZooWithLatentDataset(ModelZooDataset):
    def __init__(
        self,
        data_dir: Path,
        encoder_config: Dict,
        device: str,
        tokenizer_config: Dict,
        prompt_embedding_max_length=77,
        **kwargs,
    ):
        self.device = device

        self.encoder = instantiate_from_config(encoder_config)
        self.hyper_representations_path = Path(
            encoder_config["encoder_checkpoint_path"]
        )
        encoder_checkpoint_path = self.hyper_representations_path.joinpath(
            "checkpoint_ae.pt"
        )
        encoder_checkpoint = torch.load(encoder_checkpoint_path, map_location=torch.device(self.device))
        self.encoder.model.load_state_dict(encoder_checkpoint)

        self.tokenizer = instantiate_from_config(tokenizer_config)
        self.prompt_embedding_max_length = prompt_embedding_max_length

        super().__init__(data_dir=Path(data_dir), **kwargs)

        # Create dataset index
        self.index_dict = {}  # Dict[index nr.] = (model Nr., Checkpoint Nr.)
        self.count = 0
        for model_key, checkpoints_dict in tqdm(
            self.checkpoints_dict.items(), desc="Indexing dataset"
        ):
            n_checkpoints = len(checkpoints_dict.keys())
            for checkpoint_key in range(1, n_checkpoints):
                for permutation_key in range(self.number_of_permutations):
                    self.index_dict[self.count] = (
                        model_key,
                        checkpoint_key,
                        permutation_key,
                    )
                    self.count += 1

        # Delete encoder from memory to free up space
        del self.encoder
        del self.tokenizer

    def __getitem__(self, index) -> T_co:
        model_key, checkpoint_key, permutation_key = self.index_dict[index]

        checkpoint_dict = self.checkpoints_dict[model_key][checkpoint_key]

        checkpoint_latent_rep = checkpoint_dict[0][permutation_key]
        prompt_latent_rep = checkpoint_dict[1]
        return {
            "checkpoint_latent": checkpoint_latent_rep,
            "prompt_latent": prompt_latent_rep,
        }

    def _get_prompt(self, checkpoint_progress, prompt_path):
        # Fetch latent representation from storage or generate a new one and save it
        prompt = torch.Tensor([checkpoint_progress['train_acc'], 
                  checkpoint_progress['validation_acc'], 
                  checkpoint_progress['test_acc']])

        return prompt

    def _get_prompt_old(self, checkpoint_progress, prompt_path):
        # Fetch latent representation from storage or generate a new one and save it
        if os.path.isfile(prompt_path):
            prompt_latent_rep = torch.load(prompt_path)
        else:
            prompt = (
                f"The training loss is {checkpoint_progress['train_loss']:.4g}. "
                f"The training accuracy is {checkpoint_progress['train_acc']:.4g}. "
                f"The validation loss is {checkpoint_progress['validation_loss']:.4g}. "
                f"The validation accuracy is {checkpoint_progress['validation_acc']:.4g}. "
                f"The test loss is {checkpoint_progress['test_loss']:.4g}. "
                f"The test accuracy is {checkpoint_progress['test_acc']:.4g}. "
            )
            prompt_latent_rep = self.tokenizer(
                prompt,
                max_length=self.prompt_embedding_max_length,
                return_tensors="pt",
                padding="max_length",
            )["input_ids"]
            torch.save(prompt_latent_rep, prompt_path)
            print("GENERATED PROMPT", prompt_path)

        return prompt_latent_rep

    # parsing methods
    def _parse_checkpoint_directory(self, checkpoint_directory, model_directory):
        checkpoint_path = os.path.join(
            self.data_dir, model_directory, checkpoint_directory, "checkpoints"
        )

        checkpoint = torch.load(checkpoint_path)
        self._update_min_max_param_value(checkpoint)

        permuted_checkpoints_latent = {}
        for i in range(-1, self.number_of_permutations):
            checkpoint_latent_rep_path = os.path.join(
                self.data_dir,
                model_directory,
                checkpoint_directory,
                "checkpoints_latent_rep_v2" + "_p" + str(i),
            )

            generate = False

            try: 
                checkpoint_latent_rep = torch.load(checkpoint_latent_rep_path)
            except:
                print("Broken checkpoint", checkpoint_latent_rep_path)
                generate = True

            # Fetch latent representation from storage or generate a new one and save it
            if generate:
                # Need to convert from checkpoint to a list of checkpoints
                permuted_checkpoint = self.permutation.permute_checkpoint(checkpoint, i)
                flattened_permuted_checkpoint = get_flat_params(permuted_checkpoint)
                with torch.no_grad():
                    checkpoint_latent_rep = self.encoder.forward_encoder(
                        torch.unsqueeze(flattened_permuted_checkpoint, dim=0).to(
                            self.device
                        ),
                    )
                checkpoint_latent_rep = checkpoint_latent_rep.to("cpu")
                torch.save(checkpoint_latent_rep.to("cpu"), checkpoint_latent_rep_path)

            permuted_checkpoints_latent[i] = checkpoint_latent_rep

        return permuted_checkpoints_latent
