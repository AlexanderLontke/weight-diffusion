import torch
from typing import List, Any, Collection


def get_flat_params(state_dict):
    parameters = []
    for parameter in state_dict.values():
        parameters.append(parameter.flatten())
    return torch.cat(parameters)


def get_param_sizes(state_dict):
    return torch.tensor([p.numel() for p in state_dict.values()], dtype=torch.long)


def perform_train_test_validation_split(
    list_to_split: List[Any], dataset_split_ratios: Collection[float], split: str
):
    # Perform Train-Test(-Validation) Split
    assert sum(dataset_split_ratios) == 10, "Dataset split ratios do not add up to 10"

    # Convert to absolute amounts
    n = len(list_to_split)
    dataset_split_ratios = [int(ratio / 10.0 * n) for ratio in dataset_split_ratios]

    # Check if validation split is included or not
    if len(dataset_split_ratios) == 2:
        number_of_training_items, number_of_testing_items = dataset_split_ratios
        number_of_validation_items = None
    elif len(dataset_split_ratios) == 3:
        (
            number_of_training_items,
            number_of_validation_items,
            number_of_testing_items,
        ) = dataset_split_ratios
    else:
        print(
            f"List giving dataset split ratios must have length 2 or three but"
            f"given ratios had length {len(dataset_split_ratios)}. "
            f"Loading 100% of dataset"
        )
        return list_to_split

    # Perform split
    if split == "train":
        list_to_split = list_to_split[:number_of_training_items]
    elif split == "test":
        list_to_split = list_to_split[-number_of_testing_items:]
    elif split == "val":
        if number_of_validation_items:
            list_to_split = list_to_split[
                number_of_training_items : number_of_training_items
                + number_of_validation_items
            ]
        else:
            raise ValueError(
                "validation split requested, but only two split ratios provided."
            )
    return list_to_split
