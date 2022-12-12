import torch
from ghrp.model_definitions.def_net import NNmodule
from torchvision import transforms
from torchvision import datasets

def instantiate_MNIST_CNN(mnist_cnn_config, checkpoint, device):
    mnist_cnn = NNmodule(mnist_cnn_config, cuda=(device == "cuda"), verbosity=0)
    mnist_cnn.model.load_state_dict(checkpoint)
    mnist_cnn = mnist_cnn.to(torch.device(device))
    return mnist_cnn


def get_evaluation_datasets(evaluation_dataset_config):
    # Same seed as in
    # https://github.com/ModelZoos/ModelZooDataset/blob/main/code/zoo_generators/train_zoo_mnist_uniform.py
    dataset_seed = 42
    evaluation_datasets = {}
    # load raw dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    val_and_trainset_raw = datasets.MNIST(
        evaluation_dataset_config.data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    testset_raw = datasets.MNIST(
        evaluation_dataset_config.data_dir,
        train=False,
        download=True,
        transform=transform,
    )
    trainset_raw, valset_raw = torch.utils.data.random_split(
        val_and_trainset_raw,
        [50000, 10000],
        generator=torch.Generator().manual_seed(dataset_seed),
    )

    # temp dataloaders
    evaluation_datasets["train"] = torch.utils.data.DataLoader(
        dataset=trainset_raw, batch_size=len(trainset_raw), shuffle=True
    )
    evaluation_datasets["validation"] = torch.utils.data.DataLoader(
        dataset=valset_raw, batch_size=len(valset_raw), shuffle=True
    )
    evaluation_datasets["test"] = torch.utils.data.DataLoader(
        dataset=testset_raw, batch_size=len(testset_raw), shuffle=True
    )

    return evaluation_datasets


def finetune_MNIST_CNN(model: NNmodule, epochs, train_dataloader, prompt):
    training_losses = []
    training_accuracy = []
    for epoch in range(epochs):
        loss_runing, accuracy = model.train_epoch(train_dataloader, epoch)
        training_losses += [loss_runing]
        training_accuracy += [accuracy]
    return model, {
        "train_running_loss": training_losses,
        "train_running_accuracy": training_accuracy,
    }


def evaluate_MNIST_CNN(model: NNmodule, evaluation_datasets):
    evaluation_dict = {}

    for key, dataloader in evaluation_datasets.items():
        overall_loss, overall_accuracy = model.test_epoch(dataloader, epoch=-1)
        evaluation_dict[f"{key}_loss"] = overall_loss
        evaluation_dict[f"{key}_acc"] = overall_accuracy

    return evaluation_dict
