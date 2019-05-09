import click

import torch
from torch import cuda
from torch.utils.data import DataLoader

from model.train import train
from nn.unet import UNet
from dataset.cityscapes import CityscapesDataset
from dataset.split import split_dataset

from classes import feature_classes

desired_precision = .5


@click.command()
@click.option('--load-model', '-m', default=None)
@click.option('--dataset-dir', '-d', default='../dataset')
@click.option('--sample-rate', '-s', default=0.2)
@click.option('--no-train', is_flag=True, default=False)
@click.option('--no-test', is_flag=True, default=False)
@click.option('--epochs', '-e', default=5)
@click.option('--batch-size', '-b', default=100)
@click.option('--learning-rate', '-l', default=0.01)
@click.option('--logs-dir', default='../logs')
@click.option('--output-dir', default='../output')
def main(load_model: str,
         dataset_dir: str, sample_rate: float,
         no_train: bool, no_test: bool,
         epochs: int, batch_size: int, learning_rate: float,
         logs_dir: str, output_dir: str):
    device = torch.device('cuda:0' if cuda.is_available() else 'cpu')
    click.secho('Using device={}'.format(device), fg='blue')

    net = UNet((3, 256, 256), feature_classes.shape[0])
    net.to(device)

    if load_model is not None:
        click.secho('Loading model from \'{}\''.format(load_model), fg='yellow')
        net.load_state_dict(torch.load(load_model, map_location=device))

    # Load dataset
    click.echo('Loading dataset from \'{}\', using {}% as validation dataset'.format(dataset_dir, sample_rate * 100))

    dataset = CityscapesDataset(root=dataset_dir, classes=feature_classes)
    train_dataset, test_dataset = split_dataset(dataset, sample_rate)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    if not no_train:
        click.echo('Training model')
        net.train()
        train(net, train_loader, num_epochs=epochs, learning_rate=learning_rate)

    if not no_test:
        click.echo('Testing model')
        net.eval()


if __name__ == '__main__':
    main()
