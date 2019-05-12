import click

import torch
from torch import cuda
from torch.utils.data import DataLoader

import callbacks
from dataset import CityscapesDataset, CombinedDataset, FlippedDataset, split_dataset
from nn.classifier import Classifier
from nn.unet import UNet

from classes import cityscape_classes

desired_accuracy = .5


@click.command()
@click.option('--load-model', '-m', default=None)
@click.option('--dataset-dir', '-d', default='../dataset')
@click.option('--no-flips', '-f', is_flag=True, default=False)
@click.option('--sample-rate', '-r', default=0.1)
@click.option('--no-train', is_flag=True, default=False)
@click.option('--no-test', is_flag=True, default=False)
@click.option('--epochs', '-e', default=5)
@click.option('--batch-size', '-b', default=10)
@click.option('--learning-rate', '-l', default=0.01)
@click.option('--logs-dir', default='../logs')
@click.option('--output-dir', default='../output')
def main(load_model: str,
         dataset_dir: str, no_flips: bool, sample_rate: float,
         no_train: bool, no_test: bool,
         epochs: int, batch_size: int, learning_rate: float,
         logs_dir: str, output_dir: str):
    device = torch.device('cuda:0' if cuda.is_available() else 'cpu')
    click.secho('Using device={}'.format(device), fg='blue')

    net = UNet(in_channels=3, classes_count=len(cityscape_classes))
    net.to(device)

    if load_model is not None:
        click.secho('Loading model from \'{}\''.format(load_model), fg='yellow')
        net.load_state_dict(torch.load(load_model, map_location=device))

    # Load dataset
    click.echo('Loading dataset from \'{}\', using {}% as validation dataset'.format(dataset_dir, sample_rate * 100))

    dataset = CityscapesDataset(root=dataset_dir, classes=cityscape_classes)
    train_dataset, test_dataset = split_dataset(dataset, sample_rate)

    # Augument train set by horizontal flips
    if not no_flips:
        train_dataset = CombinedDataset(train_dataset, FlippedDataset(train_dataset))

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    save_callback = callbacks.SaverCallback(output_path=output_dir)
    classifier = Classifier(net, lr=learning_rate, train_callbacks=(save_callback,))

    if not no_train:
        click.secho('Training model', fg='blue')
        net.train()
        classifier.train(train_loader, test_loader, epochs)

    if not no_test:
        click.secho('Testing model', fg='blue')
        net.eval()
        accuracy = classifier.test(test_loader)
        color = 'green' if accuracy > desired_accuracy else 'red'
        click.secho('Accuracy={}'.format(accuracy), fg=color)


if __name__ == '__main__':
    main()
