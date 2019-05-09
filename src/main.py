import click

import torch
from torch import cuda

from nn.unet import UNet

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

    net = UNet()
    net.to(device)

    if load_model is not None:
        click.secho('Loading model from \'{}\''.format(load_model), fg='yellow')
        net.load_state_dict(torch.load(load_model, map_location=device))

    # Load dataset
    click.echo('Loading dataset from \'{}\', sample rate is {}'.format(dataset_dir, sample_rate))

    #train_dataset, test_dataset = split_dataset(dataset, sample_rate)
    #train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    #test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    if not no_train:
        click.echo('Training model using \'{}\''.format(dataset_dir))
        net.train()

    if not no_test:
        click.echo('Testing model using \'{}\''.format(dataset_dir))
        net.eval()


if __name__ == '__main__':
    main()
