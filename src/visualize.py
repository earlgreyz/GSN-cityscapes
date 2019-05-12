import click
import torch
from torch import cuda

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from classes import cityscape_classes, cityscape_colormap_float
from dataset import FlippedDataset, CityscapesDataset
from nn.unet import UNet


@click.command()
@click.option('--dataset-dir', '-d', default='../dataset')
@click.option('--load-model', '-m', default='../model.dict')
@click.option('--items', '-i', default=1)
def visualize(dataset_dir: str, load_model: str, items: int):
    device = torch.device('cuda:0' if cuda.is_available() else 'cpu')
    click.secho('Using device={}'.format(device), fg='blue')
    net = UNet(in_channels=3, classes_count=len(cityscape_classes))
    click.secho('Loading model from \'{}\''.format(load_model), fg='yellow')
    net.load_state_dict(torch.load(load_model, map_location=device))

    dataset = CityscapesDataset(root=dataset_dir, classes=cityscape_classes)
    flipped_dataset = FlippedDataset(dataset)
    colormap = ListedColormap(cityscape_colormap_float)

    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    for item in range(items):
        sample, target = dataset[item]
        sample_f, target_f = flipped_dataset[item]

        fig = plt.figure(figsize=(20, 20))

        fig.add_subplot(2, 3, 1)
        plt.imshow(sample.transpose(0, 1).transpose(1, 2))
        plt.axis('off')

        fig.add_subplot(2, 3, 2)
        plt.imshow(target, cmap=colormap)
        plt.axis('off')

        fig.add_subplot(2, 3, 3)
        outputs = net(sample.unsqueeze(0))
        _, predicted = torch.max(outputs.data, 1)
        plt.imshow(predicted.squeeze(), cmap=colormap)

        fig.add_subplot(2, 3, 4)
        plt.imshow(sample_f.transpose(0, 1).transpose(1, 2))
        plt.axis('off')

        fig.add_subplot(2, 3, 5)
        plt.imshow(target_f, cmap=colormap)
        plt.axis('off')

        fig.add_subplot(2, 3, 6)
        outputs_f = net(sample_f.unsqueeze(0))
        _, predicted_f = torch.max(outputs_f.data, 1)
        plt.imshow(predicted_f.squeeze(), cmap=colormap)
        plt.show()

    for i, (inputs, targets) in enumerate(loader):
        flipped = torch.flip(inputs, dims=(3,))
        fig = plt.figure(figsize=(20, 20))

        fig.add_subplot(2, 3, 1)
        plt.imshow(inputs.squeeze().transpose(0, 1).transpose(1, 2))
        plt.axis('off')

        fig.add_subplot(2, 3, 2)
        plt.imshow(targets.squeeze(), cmap=colormap)
        plt.axis('off')

        fig.add_subplot(2, 3, 3)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        plt.imshow(predicted.squeeze(), cmap=colormap)

        fig.add_subplot(2, 3, 4)
        plt.imshow(flipped.squeeze().transpose(0, 1).transpose(1, 2))
        plt.axis('off')

        fig.add_subplot(2, 3, 5)
        outputs_flipped = net(flipped)
        _, predicted = torch.max(outputs_flipped, 1)
        plt.imshow(predicted.squeeze(), cmap=colormap)
        plt.axis('off')

        fig.add_subplot(2, 3, 6)
        outputs_flipped = net(flipped)
        _, predicted = torch.max(torch.flip(outputs_flipped, dims=(3,)), 1)
        plt.imshow(predicted.squeeze(), cmap=colormap)
        plt.axis('off')

        plt.show()
        if i == items - 1:
            break

if __name__ == '__main__':
    visualize()