import click
import numpy as np
import torch
from torch import cuda

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from classes import cityscape_classes, cityscape_colormap_float
from dataset import FlippedDataset, CityscapesDataset
from nn.unet import UNet


def apply_colors(image, colormap):
    W, H = image.shape
    applied = np.zeros((W, H, 3))
    for x in range(W):
        for y in range(H):
            applied[x, y, :] = np.array(colormap[image[x, y]])
    return applied


@click.command()
@click.option('--dataset-dir', '-d', default='../dataset')
@click.option('--load-model', '-m', default='../output/20190513-171812')
@click.argument('items', nargs=-1, type=int)
def visualize(dataset_dir: str, load_model: str, items: [int]):
    device = torch.device('cuda:0' if cuda.is_available() else 'cpu')
    click.secho('Using device={}'.format(device), fg='blue')
    net = UNet(in_channels=3, classes_count=len(cityscape_classes))
    click.secho('Loading model from \'{}\''.format(load_model), fg='yellow')
    net.load_state_dict(torch.load(load_model, map_location=device))

    dataset = CityscapesDataset(root=dataset_dir, classes=cityscape_classes)
    flipped_dataset = FlippedDataset(dataset)
    correct_colormap = ListedColormap([(1., 0, 0), (0, 1., 0)])

    for item in items:
        sample, target = dataset[item]
        sample_f, target_f = flipped_dataset[item]

        fig = plt.figure(figsize=(15, 10))

        fig.add_subplot(2, 3, 1)
        plt.title('Sample\n')
        plt.imshow(sample.transpose(0, 1).transpose(1, 2))
        plt.imshow(apply_colors(target, cityscape_colormap_float), alpha=.5)
        plt.axis('off')

        fig.add_subplot(2, 3, 2)
        plt.title('Prediction\n')
        plt.imshow(sample.transpose(0, 1).transpose(1, 2))
        outputs = net(sample.unsqueeze(0))
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.squeeze()
        plt.imshow(apply_colors(predicted, cityscape_colormap_float), alpha=.5)
        plt.axis('off')

        fig.add_subplot(2, 3, 3)
        plt.title('Prediction correctness\n')
        correct = (predicted == target)
        plt.imshow(sample.transpose(0, 1).transpose(1, 2))
        plt.imshow(correct, alpha=.25, cmap=correct_colormap)
        plt.axis('off')

        fig.add_subplot(2, 3, 4)
        plt.title('Flipped sample\n')
        plt.imshow(sample_f.transpose(0, 1).transpose(1, 2))
        plt.imshow(apply_colors(target_f, cityscape_colormap_float), alpha=.5)
        plt.axis('off')

        fig.add_subplot(2, 3, 5)
        plt.title('Prediction on flipped sample\n')
        plt.imshow(sample_f.transpose(0, 1).transpose(1, 2))
        outputs_f = net(sample_f.unsqueeze(0))
        _, predicted_f = torch.max(outputs_f.data, 1)
        predicted_f = predicted_f.squeeze()
        plt.imshow(apply_colors(predicted_f, cityscape_colormap_float), alpha=.5)
        plt.axis('off')

        fig.add_subplot(2, 3, 6)
        plt.title('Prediction correctness\n')
        plt.imshow(sample_f.transpose(0, 1).transpose(1, 2))
        correct = (predicted_f == target_f)
        plt.imshow(correct, alpha=.25, cmap=correct_colormap)
        plt.axis('off')

        plt.show()


if __name__ == '__main__':
    visualize()
