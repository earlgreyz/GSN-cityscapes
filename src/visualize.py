import click

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from classes import cityscape_classes, cityscape_colormap_float
from dataset.cityscapes import CityscapesDataset

@click.command()
@click.option('--dataset-dir', '-d', default='../dataset')
@click.argument('items', nargs=-1, type=int)
def visualize(dataset_dir: str, items: [int]):
    dataset = CityscapesDataset(root=dataset_dir, classes=cityscape_classes)
    n = len(dataset) // 2
    colormap = ListedColormap(cityscape_colormap_float)

    for item in items:
        sample, target = dataset[item]
        sample_r, target_r = dataset[n + item]

        fig = plt.figure(figsize=(20, 20))

        fig.add_subplot(2, 2, 1)
        plt.imshow(sample.transpose(0, 1).transpose(1, 2))
        plt.axis('off')

        fig.add_subplot(2, 2, 2)
        plt.imshow(target, cmap=colormap)
        plt.axis('off')

        fig.add_subplot(2, 2, 3)
        plt.imshow(sample_r.transpose(0, 1).transpose(1, 2))
        plt.axis('off')

        fig.add_subplot(2, 2, 4)
        plt.imshow(target_r, cmap=colormap)
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    visualize()