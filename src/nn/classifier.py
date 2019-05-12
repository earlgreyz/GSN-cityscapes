import click
import torch
from torch import cuda
from torch.optim import adam
import torch.nn.functional as F


class RunningAverage:
    def __init__(self):
        self.sum = 0.0
        self.count = 0
        self.average = 0.0

    def update(self, x, n=1):
        self.sum += x
        self.count += n
        self.average = self.sum / self.count


class Classifier:
    def __init__(self, net, lr=0.1, threshold=0.5):
        self.net = net
        self.optimizer = adam.Adam(net.parameters(), lr=lr)
        self.criterion = F.cross_entropy
        self.threshold = threshold

    def train(self, train_loader, test_loader, num_epochs):
        for epoch in range(num_epochs):
            click.echo('Training epoch {}'.format(epoch))
            self.net.train()
            self._train_epoch(epoch=epoch, loader=train_loader)
            click.echo('Testing epoch {}'.format(epoch))
            self.net.eval()
            self.test(test_loader)


    def _train_epoch(self, epoch, loader):
        running_loss = RunningAverage()
        show_stats = lambda _: '[{}, {:3f}]'.format(epoch + 1, running_loss.average)

        with click.progressbar(loader, item_show_func=show_stats) as bar:
            for inputs, targets in bar:
                if cuda.is_available():
                    inputs, targets = inputs.to('cuda'), targets.to('cuda')

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss.update(loss.item())

    def test(self, loader):
        accuracy = RunningAverage()
        show_stats = lambda _: '[{:2f}]'.format(accuracy.average)

        with click.progressbar(loader, item_show_func=show_stats) as bar:
            for inputs, targets in bar:
                if cuda.is_available():
                    inputs, targets = inputs.to('cuda'), targets.to('cuda')

                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                n, h, w = targets.shape
                correct = (predicted == targets).sum(dim=(1, 2)).float() / (h * w)
                accuracy.update(correct.sum().item(), n)

        return accuracy.average