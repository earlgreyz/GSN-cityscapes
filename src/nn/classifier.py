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
    def __init__(self, net, lr=0.1, callbacks=None):
        self.net = net
        self.callbacks = callbacks
        self.optimizer = adam.Adam(net.parameters(), lr=lr)
        self.criterion = F.cross_entropy

    def train(self, train_loader, test_loader, num_epochs):
        for epoch in range(num_epochs):
            click.echo('Training epoch {}'.format(epoch))
            self.net.train()
            loss = self._train_epoch(epoch=epoch, loader=train_loader)
            click.echo('Testing epoch {}'.format(epoch))
            self.net.eval()
            accuracy, last_batch = self._test(test_loader)

            for callback in self.callbacks:
                callback(net=self.net, epoch=epoch, loss=loss, accuracy=accuracy, last_batch=last_batch)

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

        return running_loss.average

    def test(self, loader):
        accuracy, _ = self._test(loader)
        return accuracy

    def _test(self, loader):
        accuracy = RunningAverage()
        show_stats = lambda _: '[{:2f}]'.format(accuracy.average)

        last_batch = None

        with click.progressbar(loader, item_show_func=show_stats) as bar:
            for inputs, targets in bar:
                if cuda.is_available():
                    inputs, targets = inputs.to('cuda'), targets.to('cuda')

                outputs = self.net(inputs)
                flipped = self.net(torch.flip(inputs, dims=(3,)))
                averaged = (outputs + torch.flip(flipped, dims=(3,))) / 2
                _, predicted = torch.max(averaged.data, 1)

                n, h, w = targets.shape
                correct = (predicted == targets).sum(dim=(1, 2)).float() / (h * w)
                accuracy.update(correct.sum().item(), n)

                last_batch = (inputs, targets, predicted)

        return accuracy.average, last_batch