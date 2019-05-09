import click
from torch import cuda
from torch.optim import adam

from nn.criterion import BCELoss2d


def train(net, loader, num_epochs, learning_rate):
    criterion = BCELoss2d()
    optimizer = adam.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        running_loss = 0.0
        show_loss = lambda _: '[{}, {:3f}]'.format(epoch + 1, running_loss)

        with click.progressbar(loader, item_show_func=show_loss) as bar:
            for inputs, labels in bar:
                if cuda.is_available():
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()