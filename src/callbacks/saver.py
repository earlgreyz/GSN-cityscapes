import os
import time

import click
import torch

from . import Callback

class SaverCallback(Callback):
    def __init__(self, output_path, prefix=''):
        self.output_path = output_path
        self.prefix = prefix

    def __call__(self, net, *args, **kwargs):
        suffix = time.strftime("%Y%m%d-%H%M%S")
        name = self.prefix + suffix
        click.secho('Saving model as \'{}\''.format(name), fg='yellow')
        torch.save(net.state_dict(), os.path.join(self.output_path, name))