import torch
from torch import nn
import torch.nn.functional as F


def center_crop(layer, target_size):
    _, _, layer_height, layer_width = layer.size()
    diff_y = (layer_height - target_size[0]) // 2
    diff_x = (layer_width - target_size[1]) // 2
    return layer[:, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])]


class UNet(nn.Module):
    def __init__(self, in_channels, classes_count, filters=32):
        super().__init__()
        self.down1 = StackEncoder(in_channels, filters)
        self.down2 = StackEncoder(filters, filters * 2)
        self.down3 = StackEncoder(filters * 2, filters * 4)
        self.down4 = StackEncoder(filters * 4, filters * 8)

        self.up1 = StackDecoder(filters * 8, filters * 4)
        self.up2 = StackDecoder(filters * 4, filters * 2)
        self.up3 = StackDecoder(filters * 2, filters)

        self.last = nn.Conv2d(filters, classes_count, kernel_size=1)

    def forward(self, x):
        x = x_trace1 = self.down1(x)
        x = F.max_pool2d(x, 2)
        x = x_trace2 = self.down2(x)
        x = F.max_pool2d(x, 2)
        x = x_trace3 = self.down3(x)
        x = F.max_pool2d(x, 2)
        x = self.down4(x)

        x = self.up1(x, x_trace3)
        x = self.up2(x, x_trace2)
        x = self.up3(x, x_trace1)

        return self.last(x)


class StackEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        return x


class StackDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.encoder = StackEncoder(in_channels, out_channels)

    def forward(self, x, trace):
        x = self.conv(x)
        crop = center_crop(trace, x.shape[2:])
        x = torch.cat([x, crop], 1)
        return self.encoder(x)
