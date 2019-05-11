import torch
from torch import nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels, classes_count, filters=32):
        super().__init__()
        self.down1 = UNetConvBlock(in_channels, filters, True, True)
        self.down2 = UNetConvBlock(filters, filters * 2, True, True)
        self.down3 = UNetConvBlock(filters * 2, filters * 4, True, True)
        self.down4 = UNetConvBlock(filters * 4, filters * 8, True, True)

        self.up1 = UNetUpBlock(filters * 8, filters * 4, 'upconv', True, True)
        self.up2 = UNetUpBlock(filters * 4, filters * 2, 'upconv', True, True)
        self.up3 = UNetUpBlock(filters * 2, filters, 'upconv', True, True)

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


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
               :, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])
               ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out
