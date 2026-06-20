from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


def res_blocks(channels):
    return nn.Sequential(
        ResidualBlock(channels),
        ResidualBlock(channels),
        ResidualBlock(channels),
        ResidualBlock(channels),
    )


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 2, stride=2)
        self.blocks = res_blocks(out_channels)

    def forward(self, x):
        return self.blocks(F.relu(self.conv(x)))


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.blocks = res_blocks(out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = F.relu(x + skip)
        return self.blocks(x)


class DRUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, channels=(32, 64, 128, 256)):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 3, padding=1),
            res_blocks(channels[0]),
        )
        self.down_blocks = nn.ModuleList(
            [DownBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )
        self.up_blocks = nn.ModuleList(
            [
                UpBlock(channels[i + 1], channels[i])
                for i in reversed(range(len(channels) - 1))
            ]
        )
        self.last = nn.Conv2d(channels[0], out_channels, 3, padding=1)

    def forward(self, x):
        h, w = x.shape[-2:]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        x = self.first(x)
        skips = [x]
        for block in self.down_blocks:
            x = block(x)
            skips.append(x)
        for block, skip in zip(self.up_blocks, reversed(skips[:-1])):
            x = block(x, skip)
        x = self.last(x)
        if pad_h or pad_w:
            x = x[..., :h, :w]
        return x
