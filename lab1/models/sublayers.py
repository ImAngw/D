import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, n_blocks=2, skip_connection=True):
        super().__init__()
        self.skip_connection = skip_connection
        self.conv_block = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),

                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
            ) for _ in range(n_blocks)
        ])
        self.relu = nn.ReLU()

    def forward(self, x):
        for block in self.conv_block:
            x = x + block(x) if self.skip_connection else block(x)
            x = self.relu(x)
        return x
