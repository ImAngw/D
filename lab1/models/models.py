import torch.nn as nn
from lab1.models.sublayers import ResidualBlock


class MLP(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.dropout = nn.Dropout(configs.dropout)

        self.layers = nn.ModuleList([
            self._return_block(
                in_dim=self.configs.dim_hidden_layers[i][0],
                expansion_factor=configs.expansion_factor
            ) for i in range(configs.depth)
        ])

        self.adaptive_layers = nn.ModuleList([
            self._return_adaptive_layer(
                in_dim=self.configs.dim_hidden_layers[i][0],
                out_dim=self.configs.dim_hidden_layers[i][1]
            ) for i in range(configs.depth)
        ])

        self.classification_head = nn.Linear(self.configs.dim_hidden_layers[-1][1], 10)


    def forward(self, x):
        for i, layer in enumerate(self.layers):
            residual = x
            x = layer(x)
            if self.configs.require_skip_connection:
                x = x + residual
            x = self.adaptive_layers[i](x)

        output = self.classification_head(x)
        return output

    def _return_block(self, in_dim, expansion_factor=1):
        # hidden_dim = in_dim * expansion_factor
        hidden_dim = 128
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            self.dropout,

            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, in_dim),
        )

    def _return_adaptive_layer(self, in_dim, out_dim):
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            self.dropout
        )

class CNN(nn.Module):
    def __init__(self, dataset, channels_hidden_conv, img_reductions, require_skip_connection, depth):
        super().__init__()
        self.hidden_dim = 128

        in_channels = 3 if dataset == 'cifar10' else 1
        first_lin_dim = channels_hidden_conv[-1][1] # * (self.configs.img_reductions[-1]**2)

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels_hidden_conv[0][0], kernel_size=7, padding=3, stride=1, bias=False),
            nn.BatchNorm2d(channels_hidden_conv[0][0]),
            nn.ReLU()
        )

        self.layers = nn.ModuleList([
            ResidualBlock(
                in_channels=channels_hidden_conv[i][0],
                n_blocks=2,
                skip_connection=require_skip_connection
            ) for i in range(depth)
        ])

        self.adaptive_layers = nn.ModuleList([
            self._return_adaptive_layer(
                in_channels=channels_hidden_conv[i][0],
                out_channel=channels_hidden_conv[i][1]
            ) for i in range(depth)
        ])

        self.pooling_layers = nn.ModuleList([
            nn.AdaptiveAvgPool2d(img_reductions[i]) for i in range(depth)
        ])
        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        self.classification_head = nn.Sequential(
            nn.LayerNorm(first_lin_dim),
            nn.LayerNorm(first_lin_dim),
            nn.Linear(first_lin_dim, 10)
        )

    def forward(self, x):
        x = self.first_conv(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.adaptive_layers[i](x)
            x = self.pooling_layers[i](x)

        x = self.global_pooling(x).squeeze(-1).squeeze(-1)
        output = self.classification_head(x)
        return output

    @staticmethod
    def _return_adaptive_layer(in_channels, out_channel, k_size=1, padding=0, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channel, kernel_size=k_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

