import torch.nn as nn
from lab4.utils.utils import GetModelAndLoaders



class Encoder(nn.Module):
    def __init__(self, batch_size, device):
        super(Encoder, self).__init__()
        model_n_loaders = GetModelAndLoaders(batch_size, device)
        model = model_n_loaders.get_best_model()

        self.first_conv = model.first_conv
        self.layers = model.layers
        self.adaptive_layers = model.adaptive_layers
        self.pooling_layers = model.pooling_layers


    def forward(self, x):
        x = self.first_conv(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.adaptive_layers[i](x)
            x = self.pooling_layers[i](x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, batch_size, device):
        super().__init__()
        self.encoder = Encoder(batch_size, device)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
