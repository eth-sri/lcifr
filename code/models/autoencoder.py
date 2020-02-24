import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, encoder_layers, decoder_layers):
        super().__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        for in_dim, out_dim in zip(encoder_layers[:-2], encoder_layers[1:-1]):
            self.encoder_layers.append(
                nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU()).to(device)
            )

        self.encoder_layers.append(
            nn.Linear(encoder_layers[-2], encoder_layers[-1]).to(device)
        )

        for in_dim, out_dim in zip(decoder_layers[:-2], decoder_layers[1:-1]):
            self.decoder_layers.append(
                nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU()).to(device)
            )

        self.decoder_layers.append(
            nn.Linear(decoder_layers[-2], decoder_layers[-1]).to(device)
        )

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        for layer in self.encoder_layers:
            x = layer(x)

        return x

    def decode(self, x):
        for layer in self.decoder_layers:
            x = layer(x)

        return x
