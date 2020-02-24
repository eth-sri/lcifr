import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, layers):
        super().__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.layers = nn.ModuleList()

        for in_dim, out_dim in zip(layers[:-2], layers[1:-1]):
            self.layers.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU()).to(device))

        self.layers.append(nn.Linear(layers[-2], layers[-1])).to(device)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
