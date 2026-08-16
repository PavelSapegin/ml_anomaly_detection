from pathlib import Path

import torch
import torch.nn as nn


class Autoencoder(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim,16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )


    def forward(self, x):
        x = self.encoder(x)
        decoded = self.decoder(x)

        return decoded


def load_model(weights_path: Path, input_dim: int, latent_dim: int) -> Autoencoder:
    autoencoder = Autoencoder(input_dim, latent_dim)
    state_dict = torch.load(weights_path, map_location="cpu")
    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()

    return autoencoder
