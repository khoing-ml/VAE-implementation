from typing import Dict, Tuple

import torch
from torch import Tensor, nn


class VAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = None,
        latent_dim: int = 512,
        image_size: int = 64,
        downsample_factor: int = 8,
        hidden_dims=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.z_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)

        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.encoder(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        # clamp log-variance to avoid numerical overflow when exponentiating
        logvar_clamped = torch.clamp(logvar, min=-30.0, max=20.0)
        std = torch.exp(0.5 * logvar_clamped)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        x = self.decoder_input(z)
        x = x.view(-1, 256, 8, 8)
        x = self.decoder(x)
        x = self.final_layer(x)
        return x

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, x, mu, logvar

    def loss_function(self, *args, **kwargs) -> Dict[str, Tensor]:
        recons = args[0]
        input_x = args[1]
        mu = args[2]
        logvar = args[3]
        kld_weight = kwargs["M_N"]

        recons_loss = torch.nn.functional.mse_loss(recons, input_x)
        # clamp logvar used in KLD to keep values numerically stable
        logvar_clamped = torch.clamp(logvar, min=-30.0, max=20.0)
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp(), dim=1),
            dim=0,
        )
        loss = recons_loss + kld_loss * kld_weight
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": kld_loss.detach(),
        }

    def sample(self, num_samples: int, current_device: torch.device, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.z_dim, device=current_device)
        return self.decode(z)

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]



