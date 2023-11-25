import torch
import torch.nn as nn

from implicit.common import PLANE_TYPES


class VaeBtlneck(nn.Module):
    def __init__(self,
                 c_dim=64,
                 z_dim=32,
                 plane_resolution=128,
                 ):
        super().__init__()
        self.c_dim = c_dim
        self.z_dim = z_dim

        in_channels = c_dim
        hidden_dims = [64, 64, 128, 256, 512, 512]
        # ENCODER
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.enc_z = nn.Sequential(*modules)

        # LATENT
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, z_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1] * 4, z_dim)

        # DECODER
        self.fc_dec_input = nn.Linear(z_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.dec_z = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], c_dim * 3,
                      kernel_size=3, padding=1),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size(), device=mu.device)
        z = mu + std * esp
        return z

    def sample_latents(self, n, device):
        z = torch.normal(0., 1., size=(n, self.z_dim), device=device)
        return z

    def encode(self, c_planes):
        h = 0
        for plane_type in PLANE_TYPES:
            h += self.enc_z(c_planes[plane_type])
        h = h.view(h.size(0), -1)

        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def decode(self, z):
        h = self.fc_dec_input(z)
        h = h.view(h.size(0), -1, 2, 2)
        h = self.dec_z(h)
        h = self.final_layer(h)

        out = {}
        for i, plane_type in enumerate(PLANE_TYPES):
            out[plane_type] = h[:, i * self.c_dim: (i + 1) * self.c_dim]

        return out
