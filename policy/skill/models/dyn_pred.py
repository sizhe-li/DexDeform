import torch
import torch.nn as nn


class DynPredictor(nn.Module):
    def __init__(
            self,
            state_dim,
            latent_dim,
    ):
        super().__init__()
        self.stt_pred = nn.Sequential(
            nn.Linear(state_dim + latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, state_dim)
        )

    def forward(self, stt, z):
        return self.stt_pred(torch.cat([stt, z], dim=-1))
