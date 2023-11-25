import torch
import torch.nn as nn


class ActionVAE(nn.Module):
    def __init__(
            self,
            seq_len,
            state_dim,
            action_dim,
            latent_dim,
            use_lstm=False,
            n_hands=1,
            pred_hand_label=False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_lstm = use_lstm
        self.hand_dim = 1 if (n_hands == 2 and pred_hand_label) else 0  # decode which hand is being used

        action_dim += self.hand_dim

        if self.use_lstm:
            in_size = action_dim + state_dim
            self.num_layers = 5
            self.hidden_dim = 128

            self.cvae_encoder_lstm = nn.LSTM(
                input_size=in_size,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True
            )

            self.cvae_encoder_fc = nn.Sequential(
                nn.Linear(self.hidden_dim * seq_len, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 256),
            )

        else:
            self.act_encoder = nn.Sequential(
                nn.Linear(action_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 64)
            )

            self.stt_encoder = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 64)
            )

            self.cvae_encoder = nn.Sequential(
                nn.Linear((64 * 2) * seq_len, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 256),
            )

        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.cvae_decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size(), device=mu.device)
        z = mu + std * esp
        return z

    def sample_latents(self, n, device):
        z = torch.normal(0., 1., size=(n, self.latent_dim), device=device)
        return z

    def encode(self, act_seq, stt_seq):
        # act_seq, stt_seq: B, T, D

        if self.use_lstm:
            inp = torch.cat([act_seq, stt_seq], dim=-1)
            B, T = inp.shape[:2]

            h0 = torch.randn(size=(self.num_layers, B, self.hidden_dim), device=inp.device, dtype=inp.dtype)
            c0 = torch.randn(size=(self.num_layers, B, self.hidden_dim), device=inp.device, dtype=inp.dtype)

            h, _ = self.cvae_encoder_lstm(inp, (h0, c0))
            h = h.reshape(B, -1)
            h = self.cvae_encoder_fc(h)

        else:
            B, T = act_seq.shape[:2]
            act_emb = self.act_encoder(act_seq.reshape(B * T, -1))
            stt_emb = self.stt_encoder(stt_seq.reshape(B * T, -1))

            act_emb = act_emb.reshape(B, -1)
            stt_emb = stt_emb.reshape(B, -1)

            h = torch.cat([act_emb, stt_emb], dim=-1)
            h = self.cvae_encoder(h)

        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def decode(self, stt, z):
        return self.cvae_decoder(torch.cat([stt, z], dim=-1))
