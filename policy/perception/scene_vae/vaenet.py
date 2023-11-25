import torch
import torch.nn as nn
from torch import distributions as dist


class SceneVAE(nn.Module):
    def __init__(
            self,
            obs_encoder,
            vae_btlneck,
            occ_decoder,
            device=None,
    ):
        super().__init__()

        self.obs_encoder = obs_encoder.to(device)
        self.vae_btlneck = vae_btlneck.to(device)
        self.occ_decoder = occ_decoder.to(device)
        self._device = device

    def forward(self, inp_dict, **kwargs):
        ret_dict = dict()

        obs = inp_dict["obs"]
        c = self.encode_scene(obs)

        ret_dict["c"] = c

        if kwargs.get("decode_scene", True):
            pts = inp_dict["pts"]
            logits = self.decode_scene(pts, c)
            ret_dict["logits"] = logits

        if kwargs.get("use_vae", False):
            z, mu, logvar = self.vae_btlneck.encode(c)
            vae_c = self.vae_btlneck.decode(z)

            ret_dict["z"] = z
            ret_dict["mu"] = mu
            ret_dict["logvar"] = logvar
            ret_dict["vae_c"] = vae_c

        return ret_dict

    def encode_scene(self, obs, **kwargs):
        c = self.obs_encoder(obs)

        if kwargs.get("use_vae", False):
            c, _, _ = self.vae_btlneck.encode(c)

        return c

    def decode_scene(self, p, c, **kwargs):

        if kwargs.get("use_vae", False):
            c = self.vae_btlneck.decode(c)

        return self.occ_decoder(p, c)

    def decode_for_generation(self, p, c, **kwargs):
        logits = self.occ_decoder(p, c, **kwargs)
        p_r = torch.softmax(logits, dim=-1)
        p_r = dist.Bernoulli(probs=p_r[..., 1:].sum(-1))

        return p_r

    def to(self, device):
        model = super().to(device)
        model._device = device
        return model
