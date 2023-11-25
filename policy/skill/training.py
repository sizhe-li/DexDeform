from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from policy.perception.common import PLANE_TYPES
from policy.training import BaseTrainer

from .skillnet import SkillNet


def normal_kl(a, b=None):
    """Computes KL divergence based on base normal dist."""
    if b is None:
        mean = torch.zeros_like(a.mean)
        std = torch.ones_like(a.mean)
        b = torch.distributions.Normal(mean, std)

    return torch.distributions.kl.kl_divergence(a, b)


class Trainer(BaseTrainer):
    def __init__(
        self,
        cfg,
        model: SkillNet,
        optimizer,
        device=None,
    ):
        self.epoch_it = 0
        self.step_it = 0

        self.model = model
        self.optimizer = optimizer
        self.device = device

        self.kl_weights = cfg["loss"]["kl_weights"]
        self.state_vae_triplane_weights = cfg["loss"].get(
            "state_vae_triplane_weights", 1.0
        )

        self.pred_hand_label = cfg["model"].get("pred_hand_label", False)
        self.learn_prior = cfg["model"].get("learn_prior", False)
        self.train_vae = cfg["training"].get("train_vae", False)

    def epoch_step(self):
        self.epoch_it += 1

    def train_step(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        loss_dict = self.compute_loss(data)

        loss = 0.0
        _dict = {}
        for k, v in loss_dict.items():
            loss += v
            _dict[k] = v.item()
        loss_dict = _dict

        loss.backward()
        self.optimizer.step()
        self.step_it += 1

        return loss_dict

    def evaluate(self, val_loader):
        self.model.eval()

        eval_list = defaultdict(list)

        for data in tqdm(val_loader, desc="[Validating]"):
            eval_step_dict = self.eval_step(data)

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)

        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}

        return eval_dict

    def eval_step(self, data):
        with torch.no_grad():
            loss_dict = self.compute_loss(data)
        loss_dict = {k: v.item() for k, v in loss_dict.items()}
        return loss_dict

    def compute_loss(self, data):
        device = self.device

        act_seq = targ_acts = data.get("act_seq").to(device)
        stt_seq = data.get("stt_seq").to(device)

        B, N, H, _, _ = stt_seq.shape

        out = self.model(
            {"act_seq": act_seq, "stt_seq": stt_seq, "train_vae": self.train_vae}
        )

        stt_vae_ret_dict = out["stt_vae_ret_dict"]
        stt_seq = stt_vae_ret_dict["stt_vae_z"]

        pred_acts = out["pred_acts"]
        targ_dyns = stt_seq[:, 1:, 0]
        pred_dyns = out["pred_dyns"][:, :-1]
        C = pred_dyns.shape[-1]

        loss_dict = dict()
        if not self.learn_prior:
            mu, logvar = out["mu"], out["logvar"]
            loss_dict["kldiv_loss"] = self.kl_weights * (
                -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            )
        else:
            # take 1 to -1 for overlap
            mu_q, logvar_q = out["mu_q"][:, 1:], out["logvar_q"][:, 1:]
            mu_p, logvar_p = out["mu_p"][:, :-1], out["logvar_p"][:, :-1]

            std_q = (logvar_q / 2).exp()
            std_p = (logvar_p / 2).exp()

            # print(
            #     f"mu_q: {mu_q.mean().item()} | logvar_q: {logvar_q.mean().item()} | "
            #     f"mu_p: {mu_p.mean().item()} | logvar_p: {logvar_p.mean().item()}"
            # )

            dist_q = torch.distributions.Normal(mu_q, std_q)
            dist_p = torch.distributions.Normal(mu_p, std_p)

            loss_dict["kldiv_loss"] = self.kl_weights * normal_kl(dist_q, dist_p).sum()

        loss_dict["act_loss"] = F.mse_loss(pred_acts, targ_acts, reduction="sum") / (
            B * N * H
        )
        loss_dict["dyn_loss"] = F.mse_loss(pred_dyns, targ_dyns, reduction="sum") / (
            B * N * C
        )

        if self.train_vae:
            loss_dict.update(self.stt_vae_loss(stt_vae_ret_dict))

        return loss_dict

    def stt_vae_loss(self, stt_vae_ret_dict):
        loss_dict = dict()

        # trick to fit into memory: we discard the H, and keep the N
        enc_c = torch.cat(
            [stt_vae_ret_dict["c"][plane_type] for plane_type in PLANE_TYPES], dim=1
        )
        dec_c = torch.cat(
            [stt_vae_ret_dict["vae_c"][plane_type] for plane_type in PLANE_TYPES], dim=1
        )
        mu, logvar = stt_vae_ret_dict["mu"], stt_vae_ret_dict["logvar"]

        loss_dict["stt_vae_kldiv_loss"] = self.kl_weights * (
            -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / dec_c.size(0)
        )

        loss_dict["stt_vae_triplane_loss"] = (
            F.mse_loss(dec_c, enc_c.detach(), reduction="sum")
            / (dec_c.size(0) * dec_c.size(1))
            * self.state_vae_triplane_weights
        )

        return loss_dict
