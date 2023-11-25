from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from policy.perception.common import PLANE_TYPES
from policy.perception.scene_vae.vaenet import SceneVAE
from policy.training import BaseTrainer


class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
    '''

    def __init__(
            self,
            model: SceneVAE,
            optimizer,
            device=None,
            threshold=0.5,
            train_occ=False,
            train_vae=False,
            kl_weights=0.0001,

    ):

        self.epoch_it = 0
        self.step_it = 0

        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.threshold = threshold

        self.train_occ = train_occ
        self.train_vae = train_vae

        self.kl_weights = kl_weights

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

        eval_list = defaultdict(list)

        for data in tqdm(val_loader, desc="[Validating]"):
            eval_step_dict = self.eval_step(data)

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)

        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}

        return eval_dict

    def eval_step(self, data):
        self.model.eval()
        device = self.device

        pts = data.get('src_pts').to(device)
        occ = data.get('src_occ').to(device)
        obs = data.get('src_obs').to(device)
        B = pts.size(0)

        loss_dict = dict()
        with torch.no_grad():
            c = self.model.encode_scene(obs, use_vae=self.train_vae)
            logits = self.model.decode_scene(pts, c, use_vae=self.train_vae)
            loss_dict["occ_loss"] = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                occ.view(-1),
                reduction="mean"
            ).item()

        return loss_dict

    def compute_loss(self, data):
        loss_dict = {}
        loss_dict.update(self.compute_geom_loss(data))

        return loss_dict

    def compute_geom_loss(self, data):
        loss_dict = dict()
        device = self.device

        pts = data.get('src_pts').to(device)
        occ = data.get('src_occ').to(device)
        obs = data.get('src_obs').to(device)
        B = pts.size(0)

        inp_dict = {
            "pts": pts,
            "obs": obs
        }

        ret_dict = self.model.forward(inp_dict, use_vae=self.train_vae)

        if self.train_occ:
            logits = ret_dict["logits"]
            loss_dict["occ_loss"] = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                occ.view(-1),
                reduction="mean"
            )

        if self.train_vae:
            enc_c = torch.cat([ret_dict["c"][plane_type] for plane_type in PLANE_TYPES], dim=1)
            dec_c = torch.cat([ret_dict["vae_c"][plane_type] for plane_type in PLANE_TYPES], dim=1)
            mu, logvar = ret_dict["mu"], ret_dict["logvar"]

            loss_dict["kldiv_loss"] = self.kl_weights * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B)
            loss_dict["triplane_loss"] = F.mse_loss(
                dec_c, enc_c.detach(), reduction="sum"
            ) / (B * dec_c.shape[1])

        return loss_dict
