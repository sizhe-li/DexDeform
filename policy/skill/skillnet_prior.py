import torch
import torch.nn as nn

from policy.perception.scene_vae.vaenet import SceneVAE
from .models.act_embd import ActionVAE
from .models.pri_embd import PriorVAE
from .models.dyn_pred import DynPredictor


class SkillNetPrior(nn.Module):
    def __init__(
            self,
            act_embd: ActionVAE,
            dyn_pred: DynPredictor,
            stt_vae: SceneVAE,
            pri_embd: PriorVAE,
            device
    ):
        super().__init__()

        self.act_embd: ActionVAE = act_embd.to(device)
        self.dyn_pred: DynPredictor = dyn_pred.to(device)
        self.stt_vae: SceneVAE = stt_vae.to(device)
        self.pri_embd: PriorVAE = pri_embd.to(device)

        self._device = device

    def pred_acts(self, act_seq, stt_seq, **kwargs):
        z, mu_q, logvar_q = self.act_embd.encode(act_seq, stt_seq)
        _, mu_p, logvar_p = self.pri_embd.encode(act_seq, stt_seq)

        B, T = act_seq.shape[:2]
        pred_acts = []
        for t in range(T):
            act = self.act_embd.decode(stt_seq[:, t], z)
            pred_acts.append(act)
        pred_acts = torch.stack(pred_acts, dim=1)

        ret_dict = {
            "z": z,
            "mu_q": mu_q,
            "logvar_q": logvar_q,
            "mu_p": mu_p,
            "logvar_p": logvar_p,
            "pred_acts": pred_acts
        }

        return ret_dict

    def pred_dyns(self, stt, out_act, N):
        pred_stt = stt
        out_dyn = []
        for n in range(N):
            pred_stt = self.dyn_pred(pred_stt, out_act[n]["z"])
            out_dyn.append(pred_stt)
        out_dyn = torch.stack(out_dyn, dim=1)
        return out_dyn

    def encode_state_seq(self, stt_seq, train_vae=False):
        ret_dict = dict()

        # preprocessing observations to state embeddings
        B, N, H, n_particles, stt_dim = stt_seq.shape
        stt_seq = stt_seq.view(B * N * H, n_particles, stt_dim)

        if not train_vae:
            with torch.no_grad():
                # resulting shape: (B * N * H) x state_dim
                ret_dict["stt_vae_z"] = self.stt_vae.encode_scene(stt_seq, use_vae=True)
        else:
            inp_dict = {"obs": stt_seq}
            ret_dict.update(self.stt_vae.forward(inp_dict, decode_scene=False, use_vae=True))
            ret_dict["stt_vae_z"] = ret_dict.pop("z")  # trick for renaming

        ret_dict["stt_vae_z"] = ret_dict["stt_vae_z"].view(B, N, H, -1)

        return ret_dict

    def forward(self, data):
        act_seq = data["act_seq"]
        stt_seq = data["stt_seq"]

        stt_vae_ret_dict = self.encode_state_seq(stt_seq, train_vae=data["train_vae"])
        stt_seq = stt_vae_ret_dict["stt_vae_z"]

        B, N = act_seq.shape[:2]

        out_act = []
        for n in range(N):
            # stopping gradient from state
            out_act.append(self.pred_acts(act_seq[:, n], stt_seq[:, n]))

        out_dyn = self.pred_dyns(stt_seq[:, 0, 0], out_act, N)

        # put output into return dictionary
        ret_dict = dict.fromkeys(out_act[-1].keys())
        for k in ret_dict.keys():
            ret_dict[k] = torch.stack([x[k] for x in out_act], dim=1)
        ret_dict["pred_dyns"] = out_dyn
        ret_dict["stt_vae_ret_dict"] = stt_vae_ret_dict

        return ret_dict

    def sample_latents(self, act_seq, stt_seq, sample_shape):
        # B x H x dim
        _, mu_p, logvar_p = self.pri_embd.encode(act_seq, stt_seq)

        assert mu_p.size(0) == 1, "batch size has to be 1"

        std_p = (logvar_p / 2).exp()
        dist_p = torch.distributions.Normal(mu_p[0], std_p[0])
        return dist_p.sample(sample_shape=sample_shape)


    def decode_for_generation(self, *args, **kwargs):
        return self.stt_vae.decode_for_generation(*args, **kwargs)

    def to(self, device):
        model = super().to(device)
        model._device = device
        return model
