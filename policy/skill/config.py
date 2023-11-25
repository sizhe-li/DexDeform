import os.path

from .generation import Generator3D
from .models.act_embd import ActionVAE
from .models.pri_embd import PriorVAE
from .models.dyn_pred import DynPredictor
from .skillnet import SkillNet
from .skillnet_prior import SkillNetPrior
from .training import Trainer

from policy.perception.scene_vae.config import get_model as get_state_vae_model
from policy.config import load_config
from policy.utils.io import get_script_path

def freeze_network(network):
    for param in network.parameters():
        param.requires_grad = False


def load_state_vae(ckpt_file, device=None, train_vae=False):
    import torch

    if not os.path.isabs(ckpt_file) and not os.path.exists(ckpt_file):
        ckpt_file = get_script_path(ckpt_file)

    if not os.path.exists(ckpt_file):
        raise FileExistsError("state vae ckpt does not exist! \n"
                              f"=> ckpt_file: {ckpt_file}")

    cfg_file = "/".join(ckpt_file.split("/")[:-1] + ["config.yaml"])

    cfg = load_config(cfg_file)
    state_dim = cfg["model"]["vae_btlneck_kwargs"]["z_dim"]

    cfg["training"]["train_occ"] = False
    cfg["training"]["train_vae"] = True if train_vae else False

    stt_vae = get_state_vae_model(cfg, device=device)

    state_dict = torch.load(ckpt_file)
    stt_vae.load_state_dict(state_dict["model"])

    print(ckpt_file)
    print("=> Loading state vae using local file...")

    if not train_vae:
        print("freezing state vae!")
        freeze_network(stt_vae)

    return stt_vae, state_dim


def get_model(cfg, device=None):
    seq_len = cfg["data"]["horizon_length"]
    cfg_model = cfg["model"]
    learn_prior = cfg_model["learn_prior"]

    n_hands = cfg_model["n_hands"]
    action_dim = n_hands * cfg_model["action_dim"]
    latent_dim = cfg_model["latent_dim"]
    use_lstm = cfg_model.get("use_lstm", False)
    train_vae = cfg["training"].get("train_vae", False)

    pred_hand_label = cfg_model.get("pred_hand_label", False)

    state_vae_ckpt_file = cfg_model["state_encoder_kwargs"].get("ckpt_file", "")

    stt_vae, state_dim = load_state_vae(state_vae_ckpt_file, device, train_vae)
    act_embd = ActionVAE(seq_len, state_dim, action_dim, latent_dim, use_lstm, n_hands, pred_hand_label)
    dyn_pred = DynPredictor(state_dim, latent_dim)

    if not learn_prior:
        model = SkillNet(act_embd, dyn_pred, stt_vae, device=device)
    else:
        pri_embd = PriorVAE(seq_len, state_dim, action_dim, latent_dim, use_lstm, n_hands, pred_hand_label)
        model = SkillNetPrior(act_embd, dyn_pred, stt_vae, pri_embd, device=device)

    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    trainer = Trainer(cfg, model, optimizer, device=device)

    return trainer


def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.
    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    generator = Generator3D(
        model,
        device=device,
        threshold=cfg['test']['threshold'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        padding=cfg['data']['padding'],
        vol_info=None,
        vol_bound=None,
    )
    return generator
