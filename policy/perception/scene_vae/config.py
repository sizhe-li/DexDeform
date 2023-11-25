from policy.perception.decoder import decoder_dict
from policy.perception.encoder import encoder_dict

from . import training, generation
from .vae_btlneck import VaeBtlneck
from .vaenet import SceneVAE


def freeze_network(network):
    for param in network.parameters():
        param.requires_grad = False


def get_model(cfg, device=None):
    train_occ = cfg["training"].get("train_occ", False)
    train_vae = cfg["training"].get("train_vae", False)

    cfg = cfg["model"]

    obs_encoder = cfg["obs_encoder"]
    occ_decoder = cfg["occ_decoder"]

    c_dim = cfg["c_dim"]
    padding = cfg["padding"]
    plane_resolution = cfg["plane_resolution"]
    obs_encoder_kwargs = cfg["obs_encoder_kwargs"]
    occ_decoder_kwargs = cfg["occ_decoder_kwargs"]
    vae_btlneck_kwargs = cfg["vae_btlneck_kwargs"]

    obs_encoder = encoder_dict[obs_encoder](
        c_dim=c_dim, padding=padding,
        plane_resolution=plane_resolution,
        **obs_encoder_kwargs
    )

    vae_btlneck = VaeBtlneck(
        c_dim=c_dim,
        plane_resolution=plane_resolution,
        **vae_btlneck_kwargs
    )

    occ_decoder = decoder_dict[occ_decoder](
        c_dim=c_dim, padding=padding,
        **occ_decoder_kwargs
    )

    if not train_occ:
        print("freezing occupancy network!")
        freeze_network(obs_encoder)
        freeze_network(occ_decoder)

    if not train_vae:
        print("freezing vae network!")
        freeze_network(vae_btlneck)

    model = SceneVAE(
        obs_encoder,
        vae_btlneck,
        occ_decoder,
        device=device
    )

    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    ''' Returns the trainer object.
    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    trainer = training.Trainer(
        model, optimizer, device=device,
        train_occ=cfg["training"]["train_occ"],
        train_vae=cfg["training"]["train_vae"],
        kl_weights=cfg["loss"].get("kl_weights", 0.0001),
    )

    return trainer


def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.
    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    generator = generation.Generator3D(
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
