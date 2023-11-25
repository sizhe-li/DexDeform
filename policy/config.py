import yaml

from policy import skill
from policy.perception import scene_vae

method_dict = {
    "scene_vae": scene_vae,
    "skill": skill,
}


# General config
def load_config(path, default_path=None):
    """Loads config file.

    Args:
        path (str): path to config file
        default_path (bool): whether to use default path
    """
    # Load configuration from file itself
    with open(path, "r") as f:
        cfg_special = yaml.full_load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get("inherit_from")

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, "r") as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    """Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


# Models
def get_model(cfg, device=None):
    """Returns the model instance.

    Args:
        cfg (dict): config dictionary
        device (device): pytorch device
        dataset (dataset): dataset
    """
    method = cfg["method"]
    model = method_dict[method].config.get_model(cfg, device=device)
    return model


# Trainer
def get_trainer(model, optimizer, cfg, device):
    """Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary
        device (device): pytorch device
    """
    method = cfg["method"]
    trainer = method_dict[method].config.get_trainer(model, optimizer, cfg, device)
    return trainer


# Generator for final mesh extraction
def get_generator(model, cfg, device):
    """Returns a generator instance.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        device (device): pytorch device
    """
    method = cfg["method"]
    generator = method_dict[method].config.get_generator(model, cfg, device)
    return generator
