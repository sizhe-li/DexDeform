import os
from pathlib import Path

from tools import CN
from tools.config import merge_inputs
from .mujoco_parser import ASSETS_DIR

SUPPORTED_ENVS = ["folding", "rope", "bun", "dumpling", "wrap", "flip", "lift_box"]

def get_default_cfg(cfg_path, sim_cfg=None):
    assert os.path.exists(cfg_path), "config file does not exist!"

    cfg = CN(new_allowed=True)
    cfg.env_name = Path(cfg_path).stem
    cfg.merge_from_file(cfg_path)

    if sim_cfg is not None:
        cfg.defrost()
        cfg.SIMULATOR = merge_inputs(cfg.SIMULATOR, **sim_cfg)
        cfg.freeze()

    return cfg


def make(env_name, sim_cfg=None):
    from mpm.hand import HandEnv
    if env_name not in SUPPORTED_ENVS:
        raise ValueError(f'input environment name *{env_name}* is not supported!')

    cfg_file = os.path.join(ASSETS_DIR, f'env_cfgs/{env_name}.yml')
    cfg = get_default_cfg(cfg_file, sim_cfg=sim_cfg)

    env = HandEnv(cfg)
    return env
