from .config import *

import os
from pathlib import Path

from tools import CN
from tools import merge_inputs

FILEPATH = os.path.dirname(os.path.abspath(__file__))


def get_default_cfg(cfg_path, sim_cfg=None):
    if cfg_path[0] != '/':
        cfg_path = os.path.join(FILEPATH, "../../assets/env_configs", cfg_path)

    assert os.path.exists(cfg_path), "config file does not exist!"

    cfg = CN(new_allowed=True)
    cfg.env_name = Path(cfg_path).stem
    cfg.merge_from_file(cfg_path)

    if sim_cfg is not None:
        cfg.defrost()
        cfg.SIMULATOR = merge_inputs(cfg.SIMULATOR, **sim_cfg)
        cfg.freeze()

    return cfg
