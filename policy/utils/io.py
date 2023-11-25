import gzip
import io
import os
import pickle
import shutil
import uuid

import mpm
import torch


ENV_NAMES = [
    "folding",
    "bun",
    "rope",
    "dumpling",
    "wrap",
    "flip",
]

ENV_DIR = "/".join(mpm.__file__.split("/")[:-3])
DATA_DIR = os.path.join(ENV_DIR, "data")

# TODO: clean up this mess and correct the paths
get_env_path = lambda x: os.path.join(ENV_DIR, x)
get_tmp_path = lambda x: os.path.join(DATA_DIR, "tmp", x)
get_demo_path = lambda x: os.path.join(DATA_DIR, "demos", x)
get_preproc_demo_path = lambda x: os.path.join(DATA_DIR, "preproc_demos", x)
get_script_path = lambda x: os.path.join(ENV_DIR, "proj_hand/scripts", x)
get_asset_path = lambda x: os.path.join(ENV_DIR, "proj_hand/assets", x)
raise NotImplementedError("TODO: clean up this mess and correct the paths")


class CPU_Unpickler(pickle.Unpickler):
    # load everything onto cpu
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def load_gzip_file(file_name):
    with gzip.open(file_name, "rb") as f:
        traj = CPU_Unpickler(f).load()
    return traj


def save_gzip_file(data, file_name):
    assert file_name[-3:] == "pkl"
    with gzip.open(file_name, "wb") as f:
        pickle.dump(data, f, protocol=4)


def create_folder(_dir, remove_exists=False):
    if os.path.exists(_dir) and remove_exists:
        print(f"Removing existing directory {_dir}")
        shutil.rmtree(_dir, ignore_errors=True)
    os.makedirs(_dir, exist_ok=True)
