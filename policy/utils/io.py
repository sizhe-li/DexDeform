import gzip
import io
import os
import pickle
import shutil
import uuid

import torch

# load everything onto cpu
class CPU_Unpickler(pickle.Unpickler):
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

