import os

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from .base import PlabSceneDataset
from .skill_dataset import PlabSkillDataset, PlabSkillDatasetDistributed


def collate_pair_fn(batch):
    collated = {}
    for key in batch[0]:
        collated[key] = default_collate([d[key] for d in batch])
    return collated


def worker_init_fn(worker_id):
    """Worker init function to ensure true randomness."""

    def set_num_threads(nt):
        try:
            import mkl

            mkl.set_num_threads(nt)
        except:
            pass
            torch.set_num_threads(1)
            os.environ["IPC_ENABLE"] = "1"
            for o in [
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            ]:
                os.environ[o] = str(nt)

    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)


def get_plab_dataset(cfg, env_name, split="train", distributed=False, **kwargs):
    name = cfg["data"]["dataset"]
    if name == "scene":
        dataset = PlabSceneDataset(cfg, env_name=env_name, split=split)
    elif name == "skill":
        if not distributed:
            dataset = PlabSkillDataset(cfg, env_name=env_name, split=split, **kwargs)
        else:
            dataset = PlabSkillDatasetDistributed(cfg, env_name=env_name, split=split)
    else:
        raise NotImplementedError()

    return dataset


def get_plab_loader(
    cfg, env_name, split="train", distributed=False, distributed_cfg=None, **kwargs
):
    dataset = get_plab_dataset(
        cfg, env_name, split=split, distributed=distributed, **kwargs
    )
    sampler = None
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=distributed_cfg["world_size"],
            rank=distributed_cfg["global_rank"],
        )

    if split == "train":
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg["training"]["batch_size"],
            num_workers=cfg["training"]["n_workers"],
            shuffle=(sampler is None),
            # worker_init_fn=worker_init_fn,
            sampler=sampler,
            drop_last=True,
        )
    else:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg["training"]["batch_size_val"],
            num_workers=cfg["training"]["n_workers_val"],
            shuffle=False,
            sampler=sampler,
            drop_last=True,
        )

    if not distributed:
        return loader
    else:
        return loader, sampler
