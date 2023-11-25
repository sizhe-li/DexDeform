import glob
import multiprocessing as mp
import os

import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset

from policy.utils.io import get_preproc_demo_path, load_gzip_file


def np_to_th(x):
    torch.from_numpy(x).float()


class PlabSkillDataset(Dataset):
    def __init__(
        self,
        cfg,
        env_name,
        split="train",
        action_cache=None,
        frame_cache=None,
    ):
        self.split = split
        self.n_hands = cfg["model"]["n_hands"]
        self.cfg = cfg = cfg["data"]
        self.env_name = env_name
        self.use_partial = cfg.get("use_partial", False)

        if self.use_partial:
            n_obj_pts = cfg.get("n_obj_pts", 5000)
            noise_lvl = cfg.get("noise_lvl", 0.005)
            from torchvision import transforms

            from . import pcd_transforms as pcd_transforms

            self.transforms = transforms.Compose(
                [
                    pcd_transforms.SubsamplePointcloud(n_obj_pts),
                    pcd_transforms.PointcloudNoise(noise_lvl),
                ]
            )

        self.num_points = cfg["num_points"]
        self.N = int(cfg["n_horizon_clips"])
        self.H = int(cfg["horizon_length"])

        raise NotImplementedError("clean this up")

    def __len__(self):
        return len(self.data)

    def load_obs(self, f):
        if self.use_partial:
            partial_file = f[:-4] + "_rgbd.npz"
            obj_obs = np.load(partial_file)["obj_pcd"]
            # randomly subsample
            obj_obs = self.transforms(obj_obs)
        else:
            obj_obs = np.load(f)["obj_obs"]
        return obj_obs

    def __getitem__(self, idx):
        ret_dict = dict()

        frame_files, meta_file = self.data[idx]

        if self.use_cache:
            if meta_file in self.action_cache:
                traj_actions = self.action_cache[meta_file].copy()
            else:
                traj_actions = np.load(meta_file)["actions"]
                self.action_cache[meta_file] = traj_actions.copy()
        else:
            traj_actions = np.load(meta_file)["actions"]

        total_length = len(traj_actions)
        want_length = self.H * self.N

        start = np.random.randint(low=0, high=total_length - want_length)
        frame_files = frame_files[start : start + want_length]
        traj_actions = traj_actions[start : start + want_length]

        # preprocess actions
        n_hands, act_dim = traj_actions.shape[-2:]

        # preprocess states
        traj_obses = []
        for i, f in enumerate(frame_files):
            if self.use_cache:
                if f in self.frame_cache:
                    obj_obs = self.frame_cache[f].copy()
                else:
                    obj_obs = self.load_obs(f)
                    self.frame_cache[f] = obj_obs.copy()
            else:
                obj_obs = self.load_obs(f)

            traj_obses.append(obj_obs)

        traj_obses = np.stack(traj_obses, axis=0)  # T, N_particles, 3

        n_particles, dim = traj_obses.shape[-2:]

        # reshape
        traj_actions = traj_actions.reshape((self.N, self.H, n_hands * act_dim))
        traj_obses = traj_obses.reshape((self.N, self.H, n_particles, dim))

        ret_dict["act_seq"] = np_to_th(traj_actions).float()
        ret_dict["stt_seq"] = np_to_th(traj_obses).float()

        return ret_dict
