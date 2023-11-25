import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from policy.dataset import pcd_transforms
from policy.utils.io import get_preproc_demo_path, load_gzip_file


def np_to_th(x):
    torch.from_numpy(x).float()


def create_zeros_and_fill(x, shape, dtype, mask):
    sampled_x = np.zeros(shape=shape, dtype=dtype)
    sampled_x[mask] = x
    return sampled_x


def subsample(data_dict, num_points):
    sampled_pts = data_dict["sampled_pts"]
    sampled_occ = data_dict["sampled_occ"]
    occ_mask = sampled_occ != 0

    # subsample pts and occ
    subspl_inds = np.random.randint(sampled_pts.shape[0], size=num_points)
    sampled_pts = sampled_pts[subspl_inds]
    sampled_occ = sampled_occ[subspl_inds]

    return sampled_pts, sampled_occ, subspl_inds, occ_mask


class PlabSceneDataset(Dataset):
    def __init__(
        self,
        cfg,
        env_name,
        split="train",
    ):
        self.cfg = cfg = cfg["data"]
        self.env_name = env_name

        self.split = split
        self.num_points = cfg["num_points"]
        self.temporal_dist = int(cfg["temporal_distance"])
        self.use_partial = cfg.get("use_partial", False)

        if self.use_partial:
            n_obj_pts = cfg.get("n_obj_pts", 5000)
            noise_lvl = cfg.get("noise_lvl", 0.005)
            self.transforms = transforms.Compose(
                [
                    pcd_transforms.SubsamplePointcloud(n_obj_pts),
                    pcd_transforms.PointcloudNoise(noise_lvl),
                ]
            )

        raise NotImplementedError("clean this up")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_file = self.data[idx]
        src_data = np.load(src_file)

        if self.use_partial:
            src_data = dict(src_data)
            partial_file = src_file[:-4] + "_rgbd.npz"
            obj_pcd = np.load(partial_file)["obj_pcd"]
            obj_pcd = self.transforms(obj_pcd)
            src_data["obj_obs"] = obj_pcd

        if self.split == "train":
            ret_dict = self.load_train(src_data)
        else:
            ret_dict = self.load_eval(src_data)

        return ret_dict

    def load_train(self, src_data):
        ret_dict = dict()

        src_sampled_pts, src_sampled_occ, src_subspl_inds, src_occ_mask = subsample(
            src_data, self.num_points
        )
        ret_dict["src_pts"] = np_to_th(src_sampled_pts).float()
        ret_dict["src_occ"] = np_to_th(src_sampled_occ).long()

        src_obs = src_data["obj_obs"]
        ret_dict["src_obs"] = np_to_th(src_obs).float()

        return ret_dict

    def load_eval(self, src_data):
        ret_dict = dict()

        obj_obs = src_data["obj_obs"]
        ret_dict["src_obs"] = np_to_th(obj_obs).float()
        ret_dict["src_pts"] = np_to_th(obj_obs[:, :3]).float()
        ret_dict["src_occ"] = np_to_th(np.where(obj_obs[:, -1] == 0, 1.0, 2.0)).long()

        return ret_dict
