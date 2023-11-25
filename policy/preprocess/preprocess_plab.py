"""
Adapted from https://github.com/NVlabs/ACID/
"""

import gc
import os
import shutil

from tqdm import tqdm

from mpm import HandEnv
from mpm.viewer import Viewer
from policy.perception.utils.coordinate_conversion import *
from policy.utils.io import (
    ENV_NAMES,
    get_demo_path,
    get_preproc_demo_path,
    load_gzip_file,
)
from tools import get_default_cfg

NUM_SHAPE_SAMPLED_PTS = int(1e5)
NUM_HAND_SAMPLED_PTS = int(5e4)

SHAPE_OCC_LABEL = 1
HAND_OCC_LABEL = 2

SHAPE_SCENE_LABEL = 0.0
HAND_SCENE_LABEL = 1.0

NUM_SHAPE_PARTICLES = 10000
NUM_HAND_PARTICLES = 5000

ENV_TO_STD = {
    "bun": 0.08,
    "rope": 0.08,
    "dumpling": 0.08,
    "folding": 0.08,
    "wrap": 0.08,
    "flip": 0.08,
}


def create_folder(_dir, remove_exists=False):
    if os.path.exists(_dir) and remove_exists:
        print(f"Removing existing directory {_dir}")
        shutil.rmtree(_dir, ignore_errors=True)
    os.makedirs(_dir, exist_ok=True)


class Preprocessor:
    def __init__(self, remove_exists=False, debug=False):
        self.env = None
        self.env_name = None
        self.demo_paths = None

        self.remove_exists = remove_exists
        self.debug = debug

    def init_env(self, env_name):
        cfg = get_default_cfg(env_name + ".yml", sim_cfg={"max_steps": 100})
        print(cfg)

        del self.env
        gc.collect()

        self.env = HandEnv(cfg)
        self.env_name = env_name

    def set_demo_paths(self, paths=None):
        if paths is None:
            self.demo_paths = os.listdir(get_demo_path(self.env_name))
        else:
            self.demo_paths = [os.path.basename(x) for x in paths]

    def preprocess(self):
        assert all(
            elem is not None for elem in (self.env, self.env_name, self.demo_paths)
        )

        for i, demo_path in enumerate(self.demo_paths):
            demo_path = get_demo_path(os.path.join(self.env_name, demo_path))
            assert os.path.exists(demo_path), "demo path must exist!"

            demo_data = load_gzip_file(demo_path)

            demo_id = demo_path.split("/")[-1][:-4]
            dump_dir = get_preproc_demo_path(os.path.join(self.env_name, demo_id))

            if not self.remove_exists and os.path.exists(dump_dir):
                print("Skipping due to existing dir!")
                continue

            create_folder(dump_dir, self.remove_exists)

            desc_str = f"[{self.env_name} | {i + 1} / {len(self.demo_paths)}] PROCESSING STATES:"

            if self.debug:
                states_to_check = [demo_data["states"][0], demo_data["states"][-1]]
            else:
                states_to_check = demo_data["states"]

            shape_particle_mask = get_shape_particle_mask(
                states_to_check[0], NUM_SHAPE_PARTICLES
            )

            frame_transforms = find_T_prim_to_keypts(
                self.env, states_to_check[0], hot_start=None, num_pts=NUM_HAND_PARTICLES
            )
            scenes = []
            for state in tqdm(states_to_check, desc=desc_str):
                particles = state_to_scene_particles(
                    self.env,
                    state,
                    frame_transforms,
                    shape_particle_mask=shape_particle_mask,
                )
                scenes.append(particles)

            active_hand_labels = []
            for act in demo_data["actions"]:
                label = (act == 0).sum(-1).argmin().item()
                active_hand_labels.append(label)
            active_hand_labels = np.array(active_hand_labels, dtype=bool)
            hand_switch_inds = np.where(
                active_hand_labels[:-1] != active_hand_labels[1:]
            )[0]

            for frame_id, scene in enumerate(scenes):
                # NOTE THAT scene particles are 4D not 3D!

                scene_pts = transform_points(
                    scene[:, :3], PLAB_COORD_RANGE[self.env_name], IPLCT_COORD_RANGE
                )
                scene_labels = scene[:, -1]

                # enumerate over each type of particles {lh, rh, shape}
                scene_sampled_pts = []
                scene_sampled_occ = []
                scene_inds = []  # map to occupied particles
                for this_label in set(scene_labels):
                    # hyper params
                    scene_obj_mask = scene_labels == this_label
                    is_shape = this_label == 2.0
                    num_pts_to_sample = int(
                        NUM_SHAPE_SAMPLED_PTS if is_shape else NUM_HAND_SAMPLED_PTS
                    )

                    obj_pts = scene_pts[scene_obj_mask]
                    sampled_pts, sampled_occ, sampled_p_nn_inds = sample_occupancies(
                        scene_pts,
                        obj_pts,
                        num_pts_to_sample,
                        std=ENV_TO_STD[self.env_name],
                    )

                    ## post-process

                    # change occ label from [0, 1] to [0, 1, 2]
                    # occupied points' labels should be determined based on scene labels at those points
                    sampled_occ = sampled_occ.astype(np.uint8)
                    sampled_occ[sampled_occ != 0] = np.where(
                        scene_labels[sampled_p_nn_inds] == 2,
                        SHAPE_OCC_LABEL,
                        HAND_OCC_LABEL,
                    )

                    # appending
                    scene_sampled_pts.append(sampled_pts)
                    scene_sampled_occ.append(sampled_occ)
                    scene_inds.append(sampled_p_nn_inds)

                    print(
                        "perc occupied:",
                        (sampled_occ != 0).sum() / sampled_occ.shape[0],
                    )

                # post process scene labels, if it is shape then mark 0, otherwise mark 1
                scene_pts_labels = np.where(
                    scene_labels == 2, SHAPE_SCENE_LABEL, HAND_SCENE_LABEL
                ).astype(DUMP_DATA_TYPE)
                scene_pts_labels = np.expand_dims(scene_pts_labels, axis=-1)

                # scene pts + scene pts labels = scene obj obs
                scene_obj_obs = np.append(scene_pts, scene_pts_labels, axis=-1)
                scene_sampled_pts = np.concatenate(scene_sampled_pts)
                scene_sampled_occ = np.concatenate(scene_sampled_occ)
                scene_inds = np.concatenate(scene_inds)

                # dump frame-level data
                data_dict = {
                    "obj_obs": scene_obj_obs.astype(DUMP_DATA_TYPE),
                    "sampled_pts": scene_sampled_pts.astype(DUMP_DATA_TYPE),
                    "sampled_occ": scene_sampled_occ.astype(np.uint8),
                    "inds": scene_inds.astype(np.uint16),
                }

                save_path = os.path.join(dump_dir, f"{frame_id:06d}.npz")
                np.savez_compressed(save_path, **data_dict)

            # dump meta-level data
            data_dict = {
                "hand_switch_inds": hand_switch_inds.astype(np.uint16),
                "actions": torch.stack(demo_data["actions"])
                .cpu()
                .numpy()
                .astype(np.float32),
            }
            save_path = os.path.join(dump_dir, "meta_labels.npz")
            np.savez_compressed(save_path, **data_dict)


def preprocess_plab_demos_to_scenes(remove_exists, debug=False, env_name=None):
    envs_to_preprocess = ENV_NAMES if env_name is None else [env_name]
    print(f"envs to preprocess: {envs_to_preprocess}")

    proc = Preprocessor(remove_exists, debug)

    for env_name in envs_to_preprocess:
        proc.init_env(env_name)
        proc.set_demo_paths()
        proc.preprocess()
