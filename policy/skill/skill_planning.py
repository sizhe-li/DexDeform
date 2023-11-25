import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from mpm import HandEnv
from mpm.viewer import Viewer

from ..perception.utils.coordinate_conversion import (
    IPLCT_COORD_RANGE,
    PLAB_COORD_RANGE,
    find_T_prim_to_keypts,
    transform_points,
    transform_points_th,
)


def batch_input(x, device, dtype=torch.float32):
    if isinstance(x, list):
        if isinstance(x[0], dict):
            x = {k: batch_input([i[k] for i in x], device, dtype) for k in x[0].keys()}
        else:
            if not isinstance(x[0], torch.Tensor):
                x = np.array(x)  # directly batch it?
                x = torch.tensor(x, dtype=dtype).to(device)
            else:
                x = torch.stack(x, 0).to(device)
    elif isinstance(x, dict):
        return {k: batch_input(v, device, dtype) for k, v in x.items()}
    elif isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=dtype).to(device)
    elif isinstance(x, torch.Tensor):
        x = x.to(device)
    else:
        raise NotImplementedError("Can't batch type {}".format(type(x)))
    return x


class SkillPlanner:
    def __init__(
        self,
        model,
        env: HandEnv,
        device=None,
        frame_transforms=None,
        use_partial=False,
    ):
        self.model = model
        self.env = env
        self.env_name = env.cfg.env_name
        self.n_hands = env.simulator.n_hands
        self.use_partial = use_partial

        self.use_prior = False
        if hasattr(self.model, "pri_embd"):
            print("Using prior!!")
            self.use_prior = True
            self.prior_seq_len = 10
            self.act_hist = deque()
            self.stt_hist = deque()

        self.vis = None
        if use_partial:
            self.vis = Viewer(self.env)
            self.vis.refresh_views("obj_centric")

        self.frame_transforms = (
            find_T_prim_to_keypts(
                env, env.simulator.get_state(0), hot_start=None, num_pts=5000
            )
            if frame_transforms is None
            else frame_transforms
        )

        hand_points = []
        hand_labels = []
        hand_points_prim_ids = []

        for idx, (k, v) in enumerate(self.frame_transforms.items()):
            assert len(v) == 2
            hand_points.append(batch_input(v[0][:, :3, 3], "cuda:0"))
            hand_labels.append(torch.tensor(v[1], device="cuda:0", dtype=torch.long))
            hand_points_prim_ids += [idx] * v[0].shape[0]

        self.hand_points = torch.concat(hand_points, 0)
        self.hand_labels = torch.concat(hand_labels, 0)
        self.hand_points_prim_ids = torch.tensor(
            np.array(hand_points_prim_ids), device="cuda:0", dtype=torch.long
        )

        self.plab_coord_range = torch.from_numpy(PLAB_COORD_RANGE[self.env_name]).to(
            device
        )
        self.iplct_coord_range = torch.from_numpy(IPLCT_COORD_RANGE).to(device)

        self.device = device if device is not None else model.device
        self.np2th = lambda x: torch.from_numpy(x).to(self.device).float().unsqueeze(0)

    def state_to_hand_particles(self):
        tool_state = self.env.simulator.get_tool_state(0, device="cuda:0")
        tool_state = torch.stack(tool_state)

        from pytorch3d.transforms.rotation_conversions import quaternion_apply

        prim_pose = tool_state[self.hand_points_prim_ids]
        keypoints = (
            quaternion_apply(prim_pose[:, 3:], self.hand_points) + prim_pose[:, :3]
        )

        return keypoints

    def state_to_scene_particles(
        self, shape_particle_mask=None, fill_colors=False, use_partial=False
    ):
        if not use_partial:
            shape_particles = self.env.simulator.get_x(0, device="cuda:0")
            if shape_particle_mask is not None:
                shape_particles = shape_particles[shape_particle_mask]

            hand_particles = self.state_to_hand_particles()

            scene_pcd = torch.concat([shape_particles, hand_particles], dim=0)
            if not fill_colors:
                scene_lbs = torch.concat(
                    [
                        shape_particles.new_zeros((shape_particles.size(0), 1)),
                        hand_particles.new_ones((hand_particles.size(0), 1)),
                    ],
                    dim=0,
                )
            else:
                shape_cols = shape_particles.new_zeros((shape_particles.size(0), 3))
                hand_cols = hand_particles.new_ones((hand_particles.size(0), 3)) * 0.8
                shape_cols[:, 0] = 0.9266
                shape_cols[:, 1] = 0.7453
                shape_cols[:, 2] = 0.9565
                scene_lbs = torch.concat([shape_cols, hand_cols], dim=0)

            scene_pcd = torch.concat([scene_pcd, scene_lbs], dim=1)

        else:
            points, colors = self.vis.multiview_rgbd_to_pcd(device="cuda:0", spp=1)

            scene_pcd = torch.concat([points, colors], dim=-1)

        # TODO: remove hard coding for accomodating coordinate transform
        if self.env_name == "wrap":
            scene_pcd[..., 0] += 0.2

        scene_pcd[:, :3] = transform_points_th(
            scene_pcd[:, :3], self.plab_coord_range, self.iplct_coord_range
        )
        return scene_pcd

    def state2hidden(self, use_vae=True, use_partial=False, fill_colors=False):
        scene_obs = self.state_to_scene_particles(
            use_partial=use_partial, fill_colors=fill_colors
        )
        scene_obs = scene_obs.unsqueeze(0)
        h = self.model.stt_vae.encode_scene(scene_obs, use_vae=use_vae)

        return h

    def optimize_action_emb(
        self,
        h_init,
        targ_shape,
        num_inits=10,
        num_clips=5,
        n_iters=1000,
        lr=1e-1,
        use_tqdm=False,
    ):
        """
        @param h_init: 1 x hidden_dim
        @param targ_shape: 1 x num_particles x 3
        @param num_inits: how many trajectories to optimize concurrently
        @param num_clips: how many coarse time steps to take
        """
        assert len(h_init.shape) == 2 and len(targ_shape.shape) == 3

        if (
            not self.use_prior
            or (self.act_hist is None and self.stt_hist is None)
            or (len(self.act_hist) < self.prior_seq_len)
        ):
            z = self.model.act_embd.sample_latents(
                n=num_inits * num_clips, device=self.device
            ).view(num_inits, num_clips, -1)
        else:
            print("sampling from prior!")
            act_hist = torch.stack(list(self.act_hist), dim=1)
            stt_hist = torch.stack(list(self.stt_hist), dim=1)
            z = self.model.sample_latents(
                act_hist, stt_hist, sample_shape=(num_inits, num_clips)
            )
            print("z_dim", z.shape)

        z = nn.Parameter(z, requires_grad=True)

        occ_lbs = torch.ones(
            size=(targ_shape.size(1) * num_inits,), dtype=torch.long, device=self.device
        )
        targ_shape = targ_shape.expand(num_inits, -1, -1)
        h_init = h_init.expand(num_inits, -1)

        optim = torch.optim.Adam([z], lr=lr)
        ran = tqdm.trange(n_iters) if use_tqdm else range(n_iters)
        for i in ran:
            h = h_init.clone()
            for j in range(num_clips):
                h = self.model.dyn_pred.forward(h, z[:, j])

            pred_occ = self.model.stt_vae.decode_scene(targ_shape, h, use_vae=True)
            loss = F.cross_entropy(pred_occ.view(-1, 3), occ_lbs)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if use_tqdm:
                ran.set_description(f"{i} | loss: {loss.item():.06f}")

        with torch.no_grad():
            h = h_init.clone()
            for j in range(num_clips):
                h = self.model.dyn_pred.forward(h, z[:, j])

            pred_occ = self.model.stt_vae.decode_scene(targ_shape, h, use_vae=True)
            losses = F.cross_entropy(
                pred_occ.view(-1, 3), occ_lbs, reduction="none"
            ).view(num_inits, -1)
            losses = losses.detach().cpu()

        return z.data.detach(), losses

    def plan_action_emb(
        self,
        init_state,
        targ_shape,
        criterion,
        num_inits=10,
        num_clips=5,
        horizon=15,
        n_iters=1000,
        lr=1e-1,
        use_tqdm=False,
        fast_choice=False,
    ):
        self.model.eval()

        with torch.no_grad():
            self.env.simulator.set_state(0, init_state)
            h_init = self.state2hidden(use_partial=self.use_partial)

        # TODO: remove hard coding for accomodating coordinate transform
        if self.env_name == "wrap":
            targ_shape = targ_shape.copy()
            targ_shape[..., 0] += 0.2

        # convert to implicit coord range
        targ_shape_tf = transform_points(
            targ_shape, PLAB_COORD_RANGE[self.env_name], IPLCT_COORD_RANGE
        )
        targ_shape_tf = self.np2th(targ_shape_tf)
        # cast to cuda
        targ_shape = torch.from_numpy(targ_shape).to(self.device).float()

        zs, losses = self.optimize_action_emb(
            h_init=h_init,
            targ_shape=targ_shape_tf,
            num_inits=num_inits,
            num_clips=num_clips,
            n_iters=n_iters,
            lr=lr,
            use_tqdm=use_tqdm,
        )

        if fast_choice:
            sorted_inds = losses.sum(-1).argsort()
            sorted_inds = sorted_inds[: math.ceil(num_inits * 0.2)]
        else:
            sorted_inds = list(range(num_inits))

        ran = tqdm.trange(len(sorted_inds)) if use_tqdm else range(len(sorted_inds))

        losses = []
        ret_zs = []
        best_loss = 1e10
        for i in ran:
            self.env.simulator.set_state(0, init_state)

            curr_init_idx = sorted_inds[i]
            curr_z = zs[curr_init_idx]

            self.unroll_action_emb(curr_z, horizon=horizon)
            ret_zs.append(curr_z)

            curr_shape = self.env.simulator.get_state(0)[0]
            with torch.no_grad():
                curr_loss = criterion(targ_shape.new_tensor(curr_shape), targ_shape)

            curr_loss = curr_loss.item()
            if curr_loss < best_loss:
                best_loss = curr_loss

            losses.append(curr_loss)
            ran.set_description(f"loss: {curr_loss} (best_loss: {best_loss})")

        return {
            "sorted_inds": torch.tensor(losses).argsort(),
            "action_emb": torch.stack(ret_zs, dim=0),
            "best_loss": best_loss,
        }

    def unroll_action_emb(
        self,
        z,
        horizon=15,
        render_func=None,
        ret_actions=False,
        ret_states=False,
        first_only=False,
        record_hist=False,
    ):
        images = [] if render_func is not None else None
        actions = [] if ret_actions else None
        states = [self.env.simulator.get_state(0)] if ret_states else None

        for j in range(z.size(0)):
            _z = z[j : j + 1]

            # unroll actions
            for t in range(horizon):
                with torch.no_grad():
                    stt = self.state2hidden(
                        use_partial=self.use_partial
                    )  # 1 x state_dim
                    act = self.model.act_embd.decode(stt, _z)
                    act = act.view(self.n_hands, 26)

                    if record_hist and self.use_prior:
                        if len(self.act_hist) == self.prior_seq_len:
                            self.act_hist.pop()
                            self.stt_hist.pop()

                        self.act_hist.appendleft(act.view(1, -1))
                        self.stt_hist.appendleft(stt.view(1, -1))

                self.env.simulator.step(act)
                if render_func is not None:
                    images.append(render_func())
                if ret_actions:
                    actions.append(act)
                if ret_states:
                    states.append(self.env.simulator.get_state(0))

            if first_only:
                break

        return images, actions, states
