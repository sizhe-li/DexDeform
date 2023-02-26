import numpy as np
import torch
import transforms3d
from tqdm import tqdm

from mpm.cuda_env import CudaEnv
from mpm.mujoco_parser import JOINTS, JOINT_LIMITS, \
    DEFAULT_INITIAL_QPOS, ACTUATOR_JOINT_MAPPING, \
    homogeneous_matrix_from_pos_mat_np, get_actuator_mapping
from mpm.shapes import Shapes
from mpm.simulator import MPMSimulator
from tools import Configurable

LH_PRIM_RANGE = list(range(0, 19))
RH_PRIM_RANGE = list(range(19, 38))
SINGLE_PRIM_RANGE = list(range(0, 19))
DUAL_PRIM_RANGE = list(range(0, 38))


def rigid_body_motion_hand(state, actions, T):
    """
    Args:
        state: (n_hands, 4, 4)
        action: (n_hands, 6,)

    Returns:
        state: (T, n_hands, 4, 4)
    """
    import torch
    from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix, matrix_to_quaternion

    ####################################
    ###### Tile state and actions ######
    ####################################

    state = state[None, :].expand(T, -1, -1, -1).clone()
    actions = actions[None, :].expand(T, -1, -1).clone() * (
            torch.arange(T, device=actions.device)[:, None, None] + 1) / T

    #########################
    ###### translation ######
    #########################
    state[..., :3, 3] += actions[..., :3]

    ######################
    ###### rotation ######
    ######################
    q = matrix_to_quaternion(state[..., :3, :3])  # T, n_hands, 4
    rot = actions[..., 3:]  # T, n_hands, 3

    w = torch.sqrt((rot * rot).sum(axis=-1, keepdims=True) + 1e-16)  # T, n_hands
    quat = torch.cat((torch.cos(w / 2), (rot / torch.clamp(w, 1e-7, 1e9)) * torch.sin(w / 2)), -1)  # T, n_hands, 4
    # terms = q.outer_product(quat) # we should use w x q
    terms = q[..., None] * quat[..., None, :]
    w = terms[..., 0, 0] - terms[..., 1, 1] - terms[..., 2, 2] - terms[..., 3, 3]
    x = terms[..., 0, 1] + terms[..., 1, 0] - terms[..., 2, 3] + terms[..., 3, 2]
    y = terms[..., 0, 2] + terms[..., 1, 3] + terms[..., 2, 0] - terms[..., 3, 1]
    z = terms[..., 0, 3] - terms[..., 1, 2] + terms[..., 2, 1] + terms[..., 3, 0]
    out = torch.stack([w, x, y, z], -1)
    # print("out1", out[0])
    out = out / torch.linalg.norm(out, dim=-1, keepdims=True)
    # print("out2", out[0])
    state[..., :3, :3] = quaternion_to_matrix(out)

    return state


class HandSimulator(MPMSimulator):
    def __init__(self,
                 n_bodies,
                 hand_cfg,
                 cfg=None,
                 quality=None,
                 action_scale=None,
                 device='cuda:0',
                 mode=None,
                 scale=None,
                 ctrl_type=None,
                 hand_friction=None,
                 fixed_base=False
                 ):
        dt = 0.5e-4 / quality
        dx = 1. / (64 * quality)
        substeps = int(np.ceil(2e-3 / dt))
        super().__init__(n_bodies, dt=dt, dx=dx, substeps=substeps)

        ##################
        ## hyper params ##
        ##################

        self.device = device
        self.ctrl_type = cfg.ctrl_type
        self.n_hands = hand_cfg["n_hands"]
        self.n_joints_per_hand = len(JOINTS)
        self.n_actuators_per_hand = len(ACTUATOR_JOINT_MAPPING)

        if self.ctrl_type == "vel":
            self.compute_forward_kinematics = self.JointVel_Fk
        else:
            raise ValueError(f"ctrl type {self.ctrl_type} is not supported!")

        self.base_pose = [None for _ in range(self.max_steps)]  # we store matrix directly
        self.joint_rot = [None for _ in range(self.max_steps)]

        self.q_lower = self.togpu([JOINT_LIMITS[i][0] for i in JOINTS])[None, None, :]
        self.q_upper = self.togpu([JOINT_LIMITS[i][1] for i in JOINTS])[None, None, :]

        self.action_map = self.togpu(get_actuator_mapping(), dtype=torch.long)
        if action_scale is None:
            action_scale = [0.33 * 0.002] * self.n_actuators_per_hand
            if not fixed_base:
                action_scale = action_scale + [0.01] * 3 + [0.015] * 3
            else:
                action_scale = action_scale + [0.] * 6

        self.torch_action_scale = self.togpu(np.array(action_scale))

        ########################
        ## frame computations ##
        ########################

        arm_frame = hand_cfg["arm_frame"]
        root_frame = hand_cfg["root_frame"]
        site_computations = hand_cfg["site_computations"]
        joint_info = hand_cfg["joint_info"]
        geometries = hand_cfg["geometries"]

        self.base_pose[0] = self.togpu(root_frame)
        self.body_parent = []
        self.body_mats = []
        self.body2joint = []

        num_sites = 5
        for i in range(num_sites):
            idx = 0
            for idx_type, index in site_computations["inds"][i]:
                self.body_parent.append(idx)
                if idx_type != "joint_idx":
                    self.body_mats.append(self.togpu(site_computations["mats"][i][index]))
                    self.body2joint.append(None)
                else:
                    self.body_mats.append(None)
                    self.body2joint.append(index)
                idx += 1

        self.joint_pos = self.togpu([i[0] for i in joint_info]).transpose(0, 1)
        self.joint_axis = self.togpu([i[1] for i in joint_info]).transpose(0, 1)

        self.geometries = []
        self.geom_index = []
        for geom in geometries["col"]:
            if "forearm" in geom.parent_name:
                parent_frame = arm_frame
                continue
                # raise NotImplementedError
            else:
                assert not geom.is_mesh
                joint_idx = JOINTS.index(geom.parent_name)
                self.geom_index.append(joint_idx)
                mat = geom.matrix
                if geom.data['type'] == 'capsule':
                    mat[:3, :3] = mat[:3, :3] @ transforms3d.axangles.axangle2mat([1, 0, 0], np.pi / 2)
                self.geometries.append(mat)

        assert len(self.geometries) == n_bodies

        # reshape according to n_hands
        n_geoms_per_hand = len(self.geometries) // self.n_hands
        self.geometries = self.togpu(self.geometries).view(self.n_hands, n_geoms_per_hand, 4, 4)
        self.geom_index = self.togpu(self.geom_index[:n_geoms_per_hand], dtype=torch.long)
        self.joint_rot[0] = self.togpu([DEFAULT_INITIAL_QPOS[i] for i in JOINTS])[None, :].expand(self.n_hands, -1)

    def togpu(self, x, dtype=torch.float32):
        return torch.tensor(np.array(x), dtype=dtype, device=self.device)

    def download_pos_rot(self, cur, device):
        # TODO: strange interface to clear the qpos, similar to the reset function
        assert cur == 0
        for i in range(1, self.max_steps):
            self.base_pose[i] = self.joint_rot[i] = None
        return self.base_pose[cur].to(device), self.joint_rot[cur].to(device)

    def get_state(self, index):
        assert index % self.substeps == 0
        return super().get_state(index) + (
            self.base_pose[index].detach().cpu().numpy(),
            self.joint_rot[index].detach().cpu().numpy()
        )

    def set_state(self, index, state):
        assert index % self.substeps == 0

        self.base_pose[index] = torch.tensor(state[-2], device=self.device, dtype=torch.float32)
        self.joint_rot[index] = torch.tensor(state[-1], device=self.device, dtype=torch.float32)

        state = [np.float32(i) for i in state]

        pos, rot = self.hand_forward_kinematics(
            self.base_pose[index][None, :],
            self.joint_rot[index][None, :]
        )
        pos = pos[0].detach().cpu().numpy()
        rot = rot[0].detach().cpu().numpy()
        self.n_particles = int(len(state[0]))

        pos = np.float32(pos)
        rot = np.float32(rot)
        state = state[:4] + [pos, rot]
        self.states[index].set_state(state)
        self.cur = index

    def get_state_render_only(self, f=0, device="numpy"):
        p = self.get_x(f)
        base_pose = self.base_pose[f].cpu().numpy()
        joint_rot = self.joint_rot[f].cpu().numpy()

        return p, base_pose, joint_rot

    def set_state_render_only(self, p, base_pose, joint_rot, f=0):

        base_pose = torch.tensor(base_pose, device=self.device, dtype=torch.float32)
        joint_rot = torch.tensor(joint_rot, device=self.device, dtype=torch.float32)

        pos, rot = self.hand_forward_kinematics(
            base_pose[None, :],
            joint_rot[None, :]
        )

        pos = pos[0].detach().cpu().numpy()
        rot = rot[0].detach().cpu().numpy()

        self.states[f].x.upload(p)
        self.states[f].body_pos.upload(pos)
        self.states[f].body_rot.upload(rot)

    def lh_sdf_given_p(self, p):
        return self.primitive_sdf_given_p(p, LH_PRIM_RANGE)

    def rh_sdf_given_p(self, p):
        return self.primitive_sdf_given_p(p, RH_PRIM_RANGE)

    def primitive_sdf_given_p(self, p, hand_inds):
        init_state = self.get_state(0)
        num_points = len(p)
        assert num_points <= self.n_particles

        start = 0
        end = num_points

        tmp_p = np.ones((self.n_particles, 3))
        tmp_p[start: end] = p[start:end]
        self.states[0].x.upload(tmp_p)

        sdf_vals = self.get_dists(f=0, device="numpy")
        sdf_vals = sdf_vals[start:end, hand_inds].min(-1)

        self.set_state(0, init_state)
        return sdf_vals

    def sample_pts_helper(self, num_points, hand_inds, center=None, hot_start=None):
        points = np.ones((num_points, 3)) * 5

        remain_cnt = num_points
        pbar = tqdm(total=remain_cnt)

        started = False
        while remain_cnt > 0:
            p_samples = (np.random.random((self.n_particles, 3)) - 0.5) + center

            if not started and hot_start is not None:
                tmp_len = min(self.n_particles, len(hot_start))
                p_samples[:tmp_len] = hot_start[:tmp_len]
                started = True

            self.states[0].x.upload(p_samples)

            sdf_vals = self.get_dists(f=0, device="numpy")
            sdf_vals = sdf_vals[:, hand_inds].min(-1)
            accept_map = sdf_vals <= 0
            accept_cnt = sum(accept_map)

            start = num_points - remain_cnt
            points[start:start + accept_cnt] = p_samples[accept_map][:min(accept_cnt, remain_cnt)]
            remain_cnt -= accept_cnt

            pbar.update(accept_cnt)
            pbar.set_description(f"HAND SDF SAMPLING: {max(0, remain_cnt)}", refresh=True)
        pbar.close()

        assert np.all(points != 5)

        return points

    def sample_pts_inside_primitives(self, n_pts, mode=None, hot_start=None):
        init_state = self.get_state(0)

        if mode == "dual":
            n_pts_per_hand = n_pts // 2
            body_pos = self.states[0].body_pos.download()

            lh_center = body_pos[LH_PRIM_RANGE, :].mean(0)
            rh_center = body_pos[RH_PRIM_RANGE, :].mean(0)

            p1 = self.sample_pts_helper(n_pts_per_hand, LH_PRIM_RANGE, lh_center, hot_start)
            p2 = self.sample_pts_helper(n_pts_per_hand, RH_PRIM_RANGE, rh_center, hot_start)

            l1 = np.zeros((len(p1), 1))
            l2 = np.ones((len(p2), 1))

            p = np.concatenate((p1, p2), axis=0)
            labels = np.concatenate((l1, l2), axis=0)

        elif mode == "lh":
            hand_inds = LH_PRIM_RANGE if self.n_hands == 2 else SINGLE_PRIM_RANGE

            body_pos = self.states[0].body_pos.download()
            center = body_pos[hand_inds, :].mean(0)

            p = self.sample_pts_helper(n_pts, hand_inds, center, hot_start)
            labels = np.zeros((len(p), 0))

        elif mode == "rh":
            hand_inds = RH_PRIM_RANGE if self.n_hands == 2 else SINGLE_PRIM_RANGE

            body_pos = self.states[0].body_pos.download()
            center = body_pos[hand_inds, :].mean(0)

            p = self.sample_pts_helper(n_pts, hand_inds, center, hot_start)
            labels = np.zeros((len(p), 1))
        else:
            raise ValueError("Mode is not supported!")

        num_valid = len(p)
        p_samples = np.zeros(shape=(self.n_particles, 3))
        p_samples[:num_valid] = p
        self.states[0].x.upload(p_samples)
        p_sdf_vals = self.get_dists(f=0, device="numpy")[:num_valid]

        self.set_state(0, init_state)

        return p, p_sdf_vals, labels

    def mat2pos_rot(self, mat):
        from pytorch3d.transforms.rotation_conversions import matrix_to_quaternion
        return mat[..., :3, 3], matrix_to_quaternion(mat[..., :3, :3])

    def hand_forward_kinematics(self, base_pose, q):
        """
        Batched forward kinematics

        Args:
            base_pose: the 6D pose of the wrist (substeps, n_hands, 4, 4)
            q: joint rotations (substeps, n_hands, 24)
        Returns:
            T: next geom positions
            R: next geom rotations
        """
        from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix, matrix_to_quaternion

        substeps = base_pose.shape[0]

        # place holders
        T = torch.zeros((substeps, self.n_hands, self.n_joints_per_hand, 4, 4), device=q.device)
        joint_poses = torch.zeros((substeps, self.n_hands, self.n_joints_per_hand, 4, 4), device=q.device)

        T[..., :3, :3] = axis_angle_to_matrix(self.joint_axis[None, :] * q[..., None])
        T[..., :3, 3] = self.joint_pos
        T[..., 3, 3] = 1

        for parent, body_mat, joint_index in zip(self.body_parent, self.body_mats, self.body2joint):
            if parent == 0:
                base = base_pose
            if body_mat is not None:
                base = base @ body_mat
            else:
                base = joint_poses[..., joint_index, :, :] = base @ T[..., joint_index, :, :]
        geom_mat = joint_poses[..., self.geom_index, :, :] @ self.geometries
        geom_mat = geom_mat.view(substeps, -1, 4, 4)
        T = geom_mat[..., :3, 3]
        R = matrix_to_quaternion(geom_mat[..., :3, :3])
        return T, R  # pos, rot of each bodies

    def JointVel_Fk(self, f, actions, pos_rot=None):
        device = self.device

        if pos_rot is None:
            curr_base, curr_joint_pos = self.base_pose[f], self.joint_rot[f]
        else:
            curr_base, curr_joint_pos = pos_rot

        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, device=device, dtype=torch.float32)
        else:
            actions = actions.to(device)

        ###############################################
        ######## rigid body motion of wrist ###########
        ###############################################

        if actions.shape[1] == self.n_actuators_per_hand + 6:
            next_base = rigid_body_motion_hand(curr_base, actions[:, -6:] * self.torch_action_scale[None, -6:],
                                               self.substeps)
        else:
            next_base = curr_base[None, ...].expand(self.substeps, -1, -1, -1)

        ########################################
        ######## actuation of joints ###########
        ########################################

        assert actions.shape[0] == self.n_hands

        actions = (actions[..., :self.n_actuators_per_hand].clamp(-1., 1.) *
                   self.torch_action_scale[None, :self.n_actuators_per_hand])[:, self.action_map]
        next_joint_pos = curr_joint_pos[None, :] + actions[None, :] * (
                torch.arange(self.substeps, device=device)[:, None, None] + 1)
        next_joint_pos = next_joint_pos.clamp(self.q_lower, self.q_upper)

        #####################################################
        ######## forward kinematics of geometries ###########
        #####################################################

        geom_pos, geom_rot = self.hand_forward_kinematics(next_base, next_joint_pos)

        next_base_pose = next_base[-1]
        next_joint_rot = next_joint_pos[-1]
        self.base_pose[f + self.substeps] = next_base_pose
        self.joint_rot[f + self.substeps] = next_joint_rot
        return geom_pos, geom_rot, (next_base_pose, next_joint_rot)

    def step(self, action, q_state=None):
        super().step(action, q_state)
        self.base_pose[0] = self.base_pose[self.substeps]
        self.joint_rot[0] = self.joint_rot[self.substeps]


class HandEnv(CudaEnv):
    def __init__(
            self,
            cfg=None,
            MANIPULATORS=None,
            env_name=None,
    ):
        Configurable.__init__(self)

        self.cfg = cfg
        self.fixed_base = cfg.SIMULATOR.get("fixed_base", False)

        n_particles = cfg.SIMULATOR.n_particles
        if cfg.get('SHAPES', None) is not None:
            shape_maker = Shapes(cfg.SHAPES)
            objects, colors, _, particle_mu_lam_yield = shape_maker.get()
            n_particles = cfg.SIMULATOR.n_particles = max(cfg.SIMULATOR.n_particles, len(objects))

        # TODO: use "colors" returned from shape_maker
        self.particle_colors = np.zeros(n_particles) + 0xdf73ff

        # SIMULATOR
        env_params = self.parse_sim_cfg(cfg.SIMULATOR)
        primitives, hand_cfg = env_params['primitives'], env_params['hand_cfg']
        n_bodies, kwargs = self.parse_tools(primitives)
        self.simulator = HandSimulator(n_bodies, hand_cfg, cfg=cfg.SIMULATOR)
        self.simulator.init_bodies(**kwargs)

        if cfg.get('SHAPES', None) is not None:
            # write particle positions and colors
            self.simulator.states[0].x.upload(objects)

        # RENDERER
        from mpm.cuda_env import Renderer
        self.renderer = Renderer(cfg=cfg.RENDERER)

        # HAND POSE + JOINT_POS
        # TODO: support default hand positions, depending on n_hands
        root_matrix, joint_pos = self.parse_manip_cfgs(cfg.MANIPULATORS)
        self.initialize(root_matrix, joint_pos)
        self.init_state = self.simulator.get_state(0)

    def initialize(self, root_frame=None, joint_pos=None):
        state = self.simulator.get_state(0)
        state[2][:] = np.eye(3)

        if root_frame is not None:
            state[-2][:] = root_frame

        if joint_pos is not None:
            state[-1][:] = joint_pos

        self.simulator.set_color(self.particle_colors)
        self.simulator.set_state(0, state)

    def set_particle_color(self, col):
        self.particle_colors[:] = col
        self.simulator.set_color(self.particle_colors)

    def render_rgb(self, index=0, **kwargs):
        return self.renderer.render(self.simulator,
                                    self.simulator.states[index],
                                    self.simulator.stream0, **kwargs)[:, :, :3]

    def render_rgbd(self, index=0, **kwargs):
        return self.renderer.render(self.simulator,
                                    self.simulator.states[index],
                                    self.simulator.stream0, **kwargs)

    def parse_manip_cfgs(self, cfgs):
        import yaml
        from yacs.config import CfgNode as CN
        from mpm.mujoco_parser import DEFAULT_INITIAL_QPOS

        hand_inds = []
        root_mats = []
        init_qposes = []

        for i in cfgs:
            if isinstance(i, CN):
                cfg = i
            else:
                cfg = CN(new_allowed=True)
                cfg = cfg._load_cfg_from_yaml_str(yaml.safe_dump(i))

            init_rot = eval(cfg.init_rot)
            init_pos = eval(cfg.init_pos)
            if cfg.init_qpos == "default":
                init_qpos = list(DEFAULT_INITIAL_QPOS.values())
            elif cfg.init_qpos == "zero":
                init_qpos = [0.] * 24
            elif isinstance(eval(cfg.init_qpos), tuple):
                init_qpos = list(eval(cfg.init_qpos))
                assert len(init_qpos) == 24
            else:
                raise NotImplementedError()

            hand_inds.append(cfg.hand_idx)
            root_mats.append(self.get_root_matrix(init_pos, init_rot))
            init_qposes.append(np.array(init_qpos))

        if len(hand_inds) == 2 and (hand_inds[1] < hand_inds[0]):
            raise ValueError("hand config is out of order!")

        root_mats = np.stack(root_mats)
        init_qposes = np.stack(init_qposes)

        return root_mats, init_qposes

    def parse_sim_cfg(self, cfg):
        mode, scale, hand_friction = cfg.mode, cfg.scale, cfg.hand_friction

        from mpm.mujoco_parser import prepare
        if mode in ["lh", "rh"]:
            filename = "left_hand" if mode == "lh" else "right_hand"
            n_hands = 1
            root_info, site_computations, joint_info, geometries = prepare(
                f"shadow/{filename}/shadow_hand.xml",
                scale=scale)
            # padding n_hands dimension
            for k in root_info.keys():
                root_info[k] = np.expand_dims(root_info[k], axis=0)

            for i in range(len(site_computations["mats"])):
                for j in range(len(site_computations["mats"][i])):
                    site_computations["mats"][i][j] = np.expand_dims(site_computations["mats"][i][j], axis=0)

            for i, info in enumerate(joint_info):
                joint_info[i] = tuple([np.expand_dims(x, axis=0) for x in info])

        elif mode in ["dual", "lh+rh"]:
            n_hands = 2
            root_info_lh, site_computations_lh, joint_info_lh, geometries_lh = prepare(
                "shadow/left_hand/shadow_hand.xml",
                scale=scale)
            root_info_rh, site_computations_rh, joint_info_rh, geometries_rh = prepare(
                "shadow/right_hand/shadow_hand.xml",
                scale=scale)

            root_info = dict()
            for k in root_info_rh.keys():
                root_info[k] = np.stack([root_info_lh[k], root_info_rh[k]], axis=0)

            site_computations = dict()
            site_computations["inds"] = site_computations_lh["inds"]
            site_computations["mats"] = list()
            for i in range(len(site_computations_lh["mats"])):
                site_computations["mats"].append(list())
                for j in range(len(site_computations_lh["mats"][i])):
                    site_computations["mats"][i].append(np.stack([site_computations_lh["mats"][i][j],
                                                                  site_computations_rh["mats"][i][j]], axis=0))

            joint_info = list()
            for lh_info, rh_info in zip(joint_info_lh, joint_info_rh):
                joint_info.append(tuple([np.stack([x, y], axis=0) for x, y in zip(lh_info, rh_info)]))

            geometries = dict()
            for geom_type in ("vis", "col"):
                geometries[geom_type] = geometries_lh[geom_type] + geometries_rh[geom_type]
        else:
            raise ValueError(f"incorrect hand mode: {mode}")

        primitives = []
        for i in geometries['col']:
            if "type" not in i.data:
                continue

            tool_cfg = self.default_tool_config()
            if i.data['type'] == 'capsule':
                tool_cfg['shape'] = 'Capsule'
            elif i.data["type"] == 'box':
                tool_cfg['shape'] = 'Box'

            tool_cfg['size'] = i.data["size"]
            tool_cfg['round'] = 0
            tool_cfg['friction'] = hand_friction
            primitives.append(tool_cfg)

        arm_frame = root_info["root_matrix"] @ root_info["robot0:hand mount"]
        root_frame = arm_frame @ root_info["robot0:wrist"]
        # robot info
        hand_cfg = {
            "n_hands": n_hands,
            "arm_frame": arm_frame, "root_frame": root_frame,
            "site_computations": site_computations,
            "joint_info": joint_info, "geometries": geometries
        }

        return {"primitives": primitives,
                "hand_cfg": hand_cfg}

    @staticmethod
    def get_root_matrix(pos, rot):
        return homogeneous_matrix_from_pos_mat_np(pos, transforms3d.euler.euler2mat(*rot))

    def set_single_hand_pose(self, hand_idx=0, pos=None, rot=None, joint_pos=None):
        state = self.simulator.get_state(0)
        if pos is not None and rot is not None:
            state[-2][hand_idx] = self.get_root_matrix(pos, rot)
        if joint_pos is not None:
            assert self.simulator.n_joints_per_hand == len(joint_pos)
            state[-1][hand_idx] = joint_pos
        self.simulator.set_state(0, state)

    def set_dual_hand_pose(self, pos=None, rot=None, joint_pos=None):
        state = self.simulator.get_state(0)
        if pos is not None and rot is not None:
            assert self.simulator.n_hands == len(pos) == len(rot) == 2
            state[-2][:] = np.stack([self.get_root_matrix(p, r) for p, r in zip(pos, rot)], axis=0)
        if joint_pos is not None:
            assert self.simulator.n_hands == len(joint_pos) and \
                   self.simulator.n_joints_per_hand == len(joint_pos[0])
            state[-1][:] = np.stack([x for x in joint_pos], axis=0)
        self.simulator.set_state(0, state)
