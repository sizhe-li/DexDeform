# https://github.com/haosulab/ManiSkill2022/blob/plb/mpm/simulator.py
import numpy as np
import torch

from mpm.types import (
    ivec3,
    array,
    float32,
    vec3,
    quat,
    mat3,
    lib,
)

grid_lower_default = np.array([[1e6, 1e6, 1e6]], dtype=np.int32)
grid_lower_zero = np.array([[0., 0., 0.]], dtype=np.int32)


def rigid_body_motion(states, actions):
    import torch
    # state: (B, 7)
    # action: (T, B, 6)
    T, B = actions.shape[:2]
    pos, q = states

    pos = pos[None, :].expand(T, -1, -1).reshape(-1, 3)
    q = q[None, :].expand(T, -1, -1).reshape(-1, 4)
    actions = actions.reshape(-1, 6)
    rot = actions[:, 3:]
    # better if we can move this part to C++ code
    w = torch.sqrt((rot * rot).sum(axis=-1, keepdims=True) + 1e-16)
    quat = torch.cat((torch.cos(w / 2), (rot / torch.clamp(w, 1e-7, 1e9)) * torch.sin(w / 2)), 1)

    next_pos = pos + actions[:, :3]
    # terms = q.outer_product(quat) # we should use w x q
    terms = q[:, :, None] * quat[:, None, :]
    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    out = torch.stack([w, x, y, z], 1)
    next_rot = out / torch.linalg.norm(out, dim=-1, keepdims=True)
    return next_pos.reshape(T, B, 3), next_rot.reshape(T, B, 4)


class TempState:
    # store temporary grid information
    def __init__(self, n_particles, grid_dim, n_bodies):
        grid_length = grid_dim[0] * grid_dim[1] * grid_dim[2]
        self.grid_m = array(dtype=float, length=grid_length)
        self.grid_v_in = array(dtype=vec3, length=grid_length)  # v_in
        self.grid_v_out = array(dtype=vec3, length=grid_length)  # v_out

        self.grid_m_grad = array(dtype=float, length=grid_length)
        self.grid_v_in_grad = array(dtype=vec3, length=grid_length)
        self.grid_v_out_grad = array(dtype=vec3, length=grid_length)

        self.grid_body_v_in = array(dtype=vec3, length=grid_length * (n_bodies + 1))

        self.F = array(dtype=mat3, length=n_particles)  # deformation gradient
        self.U = array(dtype=mat3, length=n_particles)  # deformation gradient
        self.V = array(dtype=mat3, length=n_particles)  # deformation gradient
        self.sig = array(dtype=vec3, length=n_particles)  # deformation gradient

        self.F_grad = array(dtype=mat3, length=n_particles)  # deformation gradient
        self.U_grad = array(dtype=mat3, length=n_particles)  # deformation gradient
        self.V_grad = array(dtype=mat3, length=n_particles)  # deformation gradient
        self.sig_grad = array(dtype=vec3, length=n_particles)  # deformation gradient

        self.grid_lower = array(dtype=ivec3, length=1)

    def clear(self, stream=None):
        self.grid_m.zero(stream)
        self.grid_v_in.zero(stream)
        self.grid_v_out.zero(stream)

    def clear_grad(self, stream=None):
        self.grid_v_in_grad.zero(stream)
        self.grid_v_out_grad.zero(stream)
        self.grid_m_grad.zero(stream)

        self.sig_grad.zero(stream)
        self.F_grad.zero(stream)
        self.U_grad.zero(stream)
        self.V_grad.zero(stream)


class State:
    def __init__(self, n_particles, n_bodies):
        self.n_particles = int(n_particles)
        self.n_bodies = n_bodies

        self.x = array(dtype=vec3, length=n_particles)
        self.v = array(dtype=vec3, length=n_particles)
        self.F = array(dtype=mat3, length=n_particles)
        self.C = array(dtype=mat3, length=n_particles)

        self.x_grad = array(dtype=vec3, length=n_particles)
        self.v_grad = array(dtype=vec3, length=n_particles)
        self.F_grad = array(dtype=mat3, length=n_particles)
        self.C_grad = array(dtype=mat3, length=n_particles)

        if n_bodies:
            self.body_pos = array(dtype=vec3, length=n_bodies)
            self.body_rot = array(dtype=quat, length=n_bodies)

            self.body_pos_grad = array(dtype=vec3, length=n_bodies)
            self.body_rot_grad = array(dtype=quat, length=n_bodies)

    def reset(self):
        eye = np.zeros((self.model.n_particles, 3, 3), dtype=np.float32)
        eye[:] = np.eye(3)
        self.x.zero()
        self.v.zero()
        self.C.zero()
        self.F.upload(eye.reshape(self.F.shape))

    def get_state(self, n=None):
        x = self.x.download(n)
        v = self.v.download(n)
        F = self.F.download(n)
        C = self.C.download(n)
        return x, v, F.reshape(-1, 3, 3), C.reshape(-1, 3, 3), \
            self.body_pos.download(), self.body_rot.download()

    def set_state(self, state):
        self.x.upload(state[0])
        self.v.upload(state[1])
        self.F.upload(state[2].reshape(-1, 9))
        self.C.upload(state[3].reshape(-1, 9))

        pos, rot = state[4:]
        self.body_pos.upload(pos)
        self.body_rot.upload(rot)

    def clear_grad(self, stream):
        # TODO: should be asynchronized later after back-propogation..
        self.x_grad.zero_async(stream)
        self.v_grad.zero_async(stream)
        self.F_grad.zero_async(stream)
        self.C_grad.zero_async(stream)

        if self.n_bodies:
            self.body_pos_grad.zero_async(stream)
            self.body_rot_grad.zero_async(stream)


from tools import Configurable


class MPMSimulator(Configurable):
    def __init__(
            self,
            n_bodies,
            cfg=None,
            ground_friction=0.,  # Currently I only support 0 now.
            gravity=(0., -1., 0.),  # 9.8
            n_particles=20000,
            dx=1. / 64,
            dt=0.0001,
            grid_size=(1., 1., 1.),
            max_steps=30,
            substeps=20,

            yield_stress=30.,
            vol=(1. / 64 / 2) ** 2,  # based on plasticinelab's setup
            mass=(1. / 64 / 2) ** 2,
            E=5000.,
            nu=0.2,
    ):
        super().__init__()
        # self.cfg = cfg
        # self.process_actors(actors)
        self.dx = dx
        self.dt = dt
        self.max_particles = n_particles
        self.n_particles = n_particles
        self.n_bodies = n_bodies
        self.substeps = substeps

        grid_dim = np.ceil(np.array(grid_size) / self.dx / 4).astype(int) * 4
        self.grid_length = grid_dim[0] * grid_dim[1] * grid_dim[2]

        self.dx = float(dx)
        self.inv_dx = 1.0 / self.dx
        self.dt = float(dt)
        self.grid_dim = ivec3(grid_dim[0], grid_dim[1], grid_dim[2])

        self.gravity = array(dtype=vec3, length=1)
        self.particle_vol = array(dtype=float32, length=self.n_particles)
        self.particle_mass = array(dtype=float32, length=self.n_particles)
        self.particle_mu_lam_yield = array(dtype=vec3, length=self.n_particles)
        self.particle_color = array(dtype=int, length=self.n_particles)

        self.particle_ids = array(dtype=int, length=self.n_particles)

        self.dists = array(dtype=float, length=self.n_particles * n_bodies)
        self.dists_grad = array(dtype=float, length=self.n_particles * n_bodies)

        if n_bodies:
            self.body_type_mu_softness_round = array(dtype=quat, length=n_bodies)
            self.body_args = array(dtype=quat, length=n_bodies)  # four args now ..

        self.temp = TempState(self.n_particles, grid_dim, self.n_bodies)
        self.max_steps = max_steps
        self.states = [State(self.n_particles, n_bodies) for _ in range(max_steps)]

        self.stream0 = lib.cuda_stream_create()  # compute stream
        self.stream1 = lib.cuda_stream_create()  # data stream

        mu = E / (2 * (1 + nu))
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.init_particles(
            np.zeros(self.n_particles) + vol,
            np.zeros(self.n_particles) + mass,
            np.zeros((self.n_particles, 3)) + np.array([mu, lam, yield_stress]),
        )
        self.n_bodies = n_bodies

        self.object_id = None

        self.torch_scale = None

        self.n_grid = grid_dim[0]

        lib.print_memory_info()

    def clear_grad(self, max_steps):
        for i in range(max_steps):
            self.states[i].clear_grad()

    def get_state(self, index):
        # to be consistent with the PlasticineLab design
        state = self.states[index].get_state(self.n_particles)
        return tuple(state[:4]) + tuple([np.r_[a, b] for a, b in zip(state[4], state[5])])

    def set_state(self, index, state):
        state = [np.float32(i) for i in state]
        self.n_particles = int(len(state[0]))

        pos = [i[:3] for i in state[4:]]
        rot = [i[3:] for i in state[4:]]
        pos = np.float32(pos)
        rot = np.float32(rot)
        state = state[:4] + [pos, rot]
        # for i in range(4):
        #   empty = np.zeros((self.max_particles, *state[i].shape[1:]))
        #   empty[:self.n_particles] = state[i]
        #   state[i] = empty
        self.states[index].set_state(state)
        self.cur = index

    def set_object_id(self, object_id):
        self.object_id = object_id
        assert len(object_id) == self.n_particles, f"{len(object_id)} {self.n_particles}"
        self.particle_ids.upload(self.object_id)

    def get_object_id(self, device):
        assert device == 'numpy'
        # print(self.object_id.shape, self.n_particles)
        return self.object_id

    def set_color(self, inp):
        color = self.particle_color.download(self.n_particles)
        color[:] = inp
        self.particle_color.upload(color)

    def set_softness(self, softness):
        pp = self.body_type_mu_softness_round.download()
        pp[:, 2] = softness
        self.body_type_mu_softness_round.upload(pp)

    def get_softness(self):
        pp = self.body_type_mu_softness_round.download()
        return pp[:, 2]

    def _initialize_buffer(self, device='numpy', dims=None, dtype=None):
        dims = dims or (3,)
        if device == 'numpy':
            dtype = dtype or np.float32
            x = np.zeros((self.n_particles, *dims), dtype=dtype)
        else:
            import torch
            dtype = dtype or torch.float32
            x = torch.zeros((self.n_particles, *dims), dtype=torch.float32, device=device)
        return x

    def get_x(self, index, device='numpy'):
        return self.states[index].x.download(self.n_particles, device=device)

    def get_v(self, index, device='numpy'):
        return self.states[index].v.download(self.n_particles, device=device)

    def get_dists(self, f, grad=None, device='cuda:0'):
        state = self.states[f]
        if grad is None:
            need_grad = False
        else:
            need_grad = True
            assert grad.shape == (self.n_particles, self.n_bodies)
            grad = grad.reshape(-1).detach().cpu().numpy()
            self.dists_grad.upload(grad)

        lib.compute_dist(
            state.x.data_ptr,
            state.body_pos.data_ptr,
            state.body_rot.data_ptr,
            self.body_type_mu_softness_round.data_ptr,
            self.body_args.data_ptr,
            self.dists.data_ptr,
            self.n_bodies,
            state.x_grad.data_ptr,
            state.body_pos_grad.data_ptr,
            state.body_rot_grad.data_ptr,
            self.dists_grad.data_ptr,
            int(need_grad),
            self.n_particles,
            self.stream0
        )
        if grad is None:
            return self.dists.download(self.n_particles * self.n_bodies, device=device).reshape(-1, self.n_bodies)

    def compute_grid_mass(self, f, id=-1, device='numpy', backward_grad=None):
        need_grad = backward_grad is not None
        # assert not need_grad

        if isinstance(f, int):
            n_particles = self.n_particles
            state = self.states[f]
        else:
            if isinstance(f, torch.Tensor):
                f = f.detach().cpu().numpy()
            assert f.shape[-1] == 3 and id == -1
            state = self.states[-1]
            state.x.upload(np.float32(f))
            n_particles = len(f)
            if need_grad:
                state.clear_grad(self.stream0)
            # assert not need_grad

        if backward_grad is not None:
            self.temp.grid_m_grad.upload(backward_grad.reshape(-1).detach().cpu().numpy(), strict=True)

        self.temp.grid_m.zero()
        # raise NotImplementedError
        lib.particle2mass(state.x.data_ptr, self.particle_mass.data_ptr, self.temp.grid_lower.data_ptr, self.grid_dim,
                          self.dx, self.inv_dx, self.temp.grid_m.data_ptr, self.temp.grid_m_grad.data_ptr,
                          state.x_grad.data_ptr, self.particle_ids.data_ptr, id, need_grad, n_particles, self.stream0)

        if backward_grad is None:
            return self.temp.grid_m.download(device=device).reshape(*self.grid_dim)

    def compute_svd(self, index=0, device=None, backward_grad=None):
        raise NotImplementedError

    def get_tool_state(self, index=0, device=None):
        pos = self.states[index].body_pos.download()
        rot = self.states[index].body_rot.download()
        state = [i for i in np.concatenate((pos, rot), axis=1)]
        if device == 'numpy':
            return state
        else:
            return [torch.tensor(i).to(device) for i in state]

    def set_tool_state(self, index, pose):
        pose = np.array(pose)
        assert pose.shape == (self.n_bodies, 7), f"{pose.shape}, {self.n_bodies}"
        pos = pose[:, :3]
        rot = pose[:, 3:7]
        self.states[index].body_pos.upload(pos)
        self.states[index].body_rot.upload(rot)

    def get_action_scales(self):
        return np.array(self.action_scales)

    def init_particles(self, vol, mass, mu_lam_yield):
        assert vol.shape == (self.n_particles,)
        assert mass.shape == (self.n_particles,)
        assert mu_lam_yield.shape == (self.n_particles, 3)

        self.particle_vol.upload(vol)
        self.particle_mass.upload(mass)
        self.particle_mu_lam_yield.upload(mu_lam_yield)

        self.gravity.upload(np.float32([np.array(self._cfg.gravity) * 30]))
        self.temp.grid_lower.upload(grid_lower_zero)

    def init_bodies(self, types, softness, mu, round, args, action_scales, pos=None, rot=None):
        assert len(mu) == len(args) == self.n_bodies
        mu = np.float32(mu)
        softness = np.float32(softness)
        types = np.float32(types)
        round = np.float32(round)
        args = np.float32(args)

        self.body_type_mu_softness_round.upload(
            np.stack((types, mu, softness, round), 1).reshape(-1, 4))
        self.body_args.upload(args)
        self.action_scales = action_scales

        if pos is not None:
            self.states[0].body_pos.upload(np.array(pos).reshape(self.n_bodies, 3))
        if rot is not None:
            self.states[0].body_rot.upload(np.array(rot).reshape(self.n_bodies, 4))

    def sync(self):
        lib.cuda_stream_sync(self.stream1)
        lib.cuda_stream_sync(self.stream0)

    def __del__(self):
        if lib is not None:
            lib.cuda_stream_destroy(self.stream0)
            lib.cuda_stream_destroy(self.stream1)

    def nan_check(self, i, message=""):
        if np.isnan(self.states[i].particle_x.download()).any():
            print("x nan", message)
        if np.isnan(self.states[i].particle_v.download()).any():
            print("v nan", message)
        if np.isnan(self.states[i].particle_F.download()).any():
            print("F nan", message)
        if np.isnan(self.states[i].particle_C.download()).any():
            print("C nan", message)

    def compute_grid_lower(self, state: State, temp: TempState):
        lib.compute_grid_lower(
            state.x.data_ptr,
            self.dx,
            self.inv_dx,
            temp.grid_lower.data_ptr,
            self.n_particles,
            self.stream0,
        )

    def compute_svd(self, state: State, temp: TempState):
        lib.compute_svd(
            state.F.data_ptr,
            state.C.data_ptr,
            temp.F.data_ptr,
            temp.U.data_ptr,
            temp.V.data_ptr,
            temp.sig.data_ptr,
            self.dt,
            self.n_particles,
            self.stream0,
        )

    def compute_svd_grad(self, state: State, temp: TempState):
        lib.compute_svd_grad(
            state.F.data_ptr,
            state.C.data_ptr,
            temp.U.data_ptr,
            temp.V.data_ptr,
            temp.sig.data_ptr,

            temp.F_grad.data_ptr,
            temp.U_grad.data_ptr,
            temp.V_grad.data_ptr,
            temp.sig_grad.data_ptr,
            state.F_grad.data_ptr,
            state.C_grad.data_ptr,

            self.dt,
            self.n_particles,
            self.stream0,
        )

    def p2g(self, state1: State, temp: TempState, state2: State):
        lib.p2g(
            state1.x.data_ptr, state1.v.data_ptr, self.particle_mass.data_ptr, self.particle_vol.data_ptr,
            temp.F.data_ptr, temp.U.data_ptr, temp.sig.data_ptr, temp.V.data_ptr, state1.C.data_ptr,
            self.particle_mu_lam_yield.data_ptr, temp.grid_lower.data_ptr, self.grid_dim, self.dx, self.inv_dx, self.dt,
            state2.F.data_ptr, temp.grid_v_in.data_ptr, temp.grid_m.data_ptr, self.n_particles, self.stream0
        )

    def p2g_grad(self, state1: State, temp: TempState, state2: State):
        lib.p2g_grad(
            state1.x.data_ptr, state1.v.data_ptr, self.particle_mass.data_ptr, self.particle_vol.data_ptr,
            temp.F.data_ptr, temp.U.data_ptr, temp.sig.data_ptr, temp.V.data_ptr, state1.C.data_ptr,
            self.particle_mu_lam_yield.data_ptr, temp.grid_lower.data_ptr, self.grid_dim, self.dx, self.inv_dx, self.dt,
            state2.F.data_ptr, temp.grid_v_in.data_ptr, temp.grid_m.data_ptr,

            state1.x_grad.data_ptr, state1.v_grad.data_ptr, temp.F_grad.data_ptr, state1.C_grad.data_ptr,
            temp.U_grad.data_ptr, temp.sig_grad.data_ptr, temp.V_grad.data_ptr, state2.F_grad.data_ptr,
            temp.grid_v_in_grad.data_ptr, temp.grid_m_grad.data_ptr,

            self.n_particles, self.stream0
        )

    def get_ground_friction(self):
        return float(self._cfg.ground_friction)

    def get_ground_height(self):
        return float(3.)

    def grid_op(self, state1: State, temp: TempState, state2: State):
        lib.grid_op_v2(
            temp.grid_m.data_ptr, temp.grid_v_in.data_ptr, temp.grid_body_v_in.data_ptr,
            temp.grid_lower.data_ptr, self.gravity.data_ptr,
            state1.body_pos.data_ptr, state1.body_rot.data_ptr,
            state2.body_pos.data_ptr, state2.body_rot.data_ptr,
            self.body_type_mu_softness_round.data_ptr, self.body_args.data_ptr,
            self.dx, self.inv_dx, self.dt,
            self.get_ground_friction(),
            temp.grid_v_out.data_ptr, self.grid_dim, self.n_bodies,
            self.stream0,
        )

    def grid_op_grad(self, state1: State, temp: TempState, state2: State):
        lib.grid_op_v2_grad(
            temp.grid_m.data_ptr, temp.grid_v_in.data_ptr, temp.grid_body_v_in.data_ptr,
            temp.grid_lower.data_ptr, self.gravity.data_ptr,
            state1.body_pos.data_ptr, state1.body_rot.data_ptr,
            state2.body_pos.data_ptr, state2.body_rot.data_ptr,
            self.body_type_mu_softness_round.data_ptr, self.body_args.data_ptr,

            temp.grid_m_grad.data_ptr, temp.grid_v_in_grad.data_ptr, state1.body_pos_grad.data_ptr,
            state1.body_rot_grad.data_ptr, state2.body_pos_grad.data_ptr, state2.body_rot_grad.data_ptr,

            self.dx, self.inv_dx, self.dt,
            self.get_ground_friction(),
            temp.grid_v_out.data_ptr,
            temp.grid_v_out_grad.data_ptr,
            self.grid_dim, self.n_bodies,
            self.stream0,
        )

    def g2p(self, state1: State, temp: TempState, state2: State):
        lib.g2p(
            state1.x.data_ptr, temp.grid_v_out.data_ptr, temp.grid_lower.data_ptr,
            self.dx, self.inv_dx, self.dt,
            self.grid_dim, state2.v.data_ptr, self.get_ground_height(),
            state2.C.data_ptr, state2.x.data_ptr, self.n_particles, self.stream0
        )
        # print(state2.x.download()[0])

    def g2p_grad(self, state1: State, temp: TempState, state2: State):
        lib.g2p_grad(
            state1.x.data_ptr, temp.grid_v_out.data_ptr, temp.grid_lower.data_ptr,
            self.dx, self.inv_dx, self.dt,
            self.grid_dim, state2.v.data_ptr, self.get_ground_height(),
            state2.C.data_ptr, state2.x.data_ptr, self.n_particles,

            state1.x_grad.data_ptr,
            temp.grid_v_out_grad.data_ptr,
            state2.v_grad.data_ptr,
            state2.C_grad.data_ptr,
            state2.x_grad.data_ptr,

            self.stream0
        )

    def set_pose(self, state, pos, rot, stream):
        if isinstance(pos, torch.Tensor):
            pos = pos.detach().cpu().numpy()
        if isinstance(rot, torch.Tensor):
            rot = rot.detach().cpu().numpy()
        state.body_pos.upload_async(pos, stream=stream)
        state.body_rot.upload_async(rot, stream=stream)

    def substep(self, f, clear_grad=False):
        temp, cur, next = self.temp, self.states[f], self.states[f + 1]
        self.temp.clear(self.stream0)
        # self.sync()
        self.compute_svd(cur, temp)
        self.p2g(cur, temp, next)
        self.grid_op(cur, temp, next)
        self.g2p(cur, temp, next)

        if clear_grad:
            next.clear_grad(self.stream1)  # async clear grad

    def substep_grad(self, f):
        temp, cur, next = self.temp, self.states[f], self.states[f + 1]
        self.temp.clear(self.stream0)
        self.temp.clear_grad(self.stream0)
        # self.sync()
        self.compute_svd(cur, temp)
        self.p2g(cur, temp, next)
        self.grid_op(cur, temp, next)

        self.g2p_grad(cur, temp, next)
        self.grid_op_grad(cur, temp, next)
        self.p2g_grad(cur, temp, next)
        self.compute_svd_grad(cur, temp)

    def download_pos_rot(self, cur, device):
        pos, rot = self.states[cur].body_pos.download(), self.states[cur].body_rot.download()
        pos, rot = torch.tensor(pos, device=device, dtype=torch.float32), torch.tensor(rot, device=device,
                                                                                       dtype=torch.float32)
        return pos, rot

    # substeps 2000
    # pytorch batch 
    # theta; pos_p, pos_q = [pos_p, rot_p] * [axis * angle, trans] 

    def compute_forward_kinematics(self, f, action, pos_rot=None):
        cur = f
        device = 'cpu' if not isinstance(action, torch.Tensor) else action.device

        if pos_rot is None:
            pos, rot = self.download_pos_rot(cur, device)
        else:
            pos, rot = pos_rot

        if not isinstance(action, torch.Tensor):
            action = np.array(action, np.float32)
            action = torch.tensor(action, device=device, dtype=torch.float32)
        else:
            action = action.to(device)

        if self.torch_scale is None:
            self.torch_scale = torch.tensor(np.array(self.action_scales), device=device, dtype=torch.float32)
        else:
            self.torch_scale = self.torch_scale.to(device)
        action = action.reshape(-1, 6).clamp(-1., 1.) * self.torch_scale

        pos, rot = rigid_body_motion(
            (pos, rot),
            action[None, :].expand(self.substeps, -1, -1) * (
                    torch.arange(self.substeps, device=device)[:, None, None] + 1
            ) / self.substeps
        )
        return pos, rot, (pos, rot)

    def step(self, action, pos_rot=None):
        pos, rot, q_state = self.compute_forward_kinematics(self.cur, action, pos_rot=pos_rot)

        for f in range(0, self.substeps):
            self.set_pose(self.states[f + 1], pos[f], rot[f], self.stream0)
            self.substep(f)

        self.sync()
        self.states[0], self.states[self.substeps] = self.states[self.substeps], self.states[0]
