import numpy as np
import torch
from .cuda_env import CudaEnv
from torch.autograd import Function


class GradModel:
    def __init__(self, env: CudaEnv, return_grid=(-1,), return_svd=False):
        self.env = env
        self.sim = env.simulator
        self.dim = 3
        assert return_grid[0] == -1
        self.primitives = [i for i in range(self.sim.n_bodies)]
        self._forward_func = None
        self._set_pose_func = None
        self.return_grid = return_grid
        self.return_svd = return_svd

    @property
    def substeps(self):
        return self.sim.substeps


    def zero_grad(self, return_grid=None, return_svd=None, **kwargs):
        self.device = 'cuda:0'
        self.sim.states[0].clear_grad(self.sim.stream1) # clear grad for states 0
        if return_grid is not None:
            self.return_grid = return_grid
        if return_svd is not None:
            self.return_svd = return_svd or self.return_svd


    def wrap_obs(self, obs):
        output = {
            'pos': obs[0][:, :3],
            'vel': obs[0][:, 3:6],
            'tool': obs[1],
        }
        output['dist'] = obs[0][:, 6:]

        obs = obs[2:]
        if len(self.return_grid) > 0:
            ng = len(self.return_grid)
            grid, obs = obs[:ng], obs[ng:]
            output['grid'] = {
                k: v for k, v in zip(self.return_grid, grid)
            }
        #if self.return_svd:
        #    output['eigen'], obs = obs[0], obs[1:]
        assert len(obs) == 0
        return output


    def get_obs(self, s, device):
        f = s * self.sim.substeps
        x = self.sim.get_x(f, device)
        v = self.sim.get_v(f, device)
        c = torch.stack(self.sim.get_tool_state(f, device))
        dists = self.sim.get_dists(f, device=device)

        x = torch.cat((x, v, dists), 1)

        outputs = [x, c]

        if len(self.return_grid) > 0:
            outputs += [
                self.sim.compute_grid_mass(
                    f, i, device=device)
                for i in self.return_grid
            ]
        
        #if self.return_svd:
        #    outputs.append(
        #        self.sim.compute_svd(f, device, None)
        #    )

        return tuple(outputs)

    def set_obs_grad(self, s, particle_grad, tool_grad, *args):
        f = s * self.sim.substeps

        #if self.return_svd:
        #    svd_grad, args = args[-1], args[:-1]
        #    self.sim.compute_svd(f, svd_grad.device, backward_grad=svd_grad)

        if len(self.return_grid) > 0:
            for idx, i in enumerate(self.return_grid):
                self.sim.compute_grid_mass(f, i, backward_grad=args[idx])

        n_bodies = self.sim.n_bodies
        start = -n_bodies
        dist_grads = particle_grad[:, start:]
        self.sim.get_dists(f, dist_grads)

        particle_grad = particle_grad[..., :start]
        particle_grad = particle_grad.reshape(-1, self.dim * 2)
        tool_grad = tool_grad.reshape(n_bodies, 7)

        x = particle_grad.detach().cpu().numpy()
        c = tool_grad.detach().cpu().numpy()
        state = self.sim.states[f]
        state.x_grad.cuda_add(x[:, :3])
        state.v_grad.cuda_add(x[:, 3:])
        state.body_pos_grad.cuda_add(c[:, :3])
        state.body_rot_grad.cuda_add(c[:, 3:])

    @property
    def diff_forward(self):
        if self._forward_func is None:
            class forward(Function):
                @staticmethod
                def forward(ctx, s, pos, rot, *past_obs):
                    ctx.save_for_backward(torch.tensor([s]), *[torch.zeros_like(i) for i in past_obs])
                    f = s * self.substeps
                    for i in range(self.substeps):
                        self.sim.set_pose(self.sim.states[f+i+1], pos[i], rot[i], self.sim.stream0)
                        self.sim.substep(f+i, clear_grad=True)
                    self.sim.sync()
                    return self.get_obs(s+1, pos.device)

                @staticmethod
                def backward(ctx, *obs_grad):
                    zero_grads = ctx.saved_tensors
                    s = zero_grads[0].item()
                    self.set_obs_grad(s+1, *obs_grad) # add the gradient back into the tensors ...
                    f = s * self.substeps

                    pos = []
                    rot = []
                    for i in range(f + self.substeps-1, f-1, -1):
                        self.sim.substep_grad(i)
                        state = self.sim.states[i+1]
                        pos.append(state.body_pos_grad.download(stream=self.sim.stream1))
                        rot.append(state.body_rot_grad.download(stream=self.sim.stream1))

                    self.sim.sync()

                    pos_grad = torch.tensor(np.array(pos[::-1]), device=self.device)
                    rot_grad = torch.tensor(np.array(rot[::-1]), device=self.device)

                    return (None, pos_grad, rot_grad) + zero_grads[1:]

            self._forward_func = forward
        return self._forward_func.apply

    @property
    def diff_set_pose(self):
        if self._set_pose_func is None:
            class SetPose(Function):
                @staticmethod
                def forward(ctx, s, pos, rot):
                    ctx.s = s
                    self.sim.set_pose(self.sim.states[s * self.substeps], pos, rot, self.sim.stream0)
                    self.sim.sync()
                    return self.get_obs(s, pos.device)

                @staticmethod
                def backward(ctx, *obs_grad):
                    s = ctx.s
                    self.set_obs_grad(s, *obs_grad) # add the gradient back into the tensors ...
                    state = self.sim.states[s*self.substeps]
                    pos = state.body_pos_grad.download(stream=self.sim.stream0)
                    rot = state.body_rot_grad.download(stream=self.sim.stream0)
                    self.sim.sync()
                    pos_grad = torch.tensor(pos, device=self.device)
                    rot_grad = torch.tensor(rot, device=self.device)
                    state.body_pos_grad.zero()
                    state.body_pos_grad.zero()
                    return (None, pos_grad, rot_grad)
            self._set_pose_func = SetPose.apply
            
        return self._set_pose_func


    def forward(self, s, action, *past_obs, pos_rot=None):
        #TODO: rename pos_rot to qpos
        if s == 0 and pos_rot is None:
            self.pos_rot = self.sim.download_pos_rot(0, action.device) # pass grad along trajectory
        if pos_rot is not None:
            assert isinstance(pos_rot[0], torch.Tensor)
            past_obs = self.diff_set_pose(s, *pos_rot)
        pos, rot, q_state = self.sim.compute_forward_kinematics(s * self.substeps, action, pos_rot=self.pos_rot)
        self.pos_rot = q_state
        return self.diff_forward(s, pos, rot, *past_obs)