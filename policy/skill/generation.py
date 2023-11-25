import time

import numpy as np
import torch
import torch.optim as optim
import trimesh
from torch import autograd
from tqdm import trange

from policy.perception.common import make_3d_grid
from policy.perception.utils import libmcubes
from policy.perception.utils.libmise import MISE
from policy.perception.utils.libsimplify import simplify_mesh

counter = 0

# TODO:
#  (1) change to fit the dataloader scheme
#  (2) use inheritance

class Generator3D(object):
    '''  Generator class for Occupancy Networks.
    It provides functions to generate the final mesh as well refining options.
    Args:
        model (nn.Module): trained Occupancy Network model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        sample (bool): whether z should be sampled
        input_type (str): type of input
        vol_info (dict): volume infomation
        vol_bound (dict): volume boundary
        simplify_nfaces (int): number of faces the mesh should be simplified to
    '''

    def __init__(self, model, points_batch_size=100000,
                 threshold=0.5, refinement_step=0, device=None,
                 resolution0=16, upsampling_steps=3,
                 with_normals=False, padding=0.1, sample=False,
                 input_type=None,
                 vol_info=None,
                 vol_bound=None,
                 simplify_nfaces=None):
        self.model = model.to(device)
        self.points_batch_size = points_batch_size
        self.refinement_step = refinement_step
        self.threshold = threshold
        self.device = device
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.with_normals = with_normals
        self.input_type = input_type
        self.padding = padding
        self.sample = sample
        self.simplify_nfaces = simplify_nfaces

        # for pointcloud_crop
        self.vol_bound = vol_bound
        if vol_info is not None:
            self.input_vol, _, _ = vol_info

    def generate_mesh(self, data, return_stats=True, **kwargs):
        ''' Generates the output mesh.
        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device

        obj_obs = data.get('src_obs').to(device)

        t0 = time.time()

        scene_dict = dict()
        with torch.no_grad():
            c = self.model.encode_scene(obj_obs)
            scene_dict["c"] = c
            if kwargs.get("use_vae", False):
                z, mu, logvar = self.model.vae_btlneck.encode(c)
                vae_c = self.model.vae_btlneck.decode(z)
                scene_dict["vae_c"] = vae_c

        stats_dict = {'time (encode inputs)': time.time() - t0}

        mesh_ret_dict = dict()
        for name, c in scene_dict.items():
            mesh_ret_dict[name] = self.generate_from_latent(c, stats_dict=stats_dict)

        if return_stats:
            return mesh_ret_dict, stats_dict
        else:
            return mesh_ret_dict

    def generate_from_latent(self, c=None, stats_dict={}, **kwargs):
        ''' Generates mesh from latent.
            Works for shapes normalized to a unit cube
        Args:
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)

        t0 = time.time()
        # Compute bounding box size
        box_size = 1 + self.padding

        # Shortcut
        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid(
                (-0.5,) * 3, (0.5,) * 3, (nx,) * 3
            )

            values = self.eval_points(pointsf, c, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)

            points = mesh_extractor.query()
            while points.shape[0] != 0:
                # Query points
                pointsf = points / mesh_extractor.resolution
                # Normalize to bounding box
                pointsf = box_size * (pointsf - 0.5)
                pointsf = torch.FloatTensor(pointsf).to(self.device)
                # Evaluate model and update
                values = self.eval_points(pointsf, c, **kwargs).cpu().numpy()
                values = values.astype(np.float64)
                mesh_extractor.update(points, values)
                points = mesh_extractor.query()

            value_grid = mesh_extractor.to_dense()

        # Extract mesh
        stats_dict['time (eval points)'] = time.time() - t0

        mesh = self.extract_mesh(value_grid, c, stats_dict=stats_dict)
        return mesh

    def eval_points(self, p, c=None, vol_bound=None, **kwargs):
        ''' Evaluates the occupancy values for the points.
        Args:
            p (tensor): points
            c (tensor): encoded feature volumes
        '''
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []
        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                occ_hat = self.model.decode_for_generation(pi, c, **kwargs).logits
            occ_hats.append(occ_hat.squeeze(0).detach().cpu())

        occ_hat = torch.cat(occ_hats, dim=0)
        return occ_hat

    def extract_mesh(self, occ_hat, c=None, stats_dict=dict()):
        ''' Extracts the mesh from the predicted occupancy grid.
        Args:
            occ_hat (tensor): value grid of occupancies
            c (tensor): encoded feature volumes
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + self.padding
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        # Make sure that mesh is watertight
        t0 = time.time()
        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold)
        stats_dict['time (marching cubes)'] = time.time() - t0
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # # Undo padding
        vertices -= 1

        if self.vol_bound is not None:
            # Scale the mesh back to its original metric
            bb_min = self.vol_bound['query_vol'][:, 0].min(axis=0)
            bb_max = self.vol_bound['query_vol'][:, 1].max(axis=0)
            mc_unit = max(bb_max - bb_min) / (
                    self.vol_bound['axis_n_crop'].max() * self.resolution0 * 2 ** self.upsampling_steps)
            vertices = vertices * mc_unit + bb_min
        else:
            # Normalize to bounding box
            vertices /= np.array([n_x - 1, n_y - 1, n_z - 1])
            vertices = box_size * (vertices - 0.5)

        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               vertex_normals=None,
                               process=False)

        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        # TODO: normals are lost here
        if self.simplify_nfaces is not None:
            t0 = time.time()
            mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.)
            stats_dict['time (simplify)'] = time.time() - t0

        # Refine mesh
        if self.refinement_step > 0:
            t0 = time.time()
            self.refine_mesh(mesh, occ_hat, c)
            stats_dict['time (refine)'] = time.time() - t0

        return mesh

    def estimate_normals(self, vertices, c=None):
        ''' Estimates the normals by computing the gradient of the objective.
        Args:
            vertices (numpy array): vertices of the mesh
            c (tensor): encoded feature volumes
        '''
        device = self.device
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, self.points_batch_size)

        normals = []
        c = c.unsqueeze(0)
        for vi in vertices_split:
            vi = vi.unsqueeze(0).to(device)
            vi.requires_grad_()
            occ_hat = self.model.decode_for_generation(vi, c).logits
            out = occ_hat.sum()
            out.backward()
            ni = -vi.grad
            ni = ni / torch.norm(ni, dim=-1, keepdim=True)
            ni = ni.squeeze(0).cpu().numpy()
            normals.append(ni)

        normals = np.concatenate(normals, axis=0)
        return normals

    def refine_mesh(self, mesh, occ_hat, c=None):
        ''' Refines the predicted mesh.
        Args:
            mesh (trimesh object): predicted mesh
            occ_hat (tensor): predicted occupancy grid
            c (tensor): latent conditioned code c
        '''

        self.model.eval()

        # Some shorthands
        n_x, n_y, n_z = occ_hat.shape
        assert (n_x == n_y == n_z)
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        threshold = self.threshold

        # Vertex parameter
        v0 = torch.FloatTensor(mesh.vertices).to(self.device)
        v = torch.nn.Parameter(v0.clone())

        # Faces of mesh
        faces = torch.LongTensor(mesh.faces).to(self.device)

        # Start optimization
        optimizer = optim.RMSprop([v], lr=1e-4)

        for it_r in trange(self.refinement_step):
            optimizer.zero_grad()

            # Loss
            face_vertex = v[faces]
            eps = np.random.dirichlet((0.5, 0.5, 0.5), size=faces.shape[0])
            eps = torch.FloatTensor(eps).to(self.device)
            face_point = (face_vertex * eps[:, :, None]).sum(dim=1)

            face_v1 = face_vertex[:, 1, :] - face_vertex[:, 0, :]
            face_v2 = face_vertex[:, 2, :] - face_vertex[:, 1, :]
            face_normal = torch.cross(face_v1, face_v2)
            face_normal = face_normal / \
                          (face_normal.norm(dim=1, keepdim=True) + 1e-10)
            face_value = torch.sigmoid(
                self.model.decode_for_generation(face_point.unsqueeze(0), c).logits
            )
            normal_target = -autograd.grad(
                [face_value.sum()], [face_point], create_graph=True)[0]

            normal_target = \
                normal_target / \
                (normal_target.norm(dim=1, keepdim=True) + 1e-10)
            loss_target = (face_value - threshold).pow(2).mean()
            loss_normal = \
                (face_normal - normal_target).pow(2).sum(dim=1).mean()

            loss = loss_target + 0.01 * loss_normal

            # Update
            loss.backward()
            optimizer.step()

        mesh.vertices = v.data.cpu().numpy()

        return mesh
