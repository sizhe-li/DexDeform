import os
import open3d as o3d
import numpy as np


def length(x):
    return np.sqrt(np.einsum('ij,ij->i', x, x) + 1e-14)


def compute_cylinder_sdf(h, r):
    def sdf_func(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        vec_xz = np.stack([x, z], axis=1)

        rh = np.array([[r, h]])
        d = np.abs(np.stack([length(vec_xz), y], axis=1)) - rh

        return np.minimum(np.maximum(d[:, 0], d[:, 1]), 0.0) + length(np.maximum(d, 0.0))

    return sdf_func


def box_particles(width):
    def sample_func(n_particles):
        p = (np.random.random((n_particles, 3)) * 2 - 1) * width
        return p

    return sample_func


def rejection_sampling(init_pos, n_particles, sample_func, sdf_func):
    p = np.ones((n_particles, 3)) * 5

    remain_cnt = n_particles  # how many left to sample
    while remain_cnt > 0:
        p_samples = sample_func(remain_cnt)
        sdf_vals = sdf_func(p_samples)

        accept_map = sdf_vals <= 0
        accept_cnt = sum(accept_map)
        start = n_particles - remain_cnt
        p[start:start + accept_cnt] = p_samples[accept_map]
        remain_cnt -= accept_cnt
    assert np.all(p != 5)
    p = p + np.array(init_pos)

    return p


class Shapes:
    COLORS = [
        (127 << 16) + 127,
        (127 << 8),
        127,
        127 << 16,
    ]

    # make shapes from the configuration
    def __init__(self, cfg, dim=3):
        self.objects = []
        self.colors = []
        self.object_id = []
        self.mu_lam_yield = []
        self.dim = dim

        state = np.random.get_state()

        np.random.seed(0)  # fix seed 0
        for i in cfg:

            kwargs = {key: eval(val) if isinstance(val, str) else val for key, val in i.items() if key != 'shape'}

            # kwargs = dict()
            # for k, v in i.items():
            #     if k == "shape":
            #         continue
            #     elif k == "load_local":
            #         kwargs[k] = v
            #     elif isinstance(v, str):
            #         kwargs[k] = eval(v)
            print(kwargs)

            if i['shape'] == 'box':
                self.add_box(**kwargs)
            elif i['shape'] == 'sphere':
                self.add_sphere(**kwargs)
            elif i['shape'] == 'cylinder':
                self.add_cylinder(**kwargs)

            else:
                raise NotImplementedError(f"Shape {i['shape']} is not supported!")

        np.random.set_state(state)

    def get_n_particles(self, volume):
        return max(int(volume / 0.2 ** 3) * 10000, 1)

    def add_object(self, particles, color=None, init_rot=None, **extras):
        if init_rot is not None:
            import transforms3d
            q = transforms3d.quaternions.quat2mat(init_rot)
            origin = particles.mean(axis=0)
            particles = (particles[:, :self.dim] - origin) @ q.T + origin
        self.objects.append(particles[:, :self.dim])

        if color is None or isinstance(color, int):
            tmp = self.COLORS[len(self.objects) - 1] if color is None else color
            color = np.zeros(len(particles), np.int32)
            color[:] = tmp

        self.object_id.append([len(self.object_id)] * len(particles))
        self.colors.append(color)

        if any([x in extras for x in ("yield_stress", "E", "nu")]):
            yield_stress = extras.get("yield_stress", 30.)
            E = extras.get("E", 5000.)
            nu = extras.get("nu", 0.2)

            mu = E / (2 * (1 + nu))
            lam = E * nu / ((1 + nu) * (1 - 2 * nu))

            self.mu_lam_yield.append(np.zeros((len(particles), 3)) + np.array([mu, lam, yield_stress]))

    def add_box(self, init_pos, width, n_particles=10000, color=None, init_rot=None, **extras):
        # pass
        if isinstance(width, float):
            width = np.array([width] * self.dim)
        else:
            width = np.array(width)
        if n_particles is None:
            n_particles = self.get_n_particles(np.prod(width))

        p = (np.random.random((n_particles, self.dim)) * 2 - 1) * (0.5 * width) + np.array(init_pos)
        self.add_object(p, color, init_rot=init_rot, **extras)

    def add_sphere(self, init_pos, radius, n_particles=10000, color=None, init_rot=None, **extras):
        if n_particles is None:
            if self.dim == 3:
                volume = (radius ** 3) * 4 * np.pi / 3
            else:
                volume = (radius ** 2) * np.pi
            n_particles = self.get_n_particles(volume)

        p = np.random.normal(size=(n_particles, self.dim))
        p /= np.linalg.norm(p, axis=-1, keepdims=True)
        u = np.random.random(size=(n_particles, 1)) ** (1. / self.dim)
        p = p * u * radius + np.array(init_pos)[:self.dim]
        self.add_object(p, color, init_rot=init_rot, **extras)

    def add_cylinder(self, init_pos, h, r, n_particles=10000, color=None, init_rot=None, **extras):
        sdf_func = compute_cylinder_sdf(h=h, r=r)
        sample_func = box_particles(np.array([r, h, r]))

        p = rejection_sampling(init_pos=np.array(init_pos),
                               n_particles=n_particles,
                               sample_func=sample_func,
                               sdf_func=sdf_func)

        self.add_object(p, color, init_rot=init_rot, **extras)

    def get(self):
        assert len(self.objects) > 0, "please add at least one shape into the scene"

        mu_lam_yield = None if len(self.mu_lam_yield) == 0 else np.concatenate(self.mu_lam_yield, axis=0)

        return np.concatenate(self.objects), np.concatenate(self.colors), np.concatenate(self.object_id), mu_lam_yield

    def remove_object(self, index):
        self.objects.pop(index)
        self.colors.pop(index)


def xyz_spherical(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    r = np.sqrt(x * x + y * y + z * z)
    r_x = np.arccos(y / r)
    r_y = np.arctan2(z, x)
    return [r, r_x, r_y]


def get_rotation_matrix(r_x, r_y):
    rot_x = np.asarray([[1, 0, 0], [0, np.cos(r_x), -np.sin(r_x)],
                        [0, np.sin(r_x), np.cos(r_x)]])
    rot_y = np.asarray([[np.cos(r_y), 0, np.sin(r_y)], [0, 1, 0],
                        [-np.sin(r_y), 0, np.cos(r_y)]])
    return rot_y.dot(rot_x)


def get_extrinsic(xyz):
    rvec = xyz_spherical(xyz)
    r = get_rotation_matrix(rvec[1], rvec[2])
    t = np.asarray([0, 0, 2]).transpose()
    trans = np.eye(4)
    trans[:3, :3] = r
    trans[:3, 3] = t
    return trans


def preprocess(model):
    min_bound = model.get_min_bound()
    max_bound = model.get_max_bound()
    center = min_bound + (max_bound - min_bound) / 2.0
    scale = np.linalg.norm(max_bound - min_bound) / 2.0
    vertices = np.asarray(model.vertices)
    vertices -= center
    model.vertices = o3d.utility.Vector3dVector(vertices / scale)
    return model


def voxel_carving(mesh,
                  cubic_size=2.0,
                  voxel_resolution=128.0,
                  w=300,
                  h=300,
                  visualize=False):
    """
        carve mesh into voxel with volume preserved
    """
    mesh.compute_vertex_normals()

    # setup dense voxel grid
    voxel_carved = o3d.geometry.VoxelGrid.create_dense(
        width=cubic_size,
        height=cubic_size,
        depth=cubic_size,
        voxel_size=cubic_size / voxel_resolution,
        origin=[-cubic_size / 2.0, -cubic_size / 2.0, -cubic_size / 2.0],
        color=[1.0, 0.7, 0.0])

    # set up camera
    camera_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)

    # rescale geometry
    camera_sphere = preprocess(camera_sphere)
    mesh = preprocess(mesh)

    # setup visualizer to render depthmaps
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h, visible=False)
    vis.add_geometry(mesh)
    vis.get_render_option().mesh_show_back_face = True
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()

    # carve voxel grid
    pcd_agg = o3d.geometry.PointCloud() if visualize else None
    centers_pts = np.zeros((len(camera_sphere.vertices), 3))
    for cid, xyz in enumerate(camera_sphere.vertices):
        # get new camera pose
        trans = get_extrinsic(xyz)
        param.extrinsic = trans
        c = np.linalg.inv(trans).dot(np.asarray([0, 0, 0, 1]).transpose())
        centers_pts[cid, :] = c[:3]
        ctr.convert_from_pinhole_camera_parameters(param)

        # capture depth image and make a point cloud
        vis.poll_events()
        vis.update_renderer()
        depth = vis.capture_depth_float_buffer(False)

        if visualize:
            pcd_agg += o3d.geometry.PointCloud.create_from_depth_image(
                o3d.geometry.Image(depth),
                param.intrinsic,
                param.extrinsic,
                depth_scale=1)

        voxel_carved.carve_depth_map(o3d.geometry.Image(depth), param)
        print("Carve view %03d/%03d" % (cid + 1, len(camera_sphere.vertices)))
    vis.destroy_window()

    if visualize:
        # add voxel grid surface
        print('Surface voxel grid from pointcloud')
        voxel_surface = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            pcd_agg,
            voxel_size=cubic_size / voxel_resolution,
            min_bound=(-cubic_size / 2, -cubic_size / 2, -cubic_size / 2),
            max_bound=(cubic_size / 2, cubic_size / 2, cubic_size / 2))

        voxel_carving_surface = voxel_surface + voxel_carved

        return voxel_carving_surface, voxel_carved

    return voxel_carved


def transform_geom(geom, max_dim_size=0.3, goal_center=(0.5, 0.3, 0.5)):
    """
        centers refer to the center of the bottom here
    """
    curr_center = geom.get_center()
    goal_center = np.array(goal_center)

    scale = max_dim_size / (geom.get_max_bound() - geom.get_min_bound()).max()
    geom.scale(scale, center=curr_center)

    curr_center[1] = geom.get_min_bound()[1]
    trans = goal_center - curr_center

    geom.translate(trans)
    return geom


def sample_points_inside_voxel_grid(voxel_grid, n_particles=10000):
    xmin, ymin, zmin = voxel_grid.get_min_bound()
    xmax, ymax, zmax = voxel_grid.get_max_bound()

    p = np.ones((n_particles, 3)) * 5
    remain_cnt = n_particles
    while remain_cnt > 0:
        points = np.random.random((remain_cnt, 3))
        points[:, 0] = points[:, 0] * (xmax - xmin) + xmin
        points[:, 1] = points[:, 1] * (ymax - ymin) + ymin
        points[:, 2] = points[:, 2] * (zmax - zmin) + zmin

        accept_map = voxel_grid.check_if_included(o3d.utility.Vector3dVector(points))
        accept_map = np.array(accept_map)
        accept_cnt = accept_map.sum()
        start = n_particles - remain_cnt
        p[start:start + accept_cnt] = points[accept_map]
        remain_cnt -= accept_cnt

    assert np.all(p != 5)
    return p


def sample_mesh_volume(mesh_path, max_dim_size, goal_center, n_particles=10000):
    assert os.path.exists(mesh_path)
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    voxel_grid = voxel_carving(mesh)
    np_pcd = sample_points_inside_voxel_grid(voxel_grid, n_particles)
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(np_pcd)
    o3d_pcd = transform_geom(o3d_pcd, max_dim_size, goal_center)
    return np.array(o3d_pcd.points)
