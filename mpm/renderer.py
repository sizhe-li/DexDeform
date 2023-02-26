import numpy as np
import transforms3d
from mpm.types import (
    ivec3,
    array,
    float32,
    vec3,
    mat3,
    lib,
    int64,
)
from tools import Configurable, merge_inputs


class Renderer(Configurable):
    def __init__(
        self,
        cfg=None,
        spp=3,
        voxel_res=(168, 168, 168),
        image_res=(512, 512),
        max_ray_depth=2,
        light_direction = (1., 1., 0.),
        sdf_threshold=0.414,
        bake_size=7,
        inv_dx=150.,
        ground_height=None,
    ):
        super().__init__()
        print('init renderer')

        # added by Sizhe
        self.ground_height = ground_height

        self.voxel_res = ivec3(*voxel_res)
        self.voxel_dim = self.voxel_res[0] * self.voxel_res[1] * self.voxel_res[2]

        self.image_res = ivec3(*image_res, 3)
        self.image_dim = image_res[0] * image_res[1]

        self.bbox_min = array(dtype=vec3, length=1)
        self.bbox_max = array(dtype=vec3, length=1)

        self.volume = array(dtype=int64, length=self.voxel_dim)
        self.sdf_volume = array(dtype=float32, length=self.voxel_dim)
        self.sdf_volume2 = array(dtype=float32, length=self.voxel_dim)
        self.color_volume = array(dtype=vec3, length=self.voxel_dim)

        self.camera_pos = array(dtype=vec3, length=1)
        self.light_direction = array(dtype=vec3, length=1)
        self.camera_rot = array(dtype=mat3, length=1)
        self.camera_intrinsic = array(dtype=mat3, length=1)

        self.color_buffer = array(dtype=vec3, length=self.image_dim)
        self.depth_buffer = array(dtype=float, length=self.image_dim)

        self.center = (0.5, 0.2, 0.5)
        self.theta, self.phi, self.radius = 0., 0., 3


        self.particle_color = None

        self.empty_volume = np.zeros(self.volume.shape, dtype=np.int64)
        self.empty_volume[:] = (1<<33)-1
        self.render_seed = 0

    def initialize_camera(self, rot, pos):
        fov = 0.23
        image_res = self.image_res
        intr = np.array([
            - np.array([2 * fov / image_res[1], 0, -fov - 1e-5,]),
            - np.array([0, 2 * fov / image_res[1], -fov - 1e-5,]),
            [0, 0, 1]
        ]) # * [u, v, 1] * normalize * depth -> [x, y, z]
        self.camera_intrinsic.upload(intr.reshape(1, 9))
        self.camera_rot.upload(rot.reshape(1, 9))
        self.camera_pos.upload(pos.reshape(1, 3))
        self.light_direction.upload(np.array(self._cfg.light_direction).reshape(1, 3))

    def lookat(self, center=(0.5, 0.2, 0.5), theta=0., phi=0., radius=3):
        self.setRT(*lookat(center, theta, phi, radius))
        self.center, self.theta, self.phi, self.radius = center, theta, phi, radius

    def setRT(self, R, T):
        self.initialize_camera(R, T)


    def bake_particles(self, simulator, state, stream=None):
        from .simulator import MPMSimulator, State
        simulator: MPMSimulator = simulator
        state: State = state

        if stream is None:
            stream = simulator.stream0

        cfg = self._cfg

        x = state.x.download()[:simulator.n_particles]
        x = np.floor(x * cfg.inv_dx - 6) / cfg.inv_dx
        bbox_min = x.min(axis=0)[None,:]
        bbox_max = bbox_min + np.array([*self.voxel_res]) / cfg.inv_dx
        self.bbox_min.upload(bbox_min)
        self.bbox_max.upload(bbox_max)
        self.volume.upload(self.empty_volume)

        lib.particle_sdf(
            self.volume.data_ptr,
            state.x.data_ptr,
            simulator.particle_color.data_ptr,
            self.bbox_min.data_ptr,
            self.bbox_max.data_ptr,
            int(cfg.bake_size),
            self.voxel_res,
            float(cfg.inv_dx),
            self.sdf_volume.data_ptr,
            self.color_volume.data_ptr,
            self.sdf_volume2.data_ptr,
            simulator.n_particles,
            stream
        )

    def render(self, simulator, state, stream=None, **kwargs):
        from .simulator import MPMSimulator, State
        simulator: MPMSimulator = simulator
        state: State = state

        if stream is None:
            stream = simulator.stream0

        cfg = merge_inputs(self._cfg, **kwargs)
        self.bake_particles(simulator, state, stream)

        primitive = kwargs.get('primitive', 1)
        visualize_shape = bool(kwargs.get('shape', 1))

        lib.render(
            self.sdf_volume.data_ptr,
            self.bbox_min.data_ptr,
            self.bbox_max.data_ptr,
            self.color_volume.data_ptr,
            state.body_pos.data_ptr,
            state.body_rot.data_ptr,
            simulator.body_type_mu_softness_round.data_ptr,
            simulator.body_args.data_ptr,

            self.camera_rot.data_ptr,
            self.camera_pos.data_ptr,
            self.camera_intrinsic.data_ptr,
            self.color_buffer.data_ptr,
            self.depth_buffer.data_ptr,
            cfg.sdf_threshold,
            self.voxel_res,
            simulator.n_bodies * primitive,
            visualize_shape,
            self.image_res,
            cfg.max_ray_depth,
            cfg.spp,
            (simulator.get_ground_height() * simulator.dx
             if self.ground_height is None else self.ground_height),
            # -0.50,# simulator.get_ground_height() * simulator.dx,
            self.render_seed,
            self.light_direction.data_ptr,
            stream,
        )
        self.render_seed += 1

        color = self.color_buffer.download().reshape(*self.image_res)
        depth = self.depth_buffer.download().reshape(*self.image_res[:2])
        return np.concatenate((color, depth[..., None]), -1)

    def get_int(self, image_res=None):
        fov = 0.23
        image_res = image_res or self.image_res
        int = np.array([
            - np.array([2 * fov / image_res[1], 0, -fov - 1e-5,]),
            - np.array([0, 2 * fov / image_res[1], -fov - 1e-5,]),
            [0, 0, 1]
        ])
        return np.linalg.inv(int)

    def get_ext(self):
        T = np.zeros((4, 4))
        T[:3, :3] = self.camera_rot.download().reshape(3, 3) #self.camera_mat.to_numpy()
        T[:3, 3] = self.camera_pos.download().reshape(3) #self.camera_pos_multi.to_numpy()
        T[3, 3] = 1
        return np.linalg.inv(T)

    def get_o3d_camera(self):
        import open3d as o3d
        int = self.get_int()
        fx, fy = int[0, 0], int[1, 1]
        cx, cy = int[0, 2], int[1, 2]
        w, h = self.image_res[1], self.image_res[0]
        cam = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
        return cam

    def rgbd2pcd(self, rgb, depth):
        import open3d as o3d
        # rgb = rgb[:, ::-1] # notice that we have reversed the image
        # depth = depth[:, ::-1]
        cam_param = self.get_o3d_camera()
        extrinsic = self.get_ext()
        rgb = o3d.geometry.Image(np.ascontiguousarray(np.rot90(rgb, 0, (0, 1))).astype(np.uint8))
        depth = o3d.geometry.Image(np.ascontiguousarray(depth).astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, depth_scale=1., depth_trunc=np.inf,
                                                                  convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam_param)
        pcd.transform(np.linalg.inv(extrinsic))
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        return pcd # pcd.points, pcd.colors


    def draw_geometries(self, objects, mode='human'):
        import open3d as o3d
        if mode == 'human':
            o3d.visualization.draw_geometries(objects)
        else:
            vis = o3d.visualization.Visualizer()
            res = tuple(self.image_res)
            vis.create_window(width=res[0], height=res[1], visible=False)
            if isinstance(objects, tuple) or isinstance(objects, list):
                for geom in objects:
                    vis.add_geometry(geom)
            else:
                vis.add_geometry(objects)
            ctr = vis.get_view_control()
            cam_param = self.get_o3d_camera()

            o3d_cam = o3d.camera.PinholeCameraParameters()
            o3d_cam.intrinsic = cam_param
            o3d_cam.extrinsic = self.get_ext()

            ctr.convert_from_pinhole_camera_parameters(o3d_cam, allow_arbitrary=True)
            vis.update_renderer()
            image = vis.capture_screen_float_buffer(do_render=True)
            vis.destroy_window()
            return np.uint8(np.asarray(image) * 255)

def lookat(center, theta, phi, radius):
    R = transforms3d.euler.euler2mat(theta, phi, 0., 'sxyz')
    b = np.array([0, 0, radius], dtype=float)
    back = R[0:3, 0:3].dot(b)
    return R, center - back
