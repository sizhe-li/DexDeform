from mpm.hand import HandEnv
import open3d as o3d
import torch
import numpy as np


class Viewer:
    # TODO: clean up this class
    def __init__(self, env: HandEnv):
        self.env = env

        self.view_order = ["side", "front", "top", "bot"]
        if self.env.cfg.env_name in ["flip"]:
            self.view_order = ["top", "side", "front", "bot"]

        self.views = {
            "front": {"center": (0.5, 0.35, 1.50), "theta": np.pi / 8, "phi": np.pi, "radius": 1.5},
            "side": {"center": (0.5, 0.35, 0.45), "theta": np.pi / 8, "phi": -np.pi / 2, "radius": 1.5},
            "top": {"center": (0.5, 0.95, 0.5), "theta": np.pi / 2, "phi": -np.pi / 2, "radius": 1.5},
            "bot": {"center": (0.5, -0.75, 0.5), "theta": -np.pi / 2, "phi": -np.pi / 2, "radius": 1.5},
        }

        # TODO: move to config file
        if self.env.cfg.env_name == "flip":
            self.views["v1"] = {'center': np.array([0.70002909, 0.36180185, 0.59979075]),
                                'theta': 0.59269908169872414,
                                'phi': -1.5707963267948966,
                                'radius': 1.5}

            self.views["v2"] = {'center': np.array([0.30002909, 0.36180185, 0.59979075]),
                                'theta': 0.59269908169872414,
                                'phi': 1.5707963267948966,
                                'radius': 1.5}

            self.views["v3"] = {'center': np.array([0.50002909, 0.36180185, 0.59979075]),
                                'theta': 1.5707963267948966,
                                'phi': -1.5707963267948966,
                                'radius': 1.5}

            self.views["v4"] = {'center': np.array([0.50002909, 0.46180185, 0.40]),
                                'theta': 0.5,
                                'phi': 0.,
                                'radius': 1.5}
        else:
            self.views["v1"] = {'center': np.array([1.00002909, 0.36180185, 0.59979075]),
                                'theta': 0.39269908169872414,
                                'phi': -1.5707963267948966,
                                'radius': 1.5}

            self.views["v2"] = {'center': np.array([0.00002909, 0.36180185, 0.59979075]),
                                'theta': 0.39269908169872414,
                                'phi': 1.5707963267948966,
                                'radius': 1.5}

            self.views["v3"] = {'center': np.array([0.50002909, 0.96180185, 0.59979075]),
                                'theta': 1.5707963267948966,
                                'phi': -1.5707963267948966,
                                'radius': 1.5}

            self.views["v4"] = {'center': np.array([0.50002909, 0.36180185, -0.10]),
                                'theta': 0.5,
                                'phi': 0.,
                                'radius': 1.5}

    def get_hand_center(self):
        return torch.stack(self.env.simulator.get_tool_state())[:, :3].mean(0)
        # return self.env.simulator.base_pose[0][:, :3, 3].mean(0).detach().cpu().numpy()

    def get_obj_center(self):
        return self.env.simulator.states[0].x.download().mean(0)

    def refresh_views(self, mode="hand_centric", radius=None):
        assert mode in ["hand_centric", "obj_centric"]

        if radius is not None:
            for v in self.views.keys():
                self.views[v]["radius"] = radius

        if mode == "hand_centric":
            print("using hand-centric view")
            center = self.get_hand_center()

            # global views
            self.views["front"]["center"] = center + np.array([0., 0.3, 1.2])
            self.views["side"]["center"] = center + np.array([0.7, 0.3, 0.0])
            if self.env.cfg.env_name in ["flip"]:
                self.views["top"]["center"] = center + np.array([0., 0.2, 0.])
            else:
                self.views["top"]["center"] = center + np.array([0., 0.9, 0.])
            self.views["bot"]["center"] = center + np.array([0., -0.8, 0.])

        else:
            print("using object-centric view")
            center = self.get_obj_center()

            # global views
            self.views["front"]["center"] = center + np.array([0., 0.3, 1.2])
            self.views["side"]["center"] = center + np.array([0.7, 0.3, 0.])
            self.views["top"]["center"] = center + np.array([0., 0.9, 0.])
            self.views["bot"]["center"] = center + np.array([0., -1.0, 0.])

    def render(self, spp=10, primitive=1, shape=1):
        return np.uint8(self.env.render_rgb(spp=spp, primitive=primitive, shape=shape).clip(0, 1) * 255)

    def set_view(self, view_name):
        self.env.renderer.lookat(**self.views[view_name])

    def render_state_multiview(self, spp=10, primitive=1, shape=1, n_views=2):
        images = []

        for view_name in self.view_order[:n_views]:
            self.set_view(view_name)
            images.append(self.render(spp, primitive, shape))

        img = np.concatenate(images[:2], axis=1)
        if n_views == 4:
            img = np.concatenate((img, np.concatenate(images[2:], axis=1)), axis=0)

        return img

    def render_state_as_pcd(self, mode="human"):
        points = self.env.simulator.get_state(0)[0]
        pcd = points_to_pcd(points)

        if mode == "human":
            o3d.visualization.draw_geometries([pcd])

        return pcd

    def render_rgbd_to_pcd(self, spp=10, primitive=1, shape=1,
                           filter_ground=True, device="numpy"):

        img = self.env.render_rgbd(spp=spp, primitive=primitive, shape=shape)
        pcd = self.env.renderer.rgbd2pcd(img[..., :3].clip(0, 1) * 255,
                                         img[..., -1])

        points = np.array(pcd.points, dtype=float)
        colors = np.array(pcd.colors, dtype=float)

        if device != "numpy":
            points = torch.tensor(points, dtype=torch.float32, device=device)
            colors = torch.tensor(colors, dtype=torch.float32, device=device)

        if filter_ground:
            if self.env.renderer.ground_height is not None:
                mask = points[..., 1] >= self.env.renderer.ground_height + 0.1
            else:
                mask = points[..., 1] >= 0.056

            points = points[mask]
            colors = colors[mask]

        return points, colors


def points_to_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def pcd_to_points(pcd):
    return np.array(pcd.points)
