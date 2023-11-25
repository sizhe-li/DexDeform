import numpy as np
import scipy.stats
import torch
from pytorch3d.ops import sample_farthest_points
from sklearn.neighbors import NearestNeighbors
from transforms3d.quaternions import quat2mat

# SCENE SETTINGS
PLAB_COORD_RANGE = {
    x: np.array(
        [
            [0.05, -0.35, -0.1],
            [1.55, 1.15, 1.4],
        ]
    )
    for x in ("folding", "bun", "dumpling", "rope", "wrap")
}
PLAB_COORD_RANGE["flip"] = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ]
)

IPLCT_COORD_RANGE = np.array([[-0.5] * 3, [0.5] * 3])
DUMP_DATA_TYPE = np.float32


def get_trunc_ab(mean, std, a, b):
    return (a - mean) / std, (b - mean) / std


def get_trunc_ab_range(mean_min, mean_max, std, a, b):
    return (a - mean_min) / std, (b - mean_max) / std


def sample_occupancies(scene_pts, obj_pts, num_pts=100000, bound=0.50, std=0.1):
    displace = obj_pts[np.random.randint(obj_pts.shape[0], size=num_pts)]
    x_min, y_min, z_min = obj_pts.min(axis=0)
    x_max, y_max, z_max = obj_pts.max(axis=0)
    a, b = -bound, bound
    xs = scipy.stats.truncnorm.rvs(
        *get_trunc_ab_range(x_min, x_max, std, a, b), loc=0, scale=std, size=num_pts
    )
    ys = scipy.stats.truncnorm.rvs(
        *get_trunc_ab_range(y_min, y_max, std, a, b), loc=0, scale=std, size=num_pts
    )
    zs = scipy.stats.truncnorm.rvs(
        *get_trunc_ab_range(z_min, z_max, std, a, b), loc=0, scale=std, size=num_pts
    )
    sampled_pts = np.array([xs, ys, zs]).T + displace

    x_nn = NearestNeighbors(
        n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric="l2"
    ).fit(scene_pts)
    dist, ind = x_nn.kneighbors(sampled_pts)
    dist = dist.squeeze()
    ind = ind.squeeze()

    occ = dist < 0.01
    pt_class = ind[occ != 0]

    return sampled_pts, occ, pt_class


def transform_points(pointcloud, from_range, to_range):
    if len(pointcloud.shape) == 1:
        pointcloud = pointcloud.reshape([1, -1])

    if pointcloud.shape[1] == 6:
        xyz = pointcloud[:, :3]
        rgb = pointcloud[:, 3:]
    else:
        xyz = pointcloud
        rgb = None

    from_center = np.mean(from_range, axis=0)
    from_size = np.ptp(from_range, axis=0)
    to_center = np.mean(to_range, axis=0)
    to_size = np.ptp(to_range, axis=0)
    xyz = (xyz - from_center) / from_size * to_size + to_center
    if rgb is None:
        return xyz
    else:
        return np.concatenate([xyz, rgb], axis=-1)


def ptp_th(t, axis):
    return t.max(axis).values - t.min(axis).values


def transform_points_th(pointcloud, from_range, to_range):
    if len(pointcloud.shape) == 1:
        pointcloud = pointcloud.reshape([1, -1])

    if pointcloud.shape[1] == 6:
        xyz = pointcloud[:, :3]
        rgb = pointcloud[:, 3:]
    else:
        xyz = pointcloud
        rgb = None

    from_center = torch.mean(from_range, dim=0)
    from_size = ptp_th(from_range, axis=0)
    to_center = torch.mean(to_range, dim=0)
    to_size = ptp_th(to_range, axis=0)
    xyz = (xyz - from_center) / from_size * to_size + to_center

    if rgb is None:
        return xyz
    else:
        return torch.cat([xyz, rgb], dim=-1)


def pos_quat_to_T(pos_quat):
    T = np.eye(4)
    T[:3, 3] = pos_quat[:3]
    T[:3, :3] = quat2mat(pos_quat[3:])

    return T


def find_T_prim_to_keypts(env, state, hot_start=None, num_pts=5000):
    """
    WARNING: this function will mutate the env state
    PARTICLE LABEL MEANING:
        0 = left hand
        1 = right hand
        2 = shape
    """

    env.simulator.set_state(0, state)

    (
        hand_particles,
        hand_particles_sdf,
        hand_particle_labels,
    ) = env.simulator.sample_pts_inside_primitives(
        n_pts=num_pts, mode=env.cfg.SIMULATOR.mode, hot_start=hot_start
    )

    particle_primitive_inds = np.ones((len(hand_particles),), dtype=np.int32) * -1
    num_primitives = hand_particles_sdf.shape[-1]
    for prim_idx in range(num_primitives):
        sdf_vals = hand_particles_sdf[..., prim_idx]
        occ_mask = sdf_vals <= 0
        particle_primitive_inds[occ_mask] = prim_idx
    assert np.all(particle_primitive_inds != -1)

    frame_transforms = dict()
    for prim_idx in range(num_primitives):
        prim_pos_quat = state[4 + prim_idx]

        T_prim_geom = pos_quat_to_T(prim_pos_quat)

        prim_mask = particle_primitive_inds == prim_idx
        prim_p_pos = hand_particles[prim_mask]
        prim_p_labels = hand_particle_labels[prim_mask]

        T_keypts = np.eye(4)[np.newaxis].repeat(len(prim_p_pos), axis=0)
        T_keypts[:, :3, 3] = prim_p_pos

        T_prim_to_keypts = np.linalg.inv(T_prim_geom) @ T_keypts

        frame_transforms[prim_idx] = (T_prim_to_keypts, prim_p_labels)

    return frame_transforms


def state_to_hand_particles(
    state,
    frame_transforms,
    output_dtype=np.float32,
):
    hand_particles = []
    hand_particle_labels = []
    for prim_idx, (T_prim_to_keypts, prim_p_labels) in frame_transforms.items():
        prim_pos_quat = state[4 + prim_idx]

        T_prim_geom = pos_quat_to_T(prim_pos_quat)
        T_keypts = T_prim_geom @ T_prim_to_keypts
        pos_keypts = T_keypts[:, :3, 3]

        hand_particles.append(pos_keypts)
        hand_particle_labels.append(prim_p_labels)

    hand_particles = np.concatenate(hand_particles, axis=0).astype(output_dtype)
    hand_particle_labels = np.concatenate(hand_particle_labels, axis=0).astype(
        output_dtype
    )
    return hand_particles, hand_particle_labels


def state_to_scene_particles(
    env,
    state,
    frame_transforms,
    output_dtype=np.float32,
    shape_particle_mask=None,
):
    """
    WARNING: this function will mutate the env state
    PARTICLE LABEL MEANING:
        0 = left hand
        1 = right hand
        2 = shape
    """

    env.simulator.set_state(0, state)

    shape_particles = state[0]
    if shape_particle_mask is not None:
        shape_particles = shape_particles[shape_particle_mask]

    hand_particles, hand_particle_labels = state_to_hand_particles(
        state, frame_transforms, output_dtype
    )
    shape_particle_labels = np.array([[2.0]] * len(shape_particles))

    particle_labels = np.concatenate([shape_particle_labels, hand_particle_labels])

    total_particles = np.concatenate([shape_particles, hand_particles], axis=0)
    total_particles = np.append(total_particles, particle_labels, axis=1)

    if env.cfg.env_name == "wrap":
        # preventing out of bound error by shifting scene
        total_particles[..., 0] += 0.2

    return total_particles.astype(output_dtype)


def state_to_scene_particles_sample(
    env, state, hot_start=None, num_pts=5000, output_dtype=np.float32
):
    """
    WARNING: this function will mutate the env state
    PARTICLE LABEL MEANING:
        0 = left hand
        1 = right hand
        2 = shape
    """

    env.simulator.set_state(0, state)

    shape_particles = state[0]
    (
        hand_particles,
        _,
        hand_particle_labels,
    ) = env.simulator.sample_pts_inside_primitives(
        n_pts=num_pts, mode=env.cfg.SIMULATOR.mode, hot_start=hot_start
    )

    shape_particle_labels = np.array([[2.0]] * len(shape_particles))
    particle_labels = np.concatenate([shape_particle_labels, hand_particle_labels])

    total_particles = np.concatenate([shape_particles, hand_particles], axis=0)
    total_particles = np.append(total_particles, particle_labels, axis=1)

    return total_particles.astype(output_dtype)


def preproc_single_scene(env, state, frame_transforms):
    # WARNING: this function will mutate the env state

    env_name = env.cfg.env_name
    scene_obs = state_to_scene_particles(env, state, frame_transforms)
    scene_obs[:, :3] = transform_points(
        scene_obs[:, :3], PLAB_COORD_RANGE[env_name], IPLCT_COORD_RANGE
    )
    scene_obs[:, -1] = np.where(scene_obs[:, -1] == 2.0, 0.0, 1.0)

    return scene_obs


def get_shape_particle_mask(state, num_shape_particles):
    shape_p = state[0]
    if len(shape_p) == num_shape_particles:
        return None

    print(
        f"Received {len(shape_p)} particles, need {num_shape_particles}. "
        f"Getting shape particle mask!"
    )

    shape_p = torch.from_numpy(shape_p).unsqueeze(0)
    _, mask = sample_farthest_points(shape_p, K=num_shape_particles)
    mask = mask.squeeze(0)

    return mask
