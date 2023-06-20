# cuda env instead of taichi env provide the api similar to taichi env and mpm simulators.
import os
import yaml
from tools import CN, Configurable, merge_inputs
# from plb import load
from .simulator import MPMSimulator
from .renderer import Renderer

FILEPATH = os.path.dirname(os.path.abspath(__file__))

class CudaEnv(Configurable):
    def __init__(
        self,
        cfg:CN=None,
        cfg_path='configs/plb_cuda.yml',
        SIMULATOR=None,
        PRIMITIVES:CN=None,
        RENDERER:CN=Renderer.get_default_config(),
        SHAPES=None,
    ):
        super().__init__()
        # if PRIMITIVES is None or len(PRIMITIVES) == 0:
        #     PRIMITIVES = load(os.path.join(FILEPATH, cfg_path)).PRIMITIVES
        n_bodies, kwargs = self.parse_tools(PRIMITIVES)
        self.simulator = MPMSimulator(n_bodies, cfg=SIMULATOR)
        self.simulator.init_bodies(**kwargs)
        self.renderer = Renderer(cfg=RENDERER)
        import numpy as np
        self.particle_colors = np.zeros(self.simulator.n_particles) + (((((255)<<8)+255)<<8)+255)
        self.simulator.states[0].x.upload(np.random.random(size=(self.simulator.n_particles, 3)) * 0.2 + np.array((0.4, 0.1, 0.4)))
    
    def default_tool_config(self):
        cfg = CN()
        cfg.shape = ''
        cfg.init_pos = (0.3, 0.3, 0.3)  # default color
        cfg.init_rot = (1., 0., 0., 0.)  # default color
        cfg.color = (0.3, 0.3, 0.3)  # default color
        cfg.lower_bound = (0., 0., 0.)  # default color
        cfg.upper_bound = (1., 1., 1.)  # default color
        cfg.friction = 0.9  # default color
        cfg.variations = None  # TODO: not support now
        cfg.mass = 1.
        cfg.stiffness = 0.

        action = cfg.action = CN()
        action.dim = 0  # in this case it can't move ...
        action.scale = ()
        return cfg


    def parse_tools(self, cfgs):
        tools = []

        for i in cfgs:
            if isinstance(i, CN):
                cfg = i
            else:
                cfg = CN(new_allowed=True)
                cfg = cfg._load_cfg_from_yaml_str(yaml.safe_dump(i))

            cfg = merge_inputs(self.default_tool_config(), **cfg)
            tools.append(cfg)

        self.action_dims = [0]
        outs = []

        types = []
        softness = []
        pos = []
        rot = []
        mu = []
        round = []
        args = []
        action_scales = []

        for i in tools:
            #primitive = eval(i.shape)(cfg=i, dim=dim, max_timesteps=max_timesteps, dtype=dtype)
            #self.primitives.append(primitive)
            #self.action_dims.append(self.action_dims[-1] + primitive.action_dim)
            if i['shape'] == 'Box':
                types.append(0)
                args.append([*i['size'], 0])
            elif i['shape'] == 'Capsule':
                types.append(1)
                args.append([*i['size'], 0, 0])

            action_scales.append(i['action']['scale'])
            softness.append(666.)
            mu.append(i['friction'])
            round.append(i['round'])
            pos.append(i.init_pos)
            rot.append(i.init_rot)

        return len(tools), {
            'types': types,
            'softness': softness,
            'mu': mu,
            'round': round,
            'args': args,
            'action_scales': action_scales,
            'pos': pos,
            'rot': rot
        }
