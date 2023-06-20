from tools import Configurable, CN, as_builder


def test():
    def default_optim(lr=0.0001):
        optim = CN()
        optim.method = 'adam'
        optim.lr = lr
        return optim

    @as_builder
    class Optimizer(Configurable):
        def __init__(self, actor, ob_space, cfg=None, tau=0.1, actor_optim=default_optim(0.0001)):
            super(Optimizer, self).__init__()
            self.tau = tau
            self.actor_optim = actor_optim
            self.ob_space = ob_space

    class TD3(Optimizer):
        NAME = 'td3'

        def __init__(self, actor, ob_space,
                     actor_optim=default_optim(0.0003),
                     critic_optim=default_optim(),
                     critic_net=None, cfg=None):
            super(TD3, self).__init__(actor, ob_space)
            self.critic_optim = critic_optim
            self.critic_net = critic_net

    print('Default of TD3')
    print(TD3.get_default_config())

    a = Optimizer.build(None, None, cfg="test.yaml", **{"critic_optim": {"lr": 0.1}, "critic_optim.method": "xx"},
                        tau=10.)
    b = Optimizer.build(None, None, cfg="test.yaml", **{"critic_optim": {"lr": 0.4}, "critic_optim.method": "xx"},
                        tau=10.)

    print(a)
    print('a.tau', a.tau)
    print('a.critic_optim', a.critic_optim)
    print(b)


if __name__ == "__main__":
    test()
