import os
import numpy as np
import copy
import os
import copy
import argparse

class exp:
    def __init__(self, cmd):
        cmd = cmd.replace('python3', 'go.py').replace('python', 'go.py')
        self.cmd = cmd
        self.opts = cmd.split(' ')

    def get(self, key):
        for idx, i in enumerate(self.opts):
            if i == '--' + key:
                return self.opts[idx+1]
        return None

    def __getattr__(self, key):
        tmp = self.get(key)
        if tmp is not None:
            return tmp
        raise AttributeError(f"key {key} not found in {self.opts}")

    def add(self, key, val=None):
        if isinstance(key, dict):
            e = self
            for i, v in key.items():
                e = e.add(i, v)
            return e
        if not key.startswith('-'):
            key = '--' + key
        if val is None:
            key, val = key.split(' ')
        new_opts = copy.deepcopy(self.opts)
        found = False
        for idx, i in enumerate(new_opts):
            if i == key:
                assert idx + 1 < len(new_opts)
                new_opts[idx+1] = str(val)
                found = True
                break
        if not found:
            new_opts += [key, str(val)]
        return exp(' '.join(new_opts))

    def remove(self, key):
        if isinstance(key, list):
            for i in key:
                e = e.remove(i)
            return e

        if not key.startswith('-'):
            key = '--' + key
        for idx, i in enumerate(self.opts):
            if i == key:
                return exp(' '.join(self.opts[:idx] + self.opts[idx+2:]))
        raise AttributeError(f"No {key} in {str(self)}")

    def append(self, key, val):
        val = self.get(key) + val
        return self.add(key, val)

    def __str__(self):
        return self.cmd

def get_args(keys):
    parser=argparse.ArgumentParser()
    assert 'all' not in keys, "you can't set an exp with a name all"
    parser.add_argument("exp", help=f"{keys+['all']}")
    parser.add_argument("--go", action="store_true", help="start to run the experiments")
    parser.add_argument("--override", action="store_true", help="override the results even if the results exists")

    parser.add_argument('--devices', help='which gpu to use', default=None, type=str)
    parser.add_argument('--n_proc', default=1, type=int, help='number of workers to run the file')

    parser.add_argument('--n_seed', default=0, type=int, help='number of seed; set zero to discount its effects..')
    parser.add_argument('--show', action='store_true')

    parser.add_argument('--nautilus', action='store_true')
    parser.add_argument('--cpu', default=2, type=int)
    parser.add_argument('--exp_name', default=None, type=str)

    parser.add_argument("--multiple", default=1, type=int) # multiple to improve gpu utility
    parser.add_argument('--args', nargs=argparse.REMAINDER, default=None)

    #
    args=parser.parse_args()
    return args

def run(seed_name='trainer.seed', **kwargs):
    all_exps = list(kwargs.keys())
    args = get_args(all_exps)
    if args.exp != 'all':
        exps = args.exp.split(',')
    else:
        exps = all_exps

    for i in exps:
        e = kwargs[i]
        if '-o' not in str(e):
            e = e.add('-o', '')
        if args.n_seed > 0:
            assert seed_name is not None
            if isinstance(e, str):
                e = exp(e)
            e = e.add('--path', e.path + '_{'+seed_name.replace('.', '_')+'}').add(f'--{seed_name}', f"@'range({args.n_seed})'")


        if args.args is not None:
            for a, b in zip(args.args[::2], args.args[1::2]):
                e = e.add(a, b)

        cmd = str(e)
        def insert(cmd, option):
            cmd = cmd.split(' ')
            cmd = ' '.join(cmd[:1] + [option] + cmd[1:])
            return cmd

        if args.nautilus:
            assert cmd[0:2] == 'go'

            cmd = 'remote'+cmd[2:]
            args.devices = None
            insert(cmd, '--gpu 1')
            insert(cmd, f'--cpu {args.cpu}')
            exp_name = args.exp_name
            if exp_name is None:
                exp_name = i.replace('_', '-')
            cmd = cmd + f' --exp_name {exp_name}'
            if seed_name is not None:
                cmd += '-{' + seed_name + '}'


        if args.go:
            cmd = insert(cmd, '--go')
        if args.override:
            cmd = insert(cmd, '--override')
        if args.multiple > 1:
            cmd = insert(cmd, f'--multiple {args.multiple}')

        if not args.nautilus:
            if args.devices is not None:
                cmd = insert(cmd, f'--devices {args.devices}')
            cmd = insert(cmd, f'--n_proc {args.n_proc}')

        if args.show:
            print(cmd)

        os.system('MKL_THREADING_LAYER=GNU ' + str(cmd))