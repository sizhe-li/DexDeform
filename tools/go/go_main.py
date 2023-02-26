import argparse

from .grammar import program_parser, opt_parser


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("program", type=str, help="Program names, separted by ','. "
                                                  "For example: foo.sh,foo.py,\"foo.py --name\" ", nargs="?")

    parser.add_argument("-m", "--module", type=str, help="python modules..")

    parser.add_argument("-c", "--command", type=str, help="shell commands")

    parser.add_argument('--zip', help='zip instead of product', default=False, action="store_true")

    parser.add_argument("--go", action="store_true", help="start to execute multiple commands")
    parser.add_argument('--override', action="store_true", help='override even if path exists')

    parser.add_argument('-o', '--opts', help='Remaining args to perform hyper parameter search',
                        default=[], nargs=argparse.REMAINDER)
    parser.add_argument('--cfg', help='configurations', default=None)


    parser.add_argument('--mpi_size', default=None, type=int)
    return parser


def local_args(parser: argparse.ArgumentParser):
    # local search part
    parser.add_argument('--devices', help='which gpu to use', default=None, type=str)
    parser.add_argument('--n_proc', default=1, type=int, help='number of workers to run the file')

    # simple method to help collect outputs..
    parser.add_argument('--log', default=None, help='output a log for each execution')
    parser.add_argument('--silent', action="store_true", help='log the results in a silent mood..')


def nautilus_args(parser: argparse.ArgumentParser):
    parser.add_argument("--workspace", default=None, type=str)

    parser.add_argument("--template", default=None, type=str)
    parser.add_argument('--gpu', help='number of gpu for nautilus', type=int, default=1)
    parser.add_argument('--cpu', help='number of gpu for nautilus', type=int, default=2)
    parser.add_argument('--memory', help='number of gpu for nautilus', type=int, default=None)
    parser.add_argument('--memory_upper', help='number of gpu for nautilus', type=int, default=None)
    parser.add_argument("--username", default='hza', type=str, help="username in nautilus")
    parser.add_argument("--save_path", default='nautilus_tmp', type=str, help="path to store configs..")
    parser.add_argument("--git pull", default=None, type=str, help="Determine if we need to pull the github..")

    parser.add_argument("--multiple", default=1, type=int) # multiple to improve gpu utility


def main(go=None, nautilus=False):
    parser = get_parser()
    if not nautilus:
        local_args(parser)
    else:
        nautilus_args(parser)
    args, unknown = parser.parse_known_args()
    # programs = parse_commands(args.program)
    if args.command is not None:
        assert len(unknown) == 0, "There should be no unknown parameters to run a python module. Do you forget add -o?"
        programs = [i for i in args.command.split(',')]
        is_python = False
    else:
        assert args.program is not None or args.module is not None
        if args.module is not None:
            #assert len(unknown) == 0, "in this case there should no unknown parameters"
            programs = ["-m "+i for i in args.module.split(',')]
        elif args.program is not None:
            programs = program_parser(args.program)
        is_python = True
    unknown = ' '.join(unknown)
    programs = [i + ' ' + unknown for i in programs]
    programs = opt_parser(programs, args.opts, args.zip, python=is_python)

    if go is not None:
        args.go = go
    if not nautilus:
        from .local import go_local
        go_local(programs, args)
    else:
        from .remote import go_remote
        go_remote(programs, args)
