import multiprocessing
import os
import subprocess


def get_cmd(cmd, device, mpi_size=None):
    if mpi_size is not None:
        n_devices = len(mpi_size.split(','))
        cmd = f'CUDA_VISIBLE_DEVICES={mpi_size} MPI_SIZE={n_devices} {cmd}'
    elif device != '':
        cmd = f'CUDA_VISIBLE_DEVICES={device} {cmd}'
    return cmd


def execute(cmd, log_dir, silent):
    if log_dir is None:
        os.system(cmd)
        return
    if silent:
        with open(log_dir, 'w') as f:
            f.write(cmd + '\n')
            output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
            f.write(output)
    else:
        command = cmd
        with open(log_dir, 'w') as f:
            f.write(command + '\n')
            # https://stackoverflow.com/questions/25750468/displaying-subprocess-output-to-stdout-and-redirecting-it
            command = 'PYTHONUNBUFFERED=1 ' + command  # TODO: some very very very strange hack ...
            with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, bufsize=1, text=True) as proc:
                for line in proc.stdout:
                    print(line, end='')
                    f.write(line + '\n')


def inf_loop(a):
    while True:
        yield a


def worker(args):
    cmd, devices, log_dir, go, silent = args
    device = devices[(multiprocessing.current_process()._identity[0] - 1) % len(devices)]
    cmd = get_cmd(cmd, device, args.mpi_size)
    if go:
        print('run', cmd)
        execute(cmd, log_dir, silent)
    else:
        print(cmd)
    return None


def go_local(programs, args):
    n_proc, devices, log_dir, go = args.n_proc, args.devices, args.log, args.go

    if devices is None:
        devices = ['']
    else:
        devices = devices.split(',')

    if not args.override:
        todo = []
        for j in programs:
            opts = j.split(' ')
            p = None
            for idx in range(len(opts)):
                if opts[idx] == '--path' and idx + 1 < len(opts):
                    p = opts[idx+1]
                    break
            if p is not None:
                if not os.path.exists(p):
                    todo.append(j)
                else:
                    print(f"Remove as path exists! {p} in {j}")
            else:
                todo.append(j)
        programs = todo

    if len(programs) == 0:
        print("No program to run!!")
        return

    n_proc = max(n_proc, len(devices))
    n_proc = min(n_proc, len(programs))
    if len(programs) == 1:
        # if there is only one program, execute it immediately
        if args.go:
            execute(get_cmd(programs[0], devices[0], args.mpi_size), log_dir, args.silent)
        else:
            print(programs[0])
            print("please add --go to execute it")
    else:
        pool = multiprocessing.Pool(n_proc)
        devices = devices[:n_proc]
        if log_dir is not None and go:
            os.makedirs(log_dir, exist_ok=True)
            _log_dir = log_dir
            log_dir = [os.path.join(log_dir, str(idx + 1) + '.out') for idx, _ in enumerate(programs)]
            # TODO: find a better naming strategies .. for example, if __name__ exists.. consider coolname in runx
            with open(os.path.join(_log_dir, '.meta'), 'w') as f:
                for i in programs:
                    f.write(i + '\n')
        else:
            log_dir = inf_loop(log_dir)
        for i in pool.imap(worker, zip(programs, inf_loop(devices), log_dir, inf_loop(go), inf_loop(args.silent))):
            pass
