import itertools


def program_parser(program: str):
    if ',' in program:
        program = program.split(',')
        return sum([program_parser(i) for i in program], [])
    if program.endswith('.sh'):
        outs = []
        with open(program, 'r') as f:
            for command in f.readlines():
                command = command.strip()
                idx = command.find('#')
                if idx != -1:
                    command = command[idx:]
                if len(command) > 0:
                    outs += program_parser(command)
        return outs
    else:
        def remove(program, key):
            if program.startswith(key):
                program = program[len(key):].strip()
            return program

        program = remove(remove(program, "python3"), "python")
        assert ".py" in program, f"It seems that you are not running a python program? Your command: {program}"
        return [program]


def parse(val):
    if "{" in val:
        return None
    if "@" in val:
        return eval(val[1:])
    return val.split(',')


def parse_scope(val, scope):
    if "{" not in val:
        return None
    #locals().update(**scope)
    if val[0] is '\'' or val[0] is '\"':
        val = val[1:-1]
    if val.startswith("@"): #TODO: a hack to avoid replace '.' to '_' to execute a python code..
        val = val[1:]
    else:
        val = val.replace(".", "_")
    print(locals())
    val = "f\"" + val + "\""
    out = eval(val, scope)
    assert ',' not in out, "We don't support undecided parameter lists for now."
    return out


# opts are organized into zip
def opt_parser(programs, opts, is_zip=False, python=True):
    # opts['programs'] = ;

    if is_zip and len(programs) == 1:
        c = programs[0]

        def f():
            while True:
                yield c

        programs = f()

    if isinstance(opts, list):
        for a in opts[::2]:
            assert a[:2] == '--', f"We only support --key, {a}"
        assert len(opts) % 2 == 0
        opts = {a[2:]: b for a, b in zip(opts[::2], opts[1::2])}

    decided = []
    decided_key = []
    undecied = []
    for key, val in opts.items():
        out = parse(val)
        if out is not None:
            decided.append(out)
            decided_key.append(key)
        else:
            undecied.append(key)
    iter_method = zip if is_zip else itertools.product

    pid = 0
    answer = []
    for vals in iter_method(programs, *decided):
        SCOPE = {"PID": pid, "PROG": vals[0].split(' ')[0].replace('.py', '')}

        if python:
            out = f'python3 {vals[0]}'
        else:
            out = vals[0]
        for k, v in zip(decided_key, vals[1:]):
            SCOPE[k.replace('.', '_')] = v
            out += f' --{k} {v}'

        for k in undecied:
            out += f' --{k} {parse_scope(opts[k], SCOPE)}'

        answer.append(out)
        pid += 1
    return answer
