import os
import subprocess
from typing import Optional


def download_from_nautilus(filenames: str, nautilus_path: str, target_path: Optional[str]=None):
    assert nautilus_path.startswith("/cephfs"), "nautilus path must start with /cephfs"
    new_filenames = []
    if target_path is None:
        target_path = './'
    os.makedirs(target_path, exist_ok=True)
    for i in filenames.split(','):
        base = i.split('/')[-1]
        target = os.path.join(target_path, base)
        if not os.path.exists(target):
            cmd = f"kubectl cp hza-try:{os.path.join(nautilus_path, i)} {target}"
            print(cmd)
            subprocess.call(cmd.split(' '))
        else:
            print(f"{target} exists")
        new_filenames.append(target)
    return new_filenames
