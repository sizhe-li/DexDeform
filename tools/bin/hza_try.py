#!/usr/bin/env python
# monitor, file edit, and test
# kubectl create -f examples/pvc-example.yaml
import subprocess
import time
import os

def main():
    template_loc = os.path.dirname(os.path.abspath(__file__))
    cmd = 'kubectl exec hza-try -it -- bash -c "cd /cephfs/hza/;/bin/bash"'
    output = os.system(cmd)
    print('output code', output)
    if output != 0 and output != 130:
        os.system("kubectl delete pod hza-try") #
        os.system(f"kubectl create -f {template_loc}/pod.yml")
        output = -1
        while output != 0 and output != 130:
            print('output code', output)
            output = os.system(cmd)
            time.sleep(1)

if __name__ == '__main__':
    main()