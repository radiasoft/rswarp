from subprocess import Popen
import numpy as np
from time import sleep

run_string = "{{ time rsmpi -n 20 -h {host} python2 scan_input.py {volts} {folder} >> {folder}/stdout.txt; }} 2> {folder}/time.txt"

server_list = {1: 0, 2: 0}# , 3: 0, 4: 0, 5: 0}

voltage = [-0.05, -0.03, 0.01, 0.03, 0.05]  # list(np.linspace(-1.6, 1.6, 11))

while voltage:
    for server in server_list:
        if server_list[server] == 0:
            # Server is free, start a job
            v = voltage.pop(0)
            folder_name = 'grid_voltage_scan_contact-scan/anode_voltage{}'.format(v)
            mkfolder = Popen('mkdir -p {}'.format(folder_name), shell=True)
            mkfolder.wait()
            server_list[server] = Popen(run_string.format(host=server, volts=v, folder=folder_name), shell=True)
            sleep(5)
        else:
            # Check if server has become free
            if server_list[server].poll() is not None:
                # Server is free, change status
                server_list[server] = 0
    sleep(60)