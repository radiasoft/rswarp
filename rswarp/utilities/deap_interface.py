import os
import pysftp
from time import sleep

batch_header = """
#!/bin/bash -l
#SBATCH -p {queue} 
#SBATCH -N {nodes} 
#SBATCH -t {time}
#SBATCH -A m2783
#SBATCH -J {job}
#SBATCH -C {architecture}

export mydir="$SCRATCH/{base_directory}"

cd $SLURM_SUBMIT_DIR

cp ./* $mydir/.
cd $mydir
"""

batch_srun = """srun -N 1 -n 32 -c 2 --cpu_bind=cores python-mpi {filename} {parameters} -p 1 1 32 &
"""

batch_tail = """wait"""

batch_instructions = {
    'queue': 'debug',
    'nodes': 1,
    'time': '00:01:00',
    'job': 'eaTest',
    'architecture': 'Haswell',
    'base_directory': 'eaTest'
}

remote_transfer_settings = {
    'server': 'cori.nersc.gov',
    'username': 'hallcc',
    'private_key_path': '/Users/chall/.ssh/id_rsa',
    'private_key_phrase': 'blocktime',
    'project_directory': '/global/cscratch1/sd/hallcc/',
    'get_directory': 'new_test1/',
    'local_directory': 'new_dir/'
}

upload_settings = {
    'server': 'cori.nersc.gov',
    'username': 'hallcc',
    'private_key_path': '/Users/chall/.ssh/id_rsa',
    'private_key_phrase': 'TEST',
    'project_directory': '.',
    'upload_dir': 'upload_test/',
    'upload_file': 'new_dir/test_simulation.out'
}


def create_runfiles(population, filename=None, batch_instructions=batch_instructions,
                    run_header=batch_header, run_command=batch_srun, run_tail=batch_tail):
    if not filename:
        filename = 'PLACEHOLDER.txt'
    run_strings = []
    for ind in population:
        parameter_string = ''
        for val in ind:
            parameter_string += '{}'.format(val) + ' '
        run_strings.append(run_command.format(filename=filename, parameters=parameter_string))

    with open(filename, 'w') as f1:
        f1.write(run_header.format(**batch_instructions))
        f1.writelines(run_strings)
        f1.write(run_tail)


def retrieve_fitness(server, username, private_key_path, private_key_phrase,
                     project_directory, get_directory, local_directory):
    try:
        with pysftp.Connection(server, username=username,
                               private_key=private_key_path, private_key_pass=private_key_phrase) as sftp:
            with sftp.cd(project_directory):
                sftp.get_d(get_directory, local_directory)
            saved_e = 0
    except IOError, e:
        print "Retrieval Failure"
        saved_e = e

    return saved_e


def upload_batch_file(server, username, private_key_path, private_key_phrase,
                      project_directory, upload_dir, upload_file):
    try:
        with pysftp.Connection(server, username=username,
                               private_key=private_key_path, private_key_pass=private_key_phrase) as sftp:
            with sftp.cd(project_directory):
                if not sftp.isdir(upload_dir):
                    sftp.mkdir(upload_dir)
                with sftp.cd(upload_dir):
                    sftp.put(upload_file)
            saved_e = 0
    except IOError, e:
        print "Retrieval Failure"
        saved_e = e

    return saved_e


def manage_retrieval(remote_transfer_settings):
    # If local_directory exists remove all txt files that might hold old data, else make the directory
    if os.path.isdir(remote_transfer_settings['local_directory']):
        for fil in os.listdir(remote_transfer_settings['local_directory']):
            if fil.endswith('.txt'):
                os.remove(os.path.join(remote_transfer_settings['local_directory'], fil))
    else:
        os.mkdir(remote_transfer_settings['local_directory'])

    # Loop attempts to check on and retrieve output data from NERSC
    retrieval_completed = False
    while not retrieval_completed:
        # TODO: Change sleep value to 3000 in final version
        sleep(5)
        status = retrieve_fitness(**remote_transfer_settings)

        if status == 0:
            retrieval_completed = True
        elif isinstance(status, IOError):
            assert status[1] == 'No such file', "Unknown IOError on retrieval"

        else:
            raise Exception('Unkown error on retrieval: {}'.format(status))

        print "Retrieval Completed:", status == 0
        print status, type(status), status[0], status[1]

        retrieval_completed = True


def evalEfficiency(filename, population, generation):
    manage_retrieval(**remote_transfer_settings)