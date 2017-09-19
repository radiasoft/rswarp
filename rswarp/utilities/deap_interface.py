import os
import h5py as h5
import numpy as np
import paramiko
from time import sleep

batch_header = """#!/bin/bash -l
#SBATCH -p {queue} 
#SBATCH -N {nodes} 
#SBATCH -t {time}
#SBATCH -A m2783
#SBATCH -J {job}
#SBATCH -C {architecture}

export mydir="{base_directory}"

mkdir -p $mydir

cd $SLURM_SUBMIT_DIR

cp ./* $mydir/.
cd $mydir
"""

batch_srun = """srun -N 1 -n 32 -c 2 --cpu_bind=cores python-mpi {warp_file} {parameters} -p 1 1 32 &
"""

batch_tail = """wait
echo 0 >> COMPLETE"""

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


class JobRunner(object):

    def __init__(self, server, username):
        self.server = server
        self.username = username

        # Directory containing batch file, Warp input file, and COMPLETE flag file
        # Set when job started by `self.project_directory`
        self.project_directory = None

        #Directory containing any output from simulation
        self.output_directory = None

        # SLURM ID for current job being executed
        self.jobid = None
        # establish canonical client for instance use
        self.client = self.establish_ssh_client(self.server, self.username)
        # if needed sftp will be opened
        self.sftp_client = None

    @staticmethod
    def establish_ssh_client(server, username):
        try:
            client = paramiko.SSHClient()

            client.load_system_host_keys()
            client.connect(hostname=server, username=username)
        except IOError, e:
            print "Failed to connect to server on: {}@{}\n".format(username, server)
            return e

        return client

    @staticmethod
    def close_client(client):
        client.close()

    @staticmethod
    def establish_sftp_client(ssh_client):
        try:
            sftp_client = ssh_client.open_sftp()
            return sftp_client
        except IOError, e:
            print "Failed to connect to establish sftp connection.\n"
            return e

    def refresh_ssh_client(self):
        if not self.client.get_transport() or self.client.get_transport().is_active() != True:
            print "Reopening SSH Client"
            self.client = self.establish_ssh_client(self.server, self.username)

            return self.client
        else:
            print "SSH Client is live"
            return self.client

    def upload_batch_file(self, remote_directory, upload_file):
        # Make sure we have an SSH connection
        self.refresh_ssh_client()

        # Use existing client to run SFTP connection
        sftp_client = self.establish_sftp_client(self.client)

        # Make new directory to upload file to, if required
        # Move to new directory
        try:
            sftp_client.chdir(remote_directory)
        except IOError:
            try:
                sftp_client.mkdir(remote_directory)
            except IOError as e:
                print("Failed to create directory")
                return e

            sftp_client.chdir(remote_directory)

        # Set directory
        sftp_client.put(upload_file, os.path.split(upload_file)[-1])
        print "{} Uploaded".format(upload_file)
        sftp_client.close()
        print "SFTP Connection Closed"

    def start_job(self, job):
        # Make sure we have an SSH connection
        self.refresh_ssh_client()

        path, job_name = os.path.split(job)
        print path

        if path:
            stdin, stdout, stderr = self.client.exec_command('ls')
            out = stdout.read()
            err = stderr.read()
            if err:
                print 'fail'
                return err
            else:
                print "Moved to {}".format(path)

        stdin, stdout, stderr = self.client.exec_command('cd {}; sbatch {}'.format(path, job_name))

        out = stdout.read()
        err = stderr.read()

        if err:
            return err
        elif out:
            # return JobID
            status = out.split()
            self.jobid = status[-1]

            # Set launch director
            self.project_directory = path
            return status[-1]

    def check_job_status(self):
        if not self.jobid:
            print "No job known"
            return -1

        check_file = os.path.join(self.project_directory, 'COMPLETE')

        # Make sure we have an SSH connection
        self.refresh_ssh_client()

        stdin, stdout_file, stderr_file = self.client.exec_command('cat {}'.format(check_file))
        out_file = stdout_file.read()
        err_file = stderr_file.read()

        stdin, stdout_job, stderr_job = self.client.exec_command('squeue --job {} -o %r'.format(self.jobid))
        out_job = stdout_job.read()
        err_job = stderr_job.read()

        if out_file:
            print "FOUND COMPLETE"
            if out_file == '0':
                # Complete
                "Job {} complete".format(self.jobid)
                return 0
            else:
                # Fatal error
                "Unknown fatal error"
                return -1
        elif err_file and out_job:
            # Job not complete but still active
            print "Job active but not complete"
            return 1
        elif err_file and err_job:
            # Fatal error
            print "Error on status and file"
            return -1

    def evaluate_fitness(self, timer):
        sleep(10 * 60)
        timer -= 10 * 60
        status = self.check_job_status()
        if status == 0:
            self.retrieve_fitness()
            return 0
        elif status == -1:
            return -1

        interval = 300
        while timer > 0:
            sleep(interval)

            status = self.check_job_status()
            if status == 0:
                self.retrieve_fitness()
                return 0
            elif status == -1:
                return -1

            timer -= interval

        return -1


    # def retrieve_fitness(self, server, username,
    #                      project_directory, local_directory):
    #     """
    #     Retrieve all files from given folder. uses Paramiko's `load_system_host_keys` to validate ssh/sftp connection.
    #     Will attempt to download all files in `project_directory` folder.
    #     Args:
    #         server: Name of remote serve for ssh connection.
    #         username: Username for connection to remote server.
    #         project_directory: Path to folder to download files from.
    #         local_directory: Path on local machine to download files to.
    #
    #     Returns:
    #         0
    #     """
    #     try:
    #         with paramiko.SSHClient() as client:
    #             client.load_system_host_keys()
    #             client.connect(hostname=server, username=username)
    #             sftp_client = client.open_sftp()
    #             sftp_client.chdir(project_directory)
    #             for fil in sftp_client.listdir():
    #                 try:
    #                     sftp_client.get(fil, os.path.join(local_directory, fil))
    #                 except IOError, e:
    #                     print "File retrieval failed with:\n {}".format(e)
    #                     pass
    #             saved_e = 0
    #     except IOError, e:
    #         print "Retrieval Failure"
    #         saved_e = e
    #
    #     return saved_e

def create_runfiles(population, filename=None, batch_instructions=batch_instructions,
                    run_header=batch_header, run_command=batch_srun, run_tail=batch_tail):
    if not filename:
        filename = 'PLACEHOLDER.txt'
    run_strings = []
    for i, ind in enumerate(population):
        parameter_string = ''
        for val in ind:
            parameter_string += '{}'.format(val) + ' '
        parameter_string += '{}'.format(i) + ' '
        run_strings.append(run_command.format(warp_file=batch_instructions['warp_file'],
                                              parameters=parameter_string))

    with open(filename, 'w') as f1:
        f1.write(run_header.format(**batch_instructions))
        f1.writelines(run_strings)
        f1.write(run_tail)


def retrieve_fitness(server, username,
                     project_directory, local_directory):
    """
    Retrieve all files from given folder. uses Paramiko's `load_system_host_keys` to validate ssh/sftp connection.
    Will attempt to download all files in `project_directory` folder.
    Args:
        server: Name of remote serve for ssh connection.
        username: Username for connection to remote server.
        project_directory: Path to folder to download files from.
        local_directory: Path on local machine to download files to.

    Returns:
        0
    """
    try:
        with paramiko.SSHClient() as client:
            client.load_system_host_keys()
            client.connect(hostname=server, username=username)
            sftp_client = client.open_sftp()
            sftp_client.chdir(project_directory)
            for fil in sftp_client.listdir():
                try:
                    sftp_client.get(fil, os.path.join(local_directory, fil))
                except IOError, e:
                    print "File retrieval failed with:\n {}".format(e)
                    pass
            saved_e = 0
    except IOError, e:
        print "Retrieval Failure"
        saved_e = e

    return saved_e


def upload_batch_file(server, username,
                      project_directory, upload_file, upload_directory=None, start_job=True):
    try:
        with paramiko.SSHClient() as client:
            client.load_system_host_keys()
            client.connect(hostname=server, username=username)
            sftp_client = client.open_sftp()
            sftp_client.chdir(project_directory)

            # Make new directory to upload file to, if required
            # Move to new directory
            if upload_directory and (upload_directory not in sftp_client.listdir):
                sftp_client.mkdir(upload_directory)
                sftp_client.chdir(upload_directory)

            sftp_client.put(upload_file, os.path.split(upload_file)[-1])

            if start_job:
                stdin, stdout, sterr = client.exec_command('sbatch {}').format(upload_file)

                if sterr:
                    return sterr.read()
                elif stdout:
                    # return JobID
                    status = stdout.read().split()
                    return status[-1]
    except IOError, e:
        print "Retrieval Failure"
        saved_e = e

    return saved_e


def start_job(server, username, path, filename):
    try:
        with paramiko.SSHClient() as client:
            client.load_system_host_keys()
            client.connect(hostname=server, username=username)
            client.exec_command('cd {}'.format(path))
            stdin, stdout, sterr = client.exec_command('sbatch {}').format(filename)

            if sterr:
                return sterr.read()
            elif stdout:
                # return JobID
                status = stdout.read().split()
                return status[-1]

    except IOError, e:
        print "Connection Failure"
        return e


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

        if retrieval_completed:
            return 0
        else:
            return 1


def evalEfficiency(filename, population, generation):
    manage_retrieval(**remote_transfer_settings)


def save_data(filename, population, generation, labels=None):
    label_format = 'str'
    if not labels:
        labels = [i for i in range(len(population))]
        label_format = 'int'

    data_file = h5.File(filename, mode='a')
    data_file.attrs['generations'] = generation
    pop_group = data_file.create_group('/generation{}'.format(generation))
    for lb, ind in zip(labels, zip(*population)):
        if label_format == 'int':
            label = 'parameter{}'.format(lb)
        else:
            label = lb
        data_set = pop_group.create_dataset(label, data=ind)

    fitness_data = np.array([ind.fitness.getValues() for ind in population])
    data_set = pop_group.create_dataset('fitness', fitness_data)

    data_file.close()