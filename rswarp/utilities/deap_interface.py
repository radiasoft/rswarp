import os
import h5py as h5
import numpy as np
import paramiko
from time import sleep, ctime

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

        # Directory containing any output from simulation
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
        print 'Starting batch file: {} in directory {}'.format(job_name, path)

        # Check for path existance
        stdin, stdout, stderr = self.client.exec_command('ls {}'.format(path))
        out = stdout.read()
        err = stderr.read()
        if err:
            print 'Could not find directory: {}'.format(path)
            print err, out
            return err
        else:
            print "Contents of job directory:", out
            assert job_name in out, "Cannot find {}\n Run will not start".format(job_name)

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

    def check_job_status(self, output_directory=None):
        if not self.jobid:
            print "No job known"
            return -1
        if output_directory:
            self.output_directory = output_directory

        check_file = os.path.join(self.output_directory, 'COMPLETE')

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
            if '0' in out_file:
                # Complete
                "{}: Job {} complete".format(ctime(), self.jobid)
                return 0
            else:
                # Fatal error
                "Unknown fatal error"
                return -1
        elif err_file and out_job:
            # Job not complete but still active
            print "{}: Job active but not complete".format(ctime())
            return 1
        elif err_file and err_job:
            # Fatal error
            print "{}: Error on status and file".format(ctime())
            return -1

    def monitor_job(self, timer, remote_output_directory, local_directory):
        self.output_directory = remote_output_directory

        sleep(10 * 60)
        timer -= 10 * 60
        status = self.check_job_status()
        if status == 0:
            self.retrieve_fitness(local_directory)
            return 0
        elif status == -1:
            return -1

        interval = 300
        while timer > 0:
            sleep(interval)

            status = self.check_job_status()
            if status == 0:
                self.retrieve_fitness(local_directory)
                return 0
            elif status == -1:
                return -1

            timer -= interval

        return -1

    def retrieve_fitness(self, local_directory, match_string=None):
        # Make sure we have an SSH connection
        self.refresh_ssh_client()

        # Use existing client to run SFTP connection
        sftp_client = self.establish_sftp_client(self.client)

        # Make new directory to upload file to, if required
        # Move to new directory
        try:
            sftp_client.chdir(self.output_directory)
        except IOError as e:
            print("Failed to create directory")
            return e

        for fil in sftp_client.listdir():
            # If a match_string is given only retrieve files containing match_string
            if match_string:
                if match_string in fil:
                    pass
                else:
                    continue
            else:
                pass

            # Retrieve the next file
            try:
                sftp_client.get(fil, os.path.join(local_directory, fil))
            except IOError as e:
                print "File retrieval failed for: {}".format(fil)
                print "Error returned:\n {}".format(e)
                pass

        sftp_client.close()
        print "SFTP Connection Closed"


def create_runfiles(population, filename, batch_instructions=batch_instructions,
                    run_header=batch_header, run_command=batch_srun, run_tail=batch_tail):
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


def save_generation(filename, population, generation, labels=None):
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


def evaluate_fitness(JobRunner, remote_directory, local_directory, population, generation, time=120 * 60):
    # Create run_files separately
    # Start job separately

    JobRunner.monitor_job(time, remote_output_directory=remote_directory, local_directory=local_directory)

    # TODO: Evaluation is yet to be added