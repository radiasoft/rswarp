import os
import h5py as h5
import numpy as np
import paramiko
import yaml
from time import sleep, ctime
from rswarp.run_files.tec.tec_utilities import write_parameter_file


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

        # Check for path existence
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

    def monitor_job(self, timer, remote_output_directory, local_directory, match_string=None):
        self.output_directory = remote_output_directory

        sleep(10 * 60)
        timer -= 10 * 60
        status = self.check_job_status()
        if status == 0:
            self.retrieve_fitness(local_directory, match_string=match_string)
            return 0
        elif status == -1:
            return -1

        interval = 300
        while timer > 0:
            sleep(interval)

            status = self.check_job_status()
            if status == 0:
                self.retrieve_fitness(local_directory, match_string=match_string)
                return 0
            elif status == -1:
                return -1

            timer -= interval

        return -1

    def retrieve_fitness(self, local_directory, match_string=None):
        # TODO: stop grabbing folders

        if os.path.isdir(local_directory):
            pass
        else:
            try:
                os.makedirs(local_directory)
            except OSError as e:
                raise e

        # Make sure we have an SSH connection
        self.refresh_ssh_client()

        # Use existing client to run SFTP connection
        sftp_client = self.establish_sftp_client(self.client)

        for output_directory in self.output_directory:
            # Move to output directory
            try:
                sftp_client.chdir(output_directory)
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


def create_runfiles(generation, population, simulation_parameters, batch_format):
    """
    Create batch script for NERSC and associated YAML parameter file for each individual.
    Args:
        generation: Int the generation number for the given population
        population: List containing dicts that specificy parameter attributes
        simulation_parameters: Dictionary with additional parameters that must be supplied at the simulation start
        batch_format: YAML template file or appropriate dictionary

    Returns: None

    """
    directory = 'generation_{}'.format(generation)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Create .yaml files
    for i, individual in enumerate(population):
        individual = dict(individual, **simulation_parameters)
        individual['run_id'] = i

        filepath = os.path.join(directory, 'tec_design_{}-{}.yaml'.format(generation, individual['run_id']))
        write_parameter_file(individual, filepath)

    # create associated batch files
    if type(batch_format) == dict:
        pass
    elif type(batch_format) == str:
        batch_format = yaml.load(open(batch_format, 'r'))
    else:
        raise TypeError("batch_format must be a file name or dictionary")

    for i in range(len(population)):
        filename = batch_format['batch_instructions']['file_base_name'] + '_{}'.format(i)
        run_header = batch_format['batch_header']
        run_strings = batch_format['batch_srun']
        run_tail = batch_format['batch_tail']
        with open(os.path.join(directory, filename), 'w') as f:
            f.write(run_header.format(id=i, **batch_format['batch_instructions']))
            f.writelines(run_strings.format(id=i, **batch_format['batch_instructions']))
            f.write(run_tail)


def save_generation(filename, population, generation, labels=None, overwrite_generation=False):
    # If attributes are not labled then label with an id number
    label_format = 'str'
    if not labels:
        labels = [i for i in range(len(population))]
        label_format = 'int'

    data_file = h5.File(filename, mode='a')

    data_file.attrs['generations'] = generation

    # Check for existing generation and remove if overwrite_generation
    try:
        pop_group = data_file.create_group('/generation{}'.format(generation))
    except ValueError:
        if not overwrite_generation:
            raise ValueError("Generation exists. Enable `overwrite_generation` flag to replace.")
        else:
            pop_group = data_file['/generation{}'.format(generation)]
            for dset in pop_group.keys():
                pop_group.__delitem__(dset)

    for lb, ind in zip(labels, zip(*population)):
        if label_format == 'int':
            label = 'parameter{}'.format(lb)
        else:
            label = lb
        data_set = pop_group.create_dataset(label, data=ind)

    fitness_data = np.array([ind.fitness.getValues() for ind in population])
    data_set = pop_group.create_dataset('fitness', data=fitness_data)

    data_file.close()


    # def load_generation(filename, generation):
    #     gen = h5.File(filename, 'r')
    #
    #     try:
    #         attributes = len([key for key in gen['generation{}/'.format(generation)]])
    #     except KeyError:
    #         raise KeyError("Generation {} is not in {}".format(filename, generation))
    #
    #     individuals = [[attr for attr in ]]

