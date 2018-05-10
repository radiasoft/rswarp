import os
import h5py as h5
import numpy as np
import paramiko
import yaml
from time import sleep, ctime
from re import findall
from rswarp.run_files.tec.tec_utilities import write_parameter_file


class JobRunner(object):
    def __init__(self, server, username):
        self.server = server
        self.username = username

        # Directory containing batch file, Warp input file, and COMPLETE flag file
        # Set when job started by `self.project_directory`
        self.project_directory = []

        # Directory containing any output from simulation
        self.output_directory = []

        # SLURM ID for current job being executed
        self.jobid = []
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

    def upload_file(self, remote_directory, upload_file, debug=False):
        """
        Upload a file or series of files to a single, remote directory via sftp over an ssh client.
        Args:
            remote_directory: Directory path relative to server entry.
                              Will attempt to make directory if it is not found.
            upload_file: Relative path to file(s) to be uploaded to the remote directory.

        Returns:
            None
        """
        if type(remote_directory) != list:
            remote_directory = [remote_directory, ]
        if type(upload_file) != list:
            upload_file = [upload_file, ]
        # Make sure we have an SSH connection
        self.refresh_ssh_client()

        # Use existing client to run SFTP connection
        sftp_client = self.establish_sftp_client(self.client)
        home_directory = sftp_client.getcwd()
        for dir, ufile in zip(remote_directory, upload_file):
            # Make new directory to upload file to, if required
            # Move to new directory
            try:
                sftp_client.chdir(dir)
            except IOError:
                try:
                    sftp_client.mkdir(dir)
                except IOError as e:
                    print("Failed to create directory")
                    return e

                sftp_client.chdir(dir)

            # Guarantee upload_file is iterable
            try:
                upload_file[0]
            except IndexError:
                upload_file = [upload_file, ]

            sftp_client.put(ufile, os.path.split(ufile)[-1])
            # Back to home before next file uploaded
            sftp_client.chdir(home_directory)
            if debug:
                print "{} Uploaded".format(ufile)

        sftp_client.close()
        print "SFTP Connection Closed"

        return 0

    def start_job(self, path, job_name):
        if type(path) != list:
            path = [path, ]
        if type(job_name) != list:
            job_name = [job_name, ]
        # Make sure we have an SSH connection
        self.refresh_ssh_client()

        for p, j in zip(path, job_name):
            print 'Starting batch file: {} in directory {}'.format(j, p)

            # Check for path existence
            stdin, stdout, stderr = self.client.exec_command('ls {}'.format(p))
            out = stdout.read()
            err = stderr.read()
            if err:
                print 'Could not find directory: {}'.format(p)
                print err, out
                return err
            else:
                assert j in out, "Cannot find {}\n Run will not start".format(j)

            stdin, stdout, stderr = self.client.exec_command('cd {}; sbatch {}'.format(p, j))

            out = stdout.read()
            err = stderr.read()

            if err:
                return err
            elif out:
                # return JobID
                status = out.split()
                self.jobid.append(status[-1])

                # Set launch director
                self.project_directory.append(p)

        return self.jobid

    def check_job_status(self, output_directory=None, job_flag=None):
        if len(self.jobid) == 0:
            print "No job known"
            return -1
        if output_directory:
            self.output_directory = output_directory
        if job_flag:
            self.job_flag = job_flag

        self.refresh_ssh_client()

        status = []
        for d, j, f in zip(self.output_directory, self.jobid, job_flag):
            # Status file deposited one level up from diagnostic files
            check_file = os.path.join(os.path.split(d)[0], 'COMPLETE')
            print check_file
            # Make sure we have an SSH connection

            stdin, stdout_file, stderr_file = self.client.exec_command('cat {}'.format(check_file))
            out_file = stdout_file.read()
            err_file = stderr_file.read()
            print 'm1'
            stdin, stdout_job, stderr_job = self.client.exec_command('squeue --job {} -o %r'.format(j))
            out_job = stdout_job.read()
            err_job = stderr_job.read()
            print 'm2'
            if err_file and out_job:
                # Job not complete but still active
                print "{}: Job for run_id {} active but not complete".format(ctime(), f)
                status.append(1)
            elif err_file and err_job:
                # Fatal error
                print "{}: Error on status and file for run_id {}".format(ctime(), f)
                status.append(-1)
            elif out_file:
                matches = findall('\d+', out_file)
                if str(f) in matches:
                    # Complete
                    print "{}: Job {}, running run_id {}, complete ".format(ctime(), j, f)
                    status.append(0)
                else:
                    # Fatal error
                    print "Unknown fatal error for run_id {}".format(f)
                    status.append(-1)

        return status

    def monitor_job(self, timer, output_directory, local_directory, match_string=None, sleep_start=600):
        if type(output_directory) != list:
            self.output_directory = [output_directory, ]
        if type(local_directory) != list:
            self.local_directory = [local_directory, ]

        sleep(sleep_start)
        timer -= sleep_start
        status = self.check_job_status()
        for s, d in zip(status, self.output_directory):  # TODO: The retrieve_fitness method will need to be updated and use outputdirectory
            if s == 0:
                self.retrieve_fitness(local_directory, match_string=match_string)
                # remove entry from all lists
                # self.output_directory, self.job_flag, self.job_id
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
                print("Failed to retrieve from directory {}".format(output_directory))
                continue

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
            f.writelines(run_strings.format(gen=generation, id=i, **batch_format['batch_instructions']))
            f.write(run_tail.format(gen=generation, id=i, **batch_format['batch_instructions']))


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


def return_efficiency(generation, population, directory):
    """

    Args:
        generation: Int indicating generation number
        population: Iterable containing population individuals constructed by DEAP
        directory: Directory holding files to calculate efficiency from

    Returns:

    """
    efficiency = []
    files = os.listdir(directory)
    for i, individual in enumerate(population):
        if "efficiency_id{}-{}.h5".format(generation, i) not in files:
            efficiency.append(0.0)
            continue
        else:
            f = os.path.join(directory, "efficiency_id{}-{}.h5".format(generation, i))
            data = h5.File(f, 'r')

            penalty = 0
            total_power = 0.0
            for attr in ['P_ew', 'P_r', 'P_ec', 'P_load', 'P_gate']:
                total_power += abs(data['efficiency'].attrs[attr])
                if data['efficiency'].attrs[attr] < 0.0:
                    penalty += data['efficiency'].attrs[attr]

            penalty = 1.0 + abs(penalty / total_power)
            # print penalty, data['efficiency'].attrs['eta'], abs(data['efficiency'].attrs['eta']) * -penalty, \
            #     data['efficiency'].attrs['P_load'] > data['efficiency'].attrs['P_gate'], i
            efficiency.append(abs(data['efficiency'].attrs['eta']) * -penalty)
            data.close()

    return efficiency

