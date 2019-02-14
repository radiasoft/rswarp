import os
import h5py as h5
import numpy as np
import paramiko
import yaml
from time import sleep, ctime
from re import findall
from itertools import izip_longest
from rswarp.run_files.tec.tec_utilities import write_parameter_file

# TODO: Put in a graceful stop if connection can't be made. Should record time.

class JobRunner(object):
    status_code = {'pending': 0,
                   'running': -1,
                   'completed': -2,
                   'timeout': -3,
                   'cancelled': -4,
                   'cancelled+': -4,
                   'failure': -5  # Not a NERSC Term. Internal designation for unknown error.
                   }

    def __init__(self, server, username, key_filename=None, array_tasks=0):
        self.server = server
        self.username = username
        self.key_filename = key_filename
        self.array_tasks = array_tasks

        # Remote Directory containing batch file, Warp input file, and COMPLETE flag file
        # Set when job started by `self.project_directory`
        self._project_directory = []

        # Remote directory containing any output from simulation
        self.output_directory = []

        # SLURM IDs for current job being executed
        self.job_ids = []
        self.job_status = []
        self._full_status = []

        # establish canonical client for instance use
        self.client = self.establish_ssh_client(self.server, self.username, key_filename)
        # if needed sftp will be opened
        self.sftp_client = None
        self.job_flag = None

    @property
    def project_directory(self):
        return self._project_directory
    @project_directory.setter
    def project_directory(self, directory):
        if type(directory) != list and type(directory) != tuple:
            self._project_directory = [directory, ]
        else:
            self._project_directory = directory

    @property
    def output_directory(self):
        return self._project_directory
    @output_directory.setter
    def output_directory(self, directory):
        if type(directory) != list and type(directory) != tuple:
            self._project_directory = [directory, ]
        else:
            self._project_directory = directory

    @staticmethod
    def establish_ssh_client(server, username, key_filename):
        try:
            client = paramiko.SSHClient()

            client.load_system_host_keys()
            client.connect(hostname=server, username=username, key_filename=key_filename)
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
            self.client = self.establish_ssh_client(self.server, self.username, self.key_filename)

            return self.client
        else:
            print "SSH Client is live"
            return self.client

    def upload_file(self, remote_directory, upload_files, debug=False):
        """
        Upload a file or series of files to a single, remote directory via sftp over an ssh client.
        If len(remote_directory) < len(upload_file) the last directory in the list will be copied to
        equalize the length.
        Args:
            remote_directory: Directory path relative to server entry.
                              Will attempt to make directory if it is not found.
            upload_files: Relative path to file(s) to be uploaded to the remote directory.

        Returns:
            None
        """

        self.project_directory = remote_directory
        if type(upload_files) != list and type(upload_files) != tuple:
            upload_files = [upload_files, ]

        # Make sure we have an SSH connection
        self.refresh_ssh_client()

        # Use existing client to run SFTP connection
        sftp_client = self.establish_sftp_client(self.client)
        home_directory = sftp_client.getcwd()
        for directory, ufile in izip_longest(self.project_directory, upload_files, fillvalue=self.project_directory[-1]):
            # Make new directory to upload file to, if required
            # Move to new directory
            try:
                sftp_client.chdir(directory)
            except IOError:
                try:
                    sftp_client.mkdir(directory)
                except IOError as e:
                    print("Failed to create directory")
                    return e

                sftp_client.chdir(directory)

            sftp_client.put(ufile, os.path.split(ufile)[-1])
            # Back to home before next file uploaded
            sftp_client.chdir(home_directory)
            if debug:
                print "{} Uploaded".format(ufile)

        sftp_client.close()
        print "SFTP Connection Closed"

        return 0

    def start_job(self, job_name):
        """
        Start job(s) on NERSC. Will issue sbatch command for each batch file name in `job_name`.
        Individual paths for each job name can be provided in path or just a single directory if they are all in the
        same location.

        If job batch file sets up an array job the only the single job_name need be provided. Runner will automatically
        monitor each array task if the array_task attribute is set.

        Args:
            job_name: (list or str) Name of each batch file to start.
            path: (list or str) Relative path(s) from home (or absolute path) to batch files.


        Returns:
            List of job id numbers for successfully started jobs

        """

        if type(job_name) != list:
            job_name = [job_name, ]

        # Make sure we have an SSH connection
        self.refresh_ssh_client()

        for p, j in izip_longest(self.project_directory, job_name, fillvalue=self.project_directory[-1]):
            print 'Starting batch file: {} in directory {}'.format(j, p)

            # Check for path existence
            stdin, stdout, stderr = self.client.exec_command('ls {}'.format(p))
            out = stdout.read()
            err = stderr.read()  # Harmless module load errors make using stderr unreliable

            assert j in out, "Cannot find {}\n Run will not start".format(j)

            stdin, stdout, stderr = self.client.exec_command('cd {}; sbatch {}'.format(p, j))

            out = stdout.read()
            err = stderr.read()

            if out:
                # return JobID
                status = out.split()
                self.job_ids.append(status[-1])
            elif err:
                print(err)
                return err

        if self.array_tasks:
            assert len(self.job_ids) == 1, "Array job must return only 1 ID number"
            for i in range(self.array_tasks):
                self.job_ids.append(str(self.job_ids[0] + '_' + str(i)))
            self.job_ids.pop(0)

        return self.job_ids

    def get_job_state(self, jobid, verbose=False):
        """
        Monitor a job on NERSC with ID of jobid.
        Job status codes can be found in `job_status` attribute.
        Args:
            jobid: (int) ID number of job provided by SLURM on NERSC

        Returns:
            (int, dict): Returns integer job status code from `job_status` and dictionary of job details.
        """
        sacct_data = {}

        self.refresh_ssh_client()
        stdin, stdout, stderr = self.client.exec_command('sacct -j {}'.format(jobid))
        stdout, stderr = stdout.read(), stderr.read()

        # Catch no job yet
        if len(stdout.splitlines()) < 3:
            if verbose:
                print('Job: {} not found'.format(jobid))
            return self.status_code['failure'], None

        labels = stdout.splitlines()[0].split()
        data = None
        for ln in stdout.splitlines():
            if ln.find('{} '.format(jobid)) == 0:
                data = ln.split()
                break
        # If data format is bad catch here
        if not data:
            print('sacct output not valid')
            return self.status_code['failure'], None

        for datum, label in zip(data, labels):
            sacct_data[label] = datum

        if verbose:
            print(sacct_data['State'].lower(), sacct_data)

        return self.status_code[sacct_data['State'].lower()], sacct_data

    def monitor_jobs(self, timer=24 * 60 * 60, sleep_start=1 * 5 * 60, interval=1 * 1 * 60, verbose=False):
        # if array, then monitor parent id since task ids are not created until runtime
        parent_status = []
        array_task = False
        if self.job_ids[0].find('_') > -1:
            array_task = True
            parent_status, _ = self.get_job_state(self.job_ids[0][:-2])
            parent_status = [parent_status]

        self.job_status = [0] * len(self.job_ids)
        self._full_status = [None] * len(self.job_ids)
        if verbose:
            print("Monitoring {} jobs".format(len(self.job_status)))
            print("Monitoring will begin in {} minutes and occur every {} minutes \nfor {} hours".format(
                sleep_start / 60., interval / 60., timer / 3600.))

        # Allow interval sleep call at start of while to prevent waiting when finished
        try:
            sleep(sleep_start - interval)
            timer -= sleep_start - interval
        except IOError:
            pass
        if verbose:
            print("Starting monitoring at {}".format(ctime()))

        while np.any(np.array(self.job_status + parent_status) > -2) and timer > 0:
            sleep(interval)
            timer -= interval
            if verbose:
                print("Checking job status at {}".format(ctime()))
            for i, jobid in enumerate(self.job_ids):
                status_code, full_status = self.get_job_state(jobid)
                self.job_status[i] = status_code
                self._full_status[i] = full_status

            # Update parent status, do not record in id lists
            if array_task:
                parent_status, _ = self.get_job_state(self.job_ids[0][:-2])
                parent_status = [parent_status]

            if verbose > 1:
                print("Job status:")
                for jobid, job in zip(self.job_ids, self.job_status):
                    print("{}: {}".format(jobid, job))





    def download_files(self, local_directory, match_string=None):
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
    runfile_list = []
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

    # handle formatting for array jobs
    if batch_format['batch_header'].find('#SBATCH --array=') != -1:
        batch_format['batch_instructions']['array'] = '0-{}'.format(len(population) - 1)
        array_job = True
        batch_files = 1
    else:
        array_job = False
        batch_files = len(population)

    for i in range(batch_files):
        if array_job:
            i = 'array'
        filename = batch_format['batch_instructions']['file_base_name'] + '_{}'.format(i)
        run_header = batch_format['batch_header']
        run_strings = batch_format['batch_srun']
        run_tail = batch_format['batch_tail']

        runfile_list.append(filename)
        with open(os.path.join(directory, filename), 'w') as f:
            f.write(run_header.format(id=i, **batch_format['batch_instructions']))

            if array_job:
                i = '$SLURM_ARRAY_TASK_ID'

            f.writelines(run_strings.format(gen=generation, id=i, **batch_format['batch_instructions']))
            f.write(run_tail.format(gen=generation, id=i, **batch_format['batch_instructions']))

    return runfile_list

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

