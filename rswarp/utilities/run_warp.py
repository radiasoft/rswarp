import subprocess
from file_utils import cleanupPrevious
import os


class RunWarp():
    def __init__(self,runfile,cores=1,
                 particle_path='diags/xzsolver/hdf5/',
                 field_path={'magnetic':'diags/fields/magnetic','electric':'diags/fields/electric'}):
        """

        Args:
            runfile (str): Path to Warp input file.
            cores (int): Number of cores. Defaults to 1. Can set more for parallel execution.
            particle_path: Default: 'diags/xzsolver/hdf5/'. Can be changed to new directory.
            field_path (str, dict, or list):
            Default: {'magnetic':'diags/fields/magnetic','electric':'diags/fields/electric'}
            The field_path can not currently be changed as the FieldDiagnostic class is hardcoded to write these
            directories.
        """

        self.runflag = True
        self.runfile = runfile
        self.cores = cores
        self.particle_path = particle_path
        self.field_path = field_path
        self._set_runflag()

    def clean(self):
        cleanupPrevious(self.particle_path, self.field_path)

    def _set_runflag(self):
        if os.path.isdir(self.particle_path):
            self.runflag = False
        elif isinstance(self.field_path, dict):
            for key in self.field_path:
                if os.path.isdir(self.field_path[key]):
                    self.runflag = False
        elif isinstance(self.field_path, list):
            for key in self.field_path:
                if os.path.isdir(key):
                    self.runflag = False
        elif isinstance(self.field_path, str):
            if os.path.isdir(self.field_path):
                self.runflag = False

    def runwarp(self):

        if self.runflag == True:
            if self.cores <= 1:
                self.clean()
                call = subprocess.call("python %s" % self.runfile, shell=True)
            elif self.cores > 1:
                self.clean()
                call = subprocess.call("mpiexec -n %s python %s" % (self.cores, self.runfile), shell=True)
        elif self.runflag != True:
            print "Output directory already exists. Please change directory path to run." \
                  "\n To override this set attribute runflag to True"
            call = 1

        if call == 0:
            print "Warp Run Completed"
            self.runflag = False
        elif call != 0:
            print "Warp Failed to Run"


