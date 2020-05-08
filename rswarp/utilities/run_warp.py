# -*- coding: utf-8 -*-
u"""run warp

:copyright: Copyright (c) 2020 RadiaSoft LLC.  All Rights Reserved.
:license: http://www.apache.org/licenses/LICENSE-2.0.html
"""
from __future__ import absolute_import, division, print_function
from pykern.pkcollections import PKDict
from pykern.pkdebug import pkdc, pkdlog, pkdp
from subprocess import Popen, PIPE
from rswarp.utilities.file_utils import cleanupPrevious
import os
import sys

# TODO Pipe stderror to a parameter when Warp fails
# TODO

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
        """

        self.runflag = False
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
        else:
            self.runflag = True

        if isinstance(self.field_path, dict):
            for key in self.field_path:
                if os.path.isdir(self.field_path[key]):
                    self.runflag = False
                else:
                    self.runflag = True
        elif isinstance(self.field_path, list):
            for key in self.field_path:
                if os.path.isdir(key):
                    self.runflag = False
                else:
                    self.runflag = True
        elif isinstance(self.field_path, str):
            if os.path.isdir(self.field_path):
                self.runflag = False
            else:
                self.runflag = True


    def runwarp(self):

        if self.runflag == False:
            self._set_runflag()

        stdout, stderr = False, False

        if self.runflag == True:
            if self.cores <= 1:
                self.clean()
                call = Popen("{} {}".format(sys.executable, self.runfile),
                             stderr=PIPE, shell=True)
                stdout, stderr = call.communicate()
            elif self.cores > 1:
                self.clean()
                call = Popen("mpiexec -n {} {} {}".format(self.cores, sys.executable, self.runfile),
                             stderr=PIPE, shell=True)
                stdout, stderr = call.communicate()
        elif self.runflag != True:
            print("Output directory already exists. Please change directory path to run."
                  "\n To override this set attribute runflag to True")
            return

        if call.returncode == 0:
            print("Warp Run Completed")
            self.runflag = False

            return
        elif call.returncode != 0:
            print("Warp Failed to Run with Traceback:")
            if stderr:
                print('')
                print(stderr)
            else:
                print("No Traceback Available")

            return
