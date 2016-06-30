import numpy as np
import os
import h5py as h5
import matplotlib.pyplot as plt


class FieldDiagnostic(object):
    """
    Common functionality for field diagnostic classes
    """
    def __init__(self, solver, top, period=None):
        self.solver = solver
        self.top = top
        self.period = period

    def gathermesh(self):
        self.nx = self.solver.nx
        self.ny = self.solver.ny
        self.nz = self.solver.nz

        self.xmesh = self.solver.xmesh
        self.ymesh = self.solver.ymesh
        self.zmesh = self.solver.zmesh

    def write(self, prefix='field'):
        if self.period and self.top.it % self.period != 0:
            return False

        outdir = os.path.split(prefix)[0]
        if outdir is not '' and not os.path.lexists(outdir):
            os.makedirs(outdir)

        step = str(self.top.it)
        filename = '%s%s.h5' % (prefix, step.zfill(5))

        self.file = h5.File(filename, 'w')

        self.gathermesh()
        self.file.create_dataset('mesh/x', data=self.xmesh)
        self.file.create_dataset('mesh/y', data=self.ymesh)
        self.file.create_dataset('mesh/z', data=self.zmesh)

        self.file.attrs['nx'] = self.nx
        self.file.attrs['ny'] = self.ny
        self.file.attrs['nz'] = self.nz

        return True


class ElectrostaticFields(FieldDiagnostic):
    """
        Produce an HDF5 file with electric fields and potential.
        File tree:
        /mesh
            /x
            /y
            /z
        attributes: nx, ny, nz

        /potential
            /electric - [xdim,ydim,zdim]
        /fields
            /electric - [fieldaxis,xdim,ydim,zdim]

        Parameters:
            solver: Must be a Multigrid3D object.
            top: Object representing Warp's top package.
    """

    def gatherfields(self):
        self.efield = self.solver.getselfe()
        return self.efield

    def gatherpotential(self):
        self.phi = self.solver.getphi()

    def write(self, prefix='diags/fields/electric/efield'):
        if not super(ElectrostaticFields, self).write(prefix):
            return False

        self.gatherfields()
        self.gatherpotential()

        self.file.create_dataset('fields/electric', data=self.efield)
        self.file.create_dataset('potential/electric', data=self.phi)

        self.file.close()


class MagnetostaticFields(FieldDiagnostic):
    """
        Produce an HDF5 file with magnetic fields and vector potential.
        File tree:
        /mesh
            /x
            /y
            /z
        attributes: nx, ny, nz

        /potential
            /vector - [vectoraxis,xdim,ydim,zdim]
        /fields
            /magnetic - [fieldaxis,xdim,ydim,zdim]

        Parameters:
            solver: Must be a MagnetostaticMG object.
            top: Object representing Warp's top package.
    """

    def gatherfields(self):
        self.bfield = self.solver.getb()
        return self.bfield

    def gathervectorpotential(self):
        self.a = self.solver.geta()

    def write(self, prefix='diags/fields/magnetic/bfield'):
        if not super(MagnetostaticFields, self).write(prefix):
            return False
            
        self.gatherfields()
        self.gathervectorpotential()

        self.file.create_dataset('fields/magnetic', data=self.bfield)
        self.file.create_dataset('potential/vector', data=self.a)

        self.file.close()
