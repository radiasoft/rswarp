import numpy as np
import os
import h5py as h5
import matplotlib.pyplot as plt


class FieldDiagnostic(object):
    """
        Common functionality for field diagnostic classes

        Parameters:
            solver: Must be a Multigrid3D object.
            top: Object representing Warp's top package.
            w3d: Object representing Warp's w3d package.
    """
    def __init__(self, solver, top, w3d, period=None):
        self.solver = solver
        self.top = top
        self.w3d = w3d
        self.period = period
        self.gridsize = [self.solver.nx, self.solver.ny, self.solver.nz]
        self.mesh = [self.solver.xmesh, self.solver.ymesh, self.solver.zmesh]
        self.geometryParameters = ''

        if self.solver.solvergeom == self.w3d.XYZgeom:
            self.dims = ['x', 'y', 'z']
            self.geometry = 'cartesian'
        elif self.solver.solvergeom == self.w3d.RZgeom:
            self.dims = ['r', 't', 'z']
            self.geometry = 'thetaMode'
            self.geometryParameters = 'm=0'
        else:
            raise Exception("No handler for geometry type %i" % self.solver.geomtype)

    # Is it really necesary to gather the mesh here, or can it be done only in __init__?
    def gathermesh(self):
        self.mesh = [self.solver.xmesh, self.solver.ymesh, self.solver.zmesh]
        if self.solver.solvergeom not in [self.w3d.XYZgeom, self.w3d.RZgeom]:
            raise Exception("No handler for geometry type %i" % self.solver.geomtype)

    def writeDataset(self, data, prefix, attrs={}):
        if len(data.shape) == 3:  # Scalar data on the mesh
            self.file[prefix] = data
        elif len(data.shape) == 4:  # Vector data on the mesh
            for i, v in enumerate(data):
                self.file['%s/%s' % (prefix, self.dims[i])] = v
                self.file['%s' % prefix].attrs['n%s' % self.dims[i]] = self.gridsize[i]
        else:
            raise Exception("Unknown data shape: %s" % data.shape)

        for k, v in attrs.items():
            self.file[prefix].attrs[k] = v

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
        for i, v in enumerate(self.mesh):
            self.file['/data/meshes/mesh/%s' % self.dims[i]] = v
        self.file['/data/meshes/mesh'].attrs['geometry'] = self.geometry
        self.file['/data/meshes/mesh'].attrs['geometryParameters'] = self.geometryParameters

        return True


class ElectrostaticFields(FieldDiagnostic):
    """
        Produce an HDF5 file with electric fields and potential.
        File tree:
        /data/meshes
            /mesh
                /x
                /y
                /z

            Note that the coordinates will be replaced as appropriate for different
            solver geometries (e.g. xyz -> rtz for RZgeom).

            /potential
                /electric - [xdim,ydim,zdim]
            /fields
                /electric - [fieldaxis,xdim,ydim,zdim]

    """

    def gatherfields(self):
        self.efield = self.solver.getselfe()

    def gatherpotential(self):
        self.phi = self.solver.getphi()

    def write(self, prefix='diags/fields/electric/efield'):
        if not super(ElectrostaticFields, self).write(prefix):
            return False

        self.gatherfields()
        self.gatherpotential()

        self.writeDataset(self.efield, prefix='/data/meshes/fields/electric')
        self.writeDataset(self.phi, prefix='/data/meshes/potential/electric')

        self.file.close()


class MagnetostaticFields(FieldDiagnostic):
    """
        Produce an HDF5 file with magnetic fields and vector potential.
        File tree:
        /data/meshes/
            /mesh
                /x
                /y
                /z

            Note that the coordinates will be replaced as appropriate for different
            solver geometries (e.g. xyz -> rtz for RZgeom).

            /potential
                /vector - [vectoraxis,xdim,ydim,zdim]
            /fields
                /magnetic - [fieldaxis,xdim,ydim,zdim]

    """

    def gatherfields(self):
        self.bfield = self.solver.getb()

    def gathervectorpotential(self):
        self.a = self.solver.geta()

    def write(self, prefix='diags/fields/magnetic/bfield'):
        if not super(MagnetostaticFields, self).write(prefix):
            return False

        self.gatherfields()
        self.gathervectorpotential()

        self.writeDataset(self.bfield, prefix='/data/meshes/fields/magnetic')
        self.writeDataset(self.a, prefix='/data/meshes/potential/vector')

        self.file.close()
