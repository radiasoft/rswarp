import os

import datetime
from dateutil.tz import tzlocal
import h5py as h5
import numpy as np
from warp import getselfe, getphi, getb, geta


class FieldDiagnostic(object):
    """
        Common functionality for field diagnostic classes

        Parameters:
            solver: A solver object containing fields to be output.
            top: The object representing Warp's top package.
            w3d: The object representing Warp's w3d package.
            comm_world: Object representing an MPI communicator.
            period (int): Sets the period in steps of data writeout by the diagnostic.
                Defaults to writeout on every step if not set.
            write_dir (str): Relative path to place data output of the diagnostic.
                Defaults to 'diags/fields/electric' for electric fields/potentials, and 'diags/fields/magnetic'
                for magnetic fields/vector potentials if not set.
    """
    def __init__(self, solver, top, w3d, comm_world, period=None, write_dir=None):
        self.solver = solver
        self.top = top
        self.w3d = w3d
        self.comm_world = comm_world
        if self.comm_world is None:
            self.lparallel = 0
        else:
            self.lparallel = comm_world.Get_size()

        self.period = period
        if not write_dir:
            self.write_dir = None
        else:
            self.write_dir = write_dir

        self.geometryParameters = ''

        if self.solver.solvergeom == self.w3d.XYZgeom:
            self.geometry = 'cartesian'
            self.dims = ['x', 'y', 'z']
            self.gridsize = [self.solver.nx + 1, self.solver.ny + 1, self.solver.nz + 1]
            self.gridSpacing = [self.solver.dx, self.solver.dy, self.solver.dz]
            self.gridGlobalOffset = [self.solver.xmmin, self.solver.ymmin, self.solver.zmmin]
            self.mesh = [self.solver.xmesh, self.solver.ymesh, self.solver.zmesh]
        elif self.solver.solvergeom == self.w3d.XZgeom:
            self.geometry = 'cartesian2D'
            self.dims = ['x', 'y', 'z']
            self.gridsize = [self.solver.nx + 1, self.solver.nz + 1]
            self.gridSpacing = [self.solver.dx, self.solver.dz]
            self.gridGlobalOffset = [self.solver.xmmin, self.solver.zmmin]
            self.mesh = [self.solver.xmesh, self.solver.zmesh]
        elif self.solver.solvergeom == self.w3d.RZgeom:
            self.geometry = 'thetaMode'
            self.geometryParameters = 'm=0'
            self.dims = ['r', 't', 'z']
            self.gridsize = [self.solver.nx + 1, self.solver.nz + 1]
            self.gridSpacing = [self.solver.dx, self.solver.dz]
            self.gridGlobalOffset = [self.solver.xmmin, self.solver.zmmin]
            self.mesh = [self.solver.xmesh, self.solver.zmesh]
        else:
            raise Exception("No handler for geometry type %i" % self.solver.solvergeom)

    def write(self, write_dir=None):
        if self.period and self.top.it % self.period != 0:
            return False
        if write_dir is None:
            write_dir = self.write_dir

        if not os.path.lexists(write_dir):
            if self.lparallel == 0 or self.comm_world.rank == 0:
                os.makedirs(write_dir)

        step = str(self.top.it)
        filename = '%s/data%s.h5' % (write_dir, step.zfill(5))

        if self.lparallel == 0 or self.comm_world.rank == 0:
            f = h5.File(filename, 'w')

            # for i, v in enumerate(self.mesh):
            #     f['/data/meshes/mesh/%s' % self.dims[i]] = v
            # f['/data/meshes/mesh'].attrs['geometry'] = self.geometry
            # f['/data/meshes/mesh'].attrs['geometryParameters'] = self.geometryParameters

            # from warp.data_dumping.openpmd_diag.generic_diag
            # This header information is from https://github.com/openPMD/openPMD-standard/blob/latest/STANDARD.md#hierarchy-of-the-data-file
            f.attrs["openPMD"] = np.string_("1.0.0")
            f.attrs["openPMDextension"] = np.uint32(1)
            f.attrs["software"] = np.string_("warp")
            f.attrs["softwareVersion"] = np.string_("4")
            f.attrs["date"] = np.string_(
                datetime.datetime.now(tzlocal()).strftime('%Y-%m-%d %H:%M:%S %z'))
            f.attrs["meshesPath"] = np.string_("meshes/")
            f.attrs["particlesPath"] = np.string_("particles/")
            # Setup the basePath
            f.attrs["basePath"] = np.string_("/data/%T/")
            base_path = "/data/%d/" % self.top.it
            bp = f.require_group(base_path)

            # https://github.com/openPMD/openPMD-standard/blob/latest/STANDARD.md#required-attributes-for-the-basepath
            bp.attrs["time"] = self.top.time
            bp.attrs["dt"] = self.top.dt
            bp.attrs["timeUnitSI"] = 1.

            # https://github.com/openPMD/openPMD-standard/blob/latest/STANDARD.md#iterations-and-time-series
            f.attrs["iterationEncoding"] = np.string_("fileBased")
            f.attrs["iterationFormat"] =  np.string_("%s%%T.h5" % write_dir)

            self.basePath = base_path
            self.meshPath = f.attrs["meshesPath"]
            self.particlesPath = f.attrs["particlesPath"]
            self.file = f

        return True

    def writeDataset(self, data, prefix, attrs={}):
        # print "Shape in writeDataset", self.efield.shape
        if len(data.shape) == len(self.dims) or (self.geometry == 'cartesian2D' and len(data.shape) == len(self.dims) - 1):  # Scalar data on the mesh
            self.file[prefix] = data
            field = self.file[prefix]
            field.attrs['position'] = [0.0]*len(self.dims)  # Report scalar as on the mesh elements
            field.attrs['unitSI'] = 1.0
        elif len(data.shape) == len(self.dims) + 1 or (self.geometry == 'cartesian2D' and len(data.shape) == len(self.dims)):  # Vector data on the mesh

            if self.geometry == 'thetaMode':
                data = data.swapaxes(1, 2)  # For thetaMode, components stored in order of m,r,z
            for i, v in enumerate(data):
                self.file['%s/%s' % (prefix, self.dims[i])] = v
                coord = self.file['%s/%s' % (prefix, self.dims[i])]
                coord.attrs['position'] = [0.0]*len(self.dims)  # Report field as on the mesh elements
                coord.attrs['unitSI'] = 1.0

                field = self.file[prefix]
                # field.attrs['n%s' % self.dims[i]] = self.gridsize[i]
        else:
            raise Exception("Unknown data shape: %s" % repr(data.shape))

        field.attrs['geometry'] = self.geometry
        field.attrs['geometryParameters'] = self.geometryParameters
        field.attrs['dataOrder'] = 'C'  # C-like order
        field.attrs['axisLabels'] = self.dims
        field.attrs['gridSpacing'] = self.gridSpacing
        field.attrs['gridGlobalOffset'] = self.gridGlobalOffset
        field.attrs['gridUnitSI'] = 1.0
        field.attrs['unitSI'] = 1.0

        for k, v in attrs.items():
            self.file[prefix].attrs[k] = v


class ElectrostaticFields(FieldDiagnostic):
    """
        Test
        Produce an HDF5 file with electric fields and potential .
        File tree:
        /data/meshes
            /mesh
                /x
                /y
                /z

            Note that the coordinates will be replaced as appropriate for different
            solver geometries (e.g. xyz -> rtz for RZgeom).

            /phi
            /E
                /x
                /y
                /z

    """

    def gatherfields(self):
        if self.lparallel == 1:
            self.efield = self.solver.getselfe()
        else:
            self.efield = []
            for dim in ['x', 'y', 'z']:
                self.efield.append(getselfe(comp=dim))

            self.efield = np.array(self.efield)

    def gatherpotential(self):
        if self.lparallel == 1:
            self.phi = self.solver.getphi()
        else:
            self.phi = getphi()

    def write(self):
        if not self.write_dir:
            write_dir = 'diags/fields/electric'
        else:
            write_dir = self.write_dir

        if not super(ElectrostaticFields, self).write(write_dir):
            return False

        self.gatherfields()
        self.gatherpotential()

        if self.solver.__class__.__name__ == 'MultiGrid2D':
            # Kludge to make 2D electrostatic solver compatible with thetaMode
            # output (which is currently the only relevant option)
            self.efield = self.efield[:, :, np.newaxis, :]

            # this is particularly awful, because there is no decomposition for
            # the potential, but it's the only way to shoehorn the data into
            # OpenPMD compliance right now.
            self.phi = self.phi[np.newaxis, :, :]
        if self.lparallel == 0 or self.comm_world.rank == 0:
            self.writeDataset(self.efield, prefix='%s%sE' % (self.basePath, self.meshPath))
            self.writeDataset(self.phi, prefix='%s%sphi' % (self.basePath, self.meshPath))
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

            /vector_potential
                /x
                /y
                /z

            /B
                /x
                /y
                /z

    """

    def gatherfields(self):
        if self.lparallel == 1:
            self.bfield = self.solver.getb()
        else:
            self.bfield = []
            for dim in ['x', 'y', 'z']:
                self.bfield.append(getb(comp=dim))

            self.bfield = np.array(self.bfield)

    def gathervectorpotential(self):
        if self.lparallel == 1:
            self.a = self.solver.geta()
        else:
            self.a = []
            for dim in ['x', 'y', 'z']:
                self.a.append(geta(comp=dim))

            self.a = np.array(self.a)

    def write(self):
        if not self.write_dir:
            write_dir = 'diags/fields/magnetic'
        else:
            write_dir = self.write_dir

        if not super(MagnetostaticFields, self).write(write_dir):
            return False

        self.gatherfields()
        self.gathervectorpotential()
        if self.lparallel == 0 or self.comm_world.rank == 0:
            self.writeDataset(self.bfield, prefix='%s%sB' % (self.basePath, self.meshPath))
            self.writeDataset(self.a, prefix='%s%svector_potential' % (self.basePath, self.meshPath))

        self.file.close()
