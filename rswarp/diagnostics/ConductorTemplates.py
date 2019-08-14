import numpy as np
from warp.field_solvers.generateconductors import XPlane, YPlane, ZPlane, Box, Sphere
from warp import comm_world, toperror
from scipy.stats import gaussian_kde
from .parallel import gather_lost_particles


class Conductor(object):
    """
    Handles plotting of characteristics of different types of conductor objects in Warp.
    The `generate_faces` method of each subclass is used to create a series of points that may be interpreted by
    mayavi.mlab.mesh to create a 3D surface. The impact density distribution is provided by the `_color_mesh`
    method. Where an object must be constructed from more than 1 mesh call a subset of particles may be provided for
    each face. This should be found through a `get_particles` method for each subclass.
    I am not sure this is strictly necessary now that a 3D KDE is constructed (it used to be 2D surface) but it may
    be faster due to scaling of the KDE.
    """

    def __init__(self, top, w3d, conductor, interpolation='kde'):
        assert interpolation == 'kde', "Interpolation must be 'kde'"

        if not comm_world:
            self.lparallel = 0
        else:
            self.lparallel = comm_world.size

        self.interpolation = interpolation
        self.top = top
        self.w3d = w3d
        self.conductor = conductor
        self.numlost = top.npslost[0]
        self.pids = None
        self.center = None
        self.cbounds = None
        self.pids = self._get_pids()
        self.thresshold = 30  # how many particles need to be lost to generate a colormap
        self.points_2d = 1000  # Scatter points per line segment in 2d plots

        self.debug = False

        if w3d.solvergeom == w3d.XZgeom or w3d.solvergeom == w3d.RZgeom:
            self.axis = [0, 2, 0, 2]
            self.domain = [w3d.xmmin, w3d.zmmin, w3d.xmmax, w3d.zmmax]
            self.cell_size = [w3d.dx, w3d.dz]
            if self.pids:
                if self.lparallel < 2:
                    self.scraped_particles = np.array([self.top.xplost, self.top.zplost])
                else:
                    x, _, z = gather_lost_particles(self.top, comm_world)
                    self.scraped_particles = np.array([x, z])
            else:
                self.scraped_particles = np.array([]).reshape(0, 0)
        elif w3d.solvergeom == w3d.XYZgeom:
            self.axis = [0, 1, 2, 0, 1, 2]
            # TODO: check if w3d returns global values in parallel
            self.domain = [w3d.xmmin, w3d.ymmin, w3d.zmmin, w3d.xmmax, w3d.ymmax, w3d.zmmax]
            self.cell_size = [w3d.dx, w3d.dy, w3d.dz]
            if self.pids:
                if self. lparallel < 2:
                    self.scraped_particles = np.array([self.top.xplost, self.top.yplost, self.top.zplost])
                else:
                    x, y, z = gather_lost_particles(self.top, comm_world)
                    self.scraped_particles = np.array([x, y, z])
            else:
                self.scraped_particles = np.array([]).reshape(0, 0)

    def _get_pids(self):
        if self.lparallel < 2:
            try:
                scraped_particles = np.where(self.top.pidlost[:self.numlost, -1] == self.conductor.condid)[0]
            except toperror:
                # Will still allow plotting of surfaces with no colormap applied
                print("Warning! No lost particles found")
                return None
        else:
            conductor_ids = gather_cond_id(self.top)
            if conductor_ids.shape[0] == 0:
                print("Warning! No lost particles found")
                return None
            scraped_particles = np.where(conductor_ids[:self.numlost] == self.conductor.condid)[0]

        return scraped_particles

    def _color_mesh(self, mesh, particle_subset=None):
        # all particles lost on conductor
        if particle_subset is not None:
            pids = particle_subset
        else:
            pids = self.pids
        scraped_parts = self.scraped_particles[:, pids]

        if scraped_parts.shape[1] > self.thresshold:
            if self.interpolation == 'kde':
                # Bandwidth based on Scott's rule with empirically determinted factor of 1/3 to reduce smoothing
                bandwdith = scraped_parts.shape[1]**(-1. / (scraped_parts.shape[0] + 4)) / 3.0
                kernel = gaussian_kde(scraped_parts, bw_method=bandwdith)
                # Remove gaussian normalization factor
                s = kernel(mesh).T * kernel._norm_factor
        else:
            s = np.ones_like(mesh[0, :]) * -1.0

        return s


class BoxPlot(Conductor):
    def __init__(self, top, w3d, conductor, interpolation='kde'):
        super(BoxPlot, self).__init__(top, w3d, conductor, interpolation=interpolation)
        center = conductor.xcent, conductor.ycent, conductor.zcent
        size = conductor.xsize, conductor.ysize, conductor.zsize
        stride = (w3d.solvergeom == w3d.RZgeom or w3d.solvergeom == w3d.XZgeom) + 1
        xmin, xmax = center[0] - size[0] / 2., center[0] + size[0] / 2.
        ymin, ymax = center[1] - size[1] / 2., center[1] + size[1] / 2.
        zmin, zmax = center[2] - size[2] / 2., center[2] + size[2] / 2.
        mine, maxe = [xmin, ymin, zmin][::stride], \
                     [xmax, ymax, zmax][::stride]

        self.cbounds = np.hstack([mine, maxe])

    def get_particles(self):
        scraped_parts = self.scraped_particles.T[self.pids, :]
        for bound, axis, cell_size in zip(self.cbounds, self.axis, 2 * self.cell_size):
            if self.pids:
                pof = np.where(np.abs(scraped_parts[:, axis] - bound) <= cell_size)
                yield pof[0]
            else:
                yield None

    def generate_faces_2d(self):
        if self.debug:
            total_pid = 0
        for i, axis in enumerate(self.axis):
            a, b = np.linspace(self.cbounds[1 - (i % 2)], self.cbounds[3 - (i % 2)], self.points_2d), \
                   np.linspace(self.cbounds[i], self.cbounds[i], self.points_2d)
            z, x = [a, b][axis // 2], [b, a][axis // 2]

            if self.debug:
                print("BOX:")
                print("Bounds x:", np.min(x), np.max(x))
                print("Bounds z:", np.min(z), np.max(z))
                print("Particle count:", self.pids.size)
                print()
                total_pid += self.pids.size

            s = self._color_mesh(mesh=np.vstack([x.ravel(), z.ravel()]), particle_subset=None)

            yield x, z, s

    def generate_faces_3d(self):
        if self.debug:
            total_pid = 0
        for bound, axis, pids in zip(self.cbounds, self.axis, self.get_particles()):

            xn = 1 + 5 * np.abs(self.cbounds[(axis + 1) % 3] - self.cbounds[(axis + 1) % 3 + 3]) \
                // self.cell_size[(axis + 1) % 3]
            yn = 1 + 5 * np.abs(self.cbounds[(axis + 2) % 3] - self.cbounds[(axis + 2) % 3 + 3]) \
                // self.cell_size[(axis + 2) % 3]

            x, y = np.meshgrid(np.linspace(self.cbounds[(axis + 1) % 3],
                                           self.cbounds[(axis + 1) % 3 + 3], xn),
                               np.linspace(self.cbounds[(axis + 2) % 3],
                                           self.cbounds[(axis + 2) % 3 + 3], yn))
            z = np.ones_like(x) * bound

            x, y, z = [x, y, z][(2 * axis + 2) % 3], \
                      [x, y, z][(2 * axis + 3) % 3], \
                      [x, y, z][(2 * axis + 1) % 3]
            if self.debug:
                print("BOX:")
                print("Bounds x:", np.min(x), np.max(x))
                print("Bounds y:", np.min(y), np.max(y))
                print("Bounds z:", np.min(z), np.max(z))
                print("Particle count:", pids.size)
                print()
                total_pid += pids.size
            s = self._color_mesh(mesh=np.vstack([x.ravel(), y.ravel(), z.ravel()]), particle_subset=pids).reshape(x.shape)
            if self.debug:
                print("print total:", self.pids.size, total_pid)

            yield x, y, z, s


class Plane(Conductor):
    def generate_faces_2d(self):
        for i in self.axis:
            print(1 - (i % 2), 3 - (i % 2))
            a, b = np.linspace(self.cbounds[1 - i // 2], self.cbounds[3 - i // 2], self.points_2d), \
                   np.linspace(self.center, self.center, self.points_2d)
            z, x = [a, b][i // 2], [b, a][i // 2]

            s = self._color_mesh(mesh=np.vstack([x.ravel(), z.ravel()]), particle_subset=None)
            if self.debug:
                print("PLANE:")
                print("Bounds x:", np.min(x), np.max(x))
                print("Bounds z:", np.min(z), np.max(z))
                print("Particle count:", self.pids.size)
                print()

            yield x, z, s

    def generate_faces_3d(self):
        for bound, axis in zip(self.cbounds, self.axis):
            xn = 1 + 5 * np.abs(self.cbounds[(axis + 1) % 3] - self.cbounds[(axis + 1) % 3 + 3]) \
                 // self.cell_size[(axis + 1) % 3]
            yn = 1 + 5 * np.abs(self.cbounds[(axis + 2) % 3] - self.cbounds[(axis + 2) % 3 + 3]) \
                 // self.cell_size[(axis + 2) % 3]

            x, y = np.meshgrid(np.linspace(self.cbounds[(axis + 1) % 3],
                                           self.cbounds[(axis + 1) % 3 + 3], xn),
                               np.linspace(self.cbounds[(axis + 2) % 3],
                                           self.cbounds[(axis + 2) % 3 + 3], yn))
            z = np.ones_like(x) * self.center
            if self.debug:
                print("cells for plane", x.shape, y.shape)
            s = self._color_mesh(mesh=np.vstack([x.ravel(), y.ravel(), z.ravel()]), particle_subset=None).reshape(x.shape)
            if self.debug:
                print("PLANE:")
                print("Bounds x:", np.min(x), np.max(x))
                print("Bounds y:", np.min(y), np.max(y))
                print("Bounds z:", np.min(z), np.max(z))
                print("Particle count:", self.pids.size)
                print()

            yield x, y, z, s


class XPlanePlot(Plane):
    def __init__(self, top, w3d, conductor, interpolation='kde'):
        super(XPlanePlot, self).__init__(top, w3d, conductor, interpolation=interpolation)
        self.axis = [0, ]
        self.center = conductor.xcent
        self.cbounds = self.domain


class YPlanePlot(Plane):
    def __init__(self, top, w3d, conductor, interpolation='kde'):
        super(YPlanePlot, self).__init__(top, w3d, conductor, interpolation=interpolation)
        self.axis = [1, ]
        self.center = conductor.ycent
        self.cbounds = self.domain


class ZPlanePlot(Plane):
    def __init__(self, top, w3d, conductor, interpolation='kde'):
        super(ZPlanePlot, self).__init__(top, w3d, conductor, interpolation=interpolation)
        self.axis = [2, ]
        self.center = conductor.zcent
        self.cbounds = self.domain


class SpherePlot(Conductor):
    def __init__(self, top, w3d, conductor, interpolation='kde'):
        super(SpherePlot, self).__init__(top, w3d, conductor, interpolation=interpolation)
        self.center = [conductor.xcent, conductor.ycent, conductor.zcent]
        self.radius = conductor.radius

    def generate_faces_3d(self):
        if self.debug:
            print("Sphere center:", self.center)
            print("Particle Count:", self.pids.size)
            print()
        # TODO: Sphere has a fixed number of points right now. Everything else scales with the solver mesh spacing.
        dphi = np.linspace(0, 2 * np.pi, 250)
        dtheta = np.linspace(-np.pi, np.pi, 250)
        phi, theta = np.meshgrid(dphi, dtheta)
        x = self.radius * np.cos(phi) * np.sin(theta) + self.center[0]
        y = self.radius * np.sin(phi) * np.sin(theta) + self.center[1]
        z = self.radius * np.cos(theta) + self.center[2]

        s = self._color_mesh(mesh=np.vstack([x.ravel(), y.ravel(), z.ravel()]), particle_subset=None).reshape(x.shape)

        yield x, y, z, s


class UnstructuredPlot(Conductor):
    """
    Can represent any convex shaped conductor in Warp.
    """
    def __init__(self, top, w3d, conductor, interpolation='kde', use_aura=False):
        super(UnstructuredPlot, self).__init__(top, w3d, conductor, interpolation=interpolation)
        self.use_aura = use_aura

    def _isinside(self):
        if self.debug:
            print("Starting Unstructured Construction")
        x = np.linspace(self.w3d.xmmin, self.w3d.xmmax, self.w3d.nx)
        y = np.linspace(self.w3d.ymmin, self.w3d.ymmax, self.w3d.ny)
        z = np.linspace(self.w3d.zmmin, self.w3d.zmmax, self.w3d.nz)
        aura = None
        if self.use_aura:
            dx = abs(self.w3d.xmmax - self.w3d.xmmin) / self.w3d.nx
            dy = abs(self.w3d.ymmax - self.w3d.ymmin) / self.w3d.ny
            dz = abs(self.w3d.zmmax - self.w3d.zmmin) / self.w3d.nz
            # add an "aura" the size of the smallest cell dimension
            aura = np.min([dx, dy, dz]) / 1.
            if self.debug:
                print('inside aura {}', aura)

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        isin = self.conductor.isinside(X.ravel(), Y.ravel(), Z.ravel(), aura=aura)
        dat_isin = 1 - isin.isinside.reshape(X.shape)
        transitions = np.ones(X.shape + (3,))

        if self.w3d.nx == self.w3d.ny and self.w3d.ny == self.w3d.nz:
            if self.debug:
                print("single loop set")
            for ii in range(self.w3d.nx):
                for jj in range(self.w3d.nz):
                    transitions[:, ii, jj, 0] = self._find_edge(dat_isin[:, ii, jj])
                    transitions[ii, :, jj, 1] = self._find_edge(dat_isin[ii, :, jj])
                    transitions[ii, jj, :, 2] = self._find_edge(dat_isin[ii, jj, :])
        else:
            if self.debug:
                print("triple loop set")
            for ii in range(self.w3d.nx):
                for jj in range(self.w3d.ny):
                    transitions[ii, jj, :, 2] = self._find_edge(dat_isin[ii, jj, :])
            if self.debug:
                print("first loop")
            for ii in range(self.w3d.ny):
                for jj in range(self.w3d.nz):
                    transitions[:, ii, jj, 0] = self._find_edge(dat_isin[:, ii, jj])
            if self.debug:
                print("second loop")
            for ii in range(self.w3d.nx):
                for jj in range(self.w3d.nz):
                    transitions[ii, :, jj, 1] = self._find_edge(dat_isin[ii, :, jj])
            if self.debug:
                print("third loop")

        intersections = np.logical_or.reduce(1 - transitions, axis=3).astype('int')

        n1, n2, n3 = np.where(intersections == 1)
        mX, mY, mZ = [], [], []
        for i, j, k in zip(n1, n2, n3):
            mX.append(X[i, j, k])
            mY.append(Y[i, j, k])
            mZ.append(Z[i, j, k])

        return np.array(mX), np.array(mY), np.array(mZ)

    def _find_edge(self, points):
        new_ar = np.copy(points)
        for i in range(points.size - 1):
            if i == 0:
                continue
            if points[i - 1] == 0 and points[i + 1] == 0:
                new_ar[i] = 1
        return new_ar

    def generate_faces_3d(self):
        if self.debug:
            print("Particles on Unstructured:", self.pids.size)
        x, y, z = self._isinside()
        if self.debug:
            print("cells for unstructured", x.shape, y.shape)
        s = self._color_mesh(mesh=np.vstack([x.ravel(), y.ravel(), z.ravel()]), particle_subset=None)

        yield x, y, z, s


def gather_cond_id(top):
    try:
        if comm_world.size == 1:
            return top.pidlost[:, -1]
    except NameError:
        return top.pidlost[:, -1]

    # Prepare head to receive particles arrays from all valid ranks
    if comm_world.rank == 0:
        surface_id = []
        for rank in range(1, comm_world.size):
            surface_id.extend(comm_world.recv(source=rank, tag=0))

    # All ranks that have valid data send to head
    if comm_world.rank != 0:
        try:
            comm_world.send(top.pidlost[:, -1], dest=0, tag=0)
        except toperror:
            comm_world.send([], dest=0, tag=0)




conductor_type_2d = {XPlane: XPlanePlot,
                  ZPlane: ZPlanePlot,
                  Box: BoxPlot}

conductor_type_3d = {XPlane: XPlanePlot,
                  YPlane: YPlanePlot,
                  ZPlane: ZPlanePlot,
                  Box: BoxPlot,
                  Sphere: SpherePlot,
                  'Unstructured': UnstructuredPlot}
