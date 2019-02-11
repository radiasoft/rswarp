import numpy as np
from warp.field_solvers.generateconductors import XPlane, YPlane, ZPlane, Box, Sphere
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata


class Conductor(object):

    def __init__(self, top, w3d, conductor, interpolation='kde'):
        assert interpolation == 'kde' or interpolation == 'cubic', "Interpolation must be either 'cubic' or 'kde'"
        self.interpolation = interpolation
        self.top = top
        self.conductor = conductor
        self.numlost = top.npslost[0]
        self.faces = []
        self.pids = None
        self.center = None
        self.size = None
        self.cbounds = None
        self.x, self.y, self.z = None, None, None
        self.faces = []
        self.pids = self._get_pids()
        self.thresshold = 30  # how many particles need to be lost to generate a colormap

        if w3d.solvergeom == w3d.XZgeom or w3d.solvergeom == w3d.RZgeom:
            self.axis = [0, 2, 0, 2]
            self.domain = [w3d.xmmin, w3d.zmmin, w3d.xmmax, w3d.zmmax]
            self.cell_size = [w3d.dx, w3d.dz]
        elif w3d.solvergeom == w3d.XYZgeom:
            self.axis = [0, 1, 2, 0, 1, 2]
            self.domain = [w3d.xmmin, w3d.ymmin, w3d.zmmin, w3d.xmmax, w3d.ymmax, w3d.zmmax]
            self.cell_size = [w3d.dx, w3d.dy, w3d.dz]

    def _get_pids(self):
        scraped_particles = np.where(self.top.pidlost[:self.numlost, -1] == self.conductor.condid)[0]

        return scraped_particles

    def _color_mesh(self, mesh, particle_subset=None):
        # all particles lost on conductor
        if particle_subset is not None:
            pids = particle_subset
        else:
            pids = self.pids
        scraped_parts = np.array([self.top.xplost, self.top.yplost, self.top.zplost])[:, pids]

        if scraped_parts.shape[1] > self.thresshold:
            if self.interpolation == 'kde':
                kernel = gaussian_kde(scraped_parts)
                # Use surface point number as part of the normalization to prevent small surfaces returning outsized
                # results from the KDE
                s = kernel(mesh).T  # * mesh.size
        else:
            s = np.ones_like(mesh[0, :]) * -1.0

        return s


class BoxPlot(Conductor):
    def __init__(self, top, w3d, conductor, interpolation='kde'):
        super(BoxPlot, self).__init__(top, w3d, conductor, interpolation=interpolation)
        self.center = conductor.xcent, conductor.ycent, conductor.zcent
        self.size = conductor.xsize, conductor.ysize, conductor.zsize

        xmin, xmax = self.center[0] - self.size[0] / 2., self.center[0] + self.size[0] / 2.
        ymin, ymax = self.center[1] - self.size[1] / 2., self.center[1] + self.size[1] / 2.
        zmin, zmax = self.center[2] - self.size[2] / 2., self.center[2] + self.size[2] / 2.
        mine, maxe = [xmin, ymin, zmin], [xmax, ymax, zmax]
        self.cbounds = np.hstack([mine, maxe])

    def get_particles(self):
        scraped_parts = np.array([self.top.xplost, self.top.yplost, self.top.zplost]).T[self.pids, :]
        for bound, axis, cell_size in zip(self.cbounds, self.axis, 2 * self.cell_size):
            pof = np.where(np.abs(scraped_parts[:, axis] - bound) <= cell_size)
            yield pof[0]

    def generate_faces(self):

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
            print("BOX:")
            print("Bounds x:", np.min(x), np.max(x))
            print("Bounds y:", np.min(y), np.max(y))
            print("Bounds z:", np.min(z), np.max(z))
            print("Particle count:", pids.size)
            print()
            s = self._color_mesh(mesh=np.vstack([x.ravel(), y.ravel(), z.ravel()]), particle_subset=pids).reshape(x.shape)

            yield x, y, z, s


class Plane(Conductor):

    def generate_faces(self):
        for bound, axis in zip(self.cbounds, self.axis):
            xn = 1 + 5 * np.abs(self.cbounds[(axis + 1) % 3] - self.cbounds[(axis + 1) % 3 + 3]) \
                 // self.cell_size[(axis + 1) % 3]
            yn = 1 + 5 * np.abs(self.cbounds[(axis + 2) % 3] - self.cbounds[(axis + 2) % 3 + 3]) \
                 // self.cell_size[(axis + 2) % 3]

            x, y = np.meshgrid(np.linspace(self.cbounds[(axis + 1) % 3],
                                           self.cbounds[(axis + 1) % 3 + 3], xn),
                               np.linspace(self.cbounds[(axis + 2) % 3],
                                           self.cbounds[(axis + 2) % 3 + 3], yn))
            z = np.ones_like(x) * self.center[0]
            s = self._color_mesh(mesh=np.vstack([x.ravel(), y.ravel(), z.ravel()]), particle_subset=None).reshape(x.shape)

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
        self.center = [conductor.xcent, ]
        self.cbounds = self.domain


class YPlanePlot(Plane):
    def __init__(self, top, w3d, conductor, interpolation='kde'):
        super(YPlanePlot, self).__init__(top, w3d, conductor, interpolation=interpolation)
        self.axis = [1, ]
        self.center = [conductor.ycent, ]
        self.cbounds = self.domain


class ZPlanePlot(Plane):
    def __init__(self, top, w3d, conductor, interpolation='kde'):
        super(ZPlanePlot, self).__init__(top, w3d, conductor, interpolation=interpolation)
        self.axis = [2, ]
        self.center = [conductor.zcent, ]
        self.cbounds = self.domain


class SpherePlot(Conductor):
    def __init__(self, top, w3d, conductor, interpolation='kde'):
        super(SpherePlot, self).__init__(top, w3d, conductor, interpolation=interpolation)
        self.center = [conductor.xcent, conductor.ycent, conductor.zcent]
        self.radius = conductor.radius

    def generate_faces(self):
        # TODO: Sphere has a fixed number of points right now. Everything else scales with the solver mesh spacing.
        print("SHERE HERE", self.pids.size)
        dphi = np.linspace(0, 2 * np.pi, 250)
        dtheta = np.linspace(-np.pi, np.pi, 250)
        phi, theta = np.meshgrid(dphi, dtheta)
        x = self.radius * np.cos(phi) * np.sin(theta) + self.center[0]
        y = self.radius * np.sin(phi) * np.sin(theta) + self.center[1]
        z = self.radius * np.cos(theta) + self.center[2]

        s = self._color_mesh(mesh=np.vstack([x.ravel(), y.ravel(), z.ravel()]), particle_subset=None).reshape(x.shape)

        yield x, y, z, s





conductor_type = {XPlane: XPlanePlot,
                  YPlane: YPlanePlot,
                  ZPlane: ZPlanePlot,
                  Box: BoxPlot,
                  Sphere: SpherePlot}
