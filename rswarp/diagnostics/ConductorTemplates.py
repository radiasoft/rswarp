import numpy as np
from warp.field_solvers.generateconductors import XPlane, YPlane, ZPlane, Box
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

    def _generate_face(self):
        scraped_parts = np.array([self.top.xplost, self.top.yplost, self.top.zplost]).T[self.pids, :]
        for bound, axis, face in zip(self.cbounds, self.axis, self.faces):
            xp = 1 + 5 * np.abs(self.cbounds[(axis + 1) % 3] - self.cbounds[(axis + 1) % 3 + 3]) \
                // self.cell_size[(axis + 1) % 3]
            yp = 1 + 5 * np.abs(self.cbounds[(axis + 2) % 3] - self.cbounds[(axis + 2) % 3 + 3]) \
                // self.cell_size[(axis + 2) % 3]

            x, y = np.meshgrid(np.linspace(self.cbounds[(axis + 1) % 3],
                                           self.cbounds[(axis + 1) % 3 + 3], xp),
                               np.linspace(self.cbounds[(axis + 2) % 3],
                                           self.cbounds[(axis + 2) % 3 + 3], yp))
            z = np.ones_like(x) * bound
            positions = np.vstack([x.ravel(), y.ravel()])

            if face.size > 30:
                if self.interpolation == 'kde':
                    # 30 is empirically chosen as a bound where the KDE returns reasonable results
                    kernel = gaussian_kde(scraped_parts[face, :][:, [(axis + 1) % 3, (axis + 2) % 3]].T)
                    # Use surface point number as part of the normalization to prevent small surfaces returning outsized
                    # results from the KDE
                    s = np.reshape(kernel(positions).T, x.shape) * x.size
                elif self.interpolation == 'cubic':
                    s, _, _ = np.histogram2d(scraped_parts[face, (axis + 1) % 3], scraped_parts[face, (axis + 2) % 3],
                                             bins=x.shape, range=[[self.cbounds[(axis + 1) % 3],
                                               self.cbounds[(axis + 1) % 3 + 3]], [self.cbounds[(axis + 2) % 3],
                                               self.cbounds[(axis + 2) % 3 + 3]]])
                    s = griddata(positions.T, s.ravel(), (x, y), method='cubic') * x.size
            elif face.size > 1:
                s, _, _ = np.histogram2d(scraped_parts[face, (axis + 1) % 3], scraped_parts[face, (axis + 2) % 3],
                                         bins=x.shape)
            else:
                s = np.ones_like(x) * -1.0

            yield x, y, z, s


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
            self.faces.append(pof[0])

    def generate_faces(self):
        if len(self.faces) == 0:
            self.get_particles()
        for axis, mesh in zip(self.axis, self._generate_face()):
            x, y, z = [mesh[0], mesh[1], mesh[2]][(2 * axis + 2) % 3], \
                      [mesh[0], mesh[1], mesh[2]][(2 * axis + 3) % 3], \
                      [mesh[0], mesh[1], mesh[2]][(2 * axis + 1) % 3]
            s = mesh[3]
            yield x, y, z, s


class Plane(Conductor):

    def get_particles(self):
        # The plane has only one face so we use all particles with appropriate pid
        self.faces.append(np.arange(self.pids.size))

    def generate_faces(self):
        if len(self.faces) == 0:
            self.get_particles()
        for mesh in self._generate_face():
            x, y, z = mesh[0], mesh[1], np.ones_like(mesh[0]) * self.center[0]
            s = mesh[3]
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


conductor_type = {XPlane: XPlanePlot,
                  YPlane: YPlanePlot,
                  ZPlane: ZPlanePlot,
                  Box: BoxPlot}
