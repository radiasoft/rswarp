from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
try:
    from mayavi import mlab
except ImportError:
    print("Mayavi not found. 3D plotting not enabled.")
from sys import maxint
from scipy.interpolate import interp1d
from ConductorTemplates import conductor_type
import numpy as np
import matplotlib.cm as cm
from scipy.constants import e

# TODO: Add attributes:
#   scatter points for surfaces?


class PlotDensity(object):

    """Plots density of scraped particles on conducting objects.

    """

    def __init__(self, ax, ax_colorbar, scraper, top, w3d, interpolation='kde'):
        """
        Plots density of scraped particles on conducting objects.

        Can evaluate density on each surface of a Box or ZPlane separately and produce shaded density plots.
        To run automatically: call an initialized PlotDensity object.

        # TODO: Change line below when new plotting capabilities are finished
        Warning: Only Box and ZPlane are supported at this time. Other conductor shapes will not be evaluated correctly.
                Only for 2D XZ simulations at this time.

        Args:
            ax: Matplotlib axes object for surface density plots.
            ax_colorbar: Matplotlib axes object for colorbar.
            scraper: Warp ParticleScraper object. Only used to acquire conductor positions and dimensions.
            top: Warp top object.
            w3d: Warp w3d object.

        Useful attributes:
            ax: Matplotilb axes object for density plots
            ax_colorbar: Matplotlib axes object for colorbar
            scraper: Warp ParticleScraper object
            zplost: Array of z-positions of lost particles. Defaults to top.zplost.
            xplost: Array of x-positions of lost particles. Defaults to top.xplost.
            dz, dx: z and x widths used to gate on particles collected by conductor side.
                Defaults to w3d.dz and w3d.dx
            scale: Set scale of x and z units. Defaults to 1e6 (units of microns).
            cmap: matplotlib.cm colormap. Defaults to coolwarm.
            normalization: matplotlib.colors normalization function. Defaults to Normalize (linear normalization).
        """
        assert scraper.lcollectlpdata, "Flag 'lcollectlpdata' not enabled for scraper. No particle data to plot."
        assert interpolation == 'kde' or interpolation == 'cubic', "Interpolation must be either 'cubic' or 'kde'"
        self.interpolation = interpolation

        self.scraper = scraper
        self.top = top
        self.w3d = w3d

        self.gated_ids = {}
        self.conductors = {}  # will replace gated_ids
        for cond in scraper.conductors:
            try:
                self.conductors[cond.condid] = conductor_type[type(cond)](top, w3d, cond,
                                                                          interpolation=self.interpolation)
                self.gated_ids[cond.condid] = conductor_type[type(cond)](top, w3d, cond,
                                                                          interpolation=self.interpolation)
            except KeyError:
                print("{} not currently implemented. Falling back to Unstructured method.".format(type(cond)))
                self.conductors[cond.condid] = conductor_type['Unstructured'](top, w3d, cond,
                                                                          interpolation=self.interpolation)
        self.dx = w3d.dx
        self.dz = w3d.dz
        self.scale = [1e9, 1e9, 1e9]
        # categorize the number lost to avoid padded values at end of array
        self.numlost = top.npslost[0]
        assert self.numlost > 1, "No particles lost in simulation. Nothing to plot."

        self.xplost = self.top.xplost[:self.numlost]
        self.yplost = self.top.yplost[:self.numlost]
        self.zplost = self.top.zplost[:self.numlost]
        self.time = top.dt * top.it

        if w3d.solvergeom == w3d.XZgeom or w3d.solvergeom == w3d.RZgeom:
            self.ax = ax
            self.ax_colorbar = ax_colorbar
            self.cmap = cm.coolwarm
            self.normalization = Normalize
            self.cmap_normalization = None
        else:
            self.ax = mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(800, 600))
            self.clf = mlab.clf()

    def __call__(self, *args, **kwargs):
        """
        Will produce produce matplotlib scatter plots and colorbar for density.
        Args:
            *args:
            **kwargs:

        Returns:

        """
        # self.gate_scraped_particles()
        # self.map_density()
        # self.generate_plots()

    # def gate_scraped_particles_3d(self):
    #     # If we don't go a class route:
    #     # for cond in self.scraper.conductors:
    #     #     self.conductors[cond.condid] = {}
    #
    #     for cond in self.conductors:
    #         scraped_particles = np.where(self.top.pidlost[:, -1] == cond.condid)


    #
    # def gate_scraped_particles(self):
    #     """
    #     Isolate particle PIDs for each conductor surface.
    #     Returns:
    #
    #     """
    #     xmmin = self.w3d.xmmin
    #     xmmax = self.w3d.xmmax
    #     zmmin = self.w3d.zmmin
    #     zmmax = self.w3d.zmmax
    #
    #     for cond in self.scraper.conductors:
    #         self.gated_ids[cond.condid] = {}
    #
    #         if isinstance(cond, ZPlane):
    #             zmin, zmax = cond.zcent - self.dz / 2., cond.zcent + self.dz / 2.
    #             xmin, xmax = xmmin, xmmax
    #             ids = np.where((zmin < self.zplost) & (self.zplost < zmax) &
    #                            (xmin < self.xplost) & (self.xplost < xmax))
    #             self.gated_ids[cond.condid]['left'] = {'pids': list(ids), 'limits': [zmin, zmax, xmin, xmax]}
    #             continue
    #
    #         # top
    #         zmin, zmax = cond.zcent - cond.zsize / 2., cond.zcent + cond.zsize / 2.
    #         xmin, xmax = cond.xcent + cond.xsize / 2. - self.dx / 2., cond.xcent + cond.xsize / 2. + self.dx / 2.
    #
    #         ids = np.where((zmin < self.zplost) & (self.zplost < zmax) & (xmin < self.xplost) & (self.xplost < xmax))
    #         self.gated_ids[cond.condid]['top'] = {'pids': list(ids), 'limits': [zmin, zmax, xmin, xmax]}
    #
    #         # bottom
    #         zmin, zmax = cond.zcent - cond.zsize / 2., cond.zcent + cond.zsize / 2.
    #         xmin, xmax = cond.xcent - cond.xsize / 2. - self.dx / 2., cond.xcent - cond.xsize / 2. + self.dx / 2.
    #
    #         ids = np.where((zmin < self.zplost) & (self.zplost < zmax) & (xmin < self.xplost) & (self.xplost < xmax))
    #         self.gated_ids[cond.condid]['bottom'] = {'pids': list(ids), 'limits': [zmin, zmax, xmin, xmax]}
    #
    #         # left
    #         zmin, zmax = cond.zcent - cond.zsize / 2. - self.dz / 2., cond.zcent - cond.zsize / 2. + self.dz / 2.
    #         xmin, xmax = cond.xcent - cond.xsize / 2., cond.xcent + cond.xsize / 2.
    #
    #         ids = np.where((zmin < self.zplost) & (self.zplost < zmax) & (xmin < self.xplost) & (self.xplost < xmax))
    #         self.gated_ids[cond.condid]['left'] = {'pids': list(ids), 'limits': [zmin, zmax, xmin, xmax]}
    #
    #         # right
    #         zmin, zmax = cond.zcent + cond.zsize / 2. - self.dz / 2., cond.zcent + cond.zsize / 2. + self.dz / 2.
    #         xmin, xmax = cond.xcent - cond.xsize / 2., cond.xcent + cond.xsize / 2.
    #
    #         ids = np.where((zmin < self.zplost) & (self.zplost < zmax) & (xmin < self.xplost) & (self.xplost < xmax))
    #         self.gated_ids[cond.condid]['right'] = {'pids': list(ids), 'limits': [zmin, zmax, xmin, xmax]}

    def map_density(self):
        """
        Use linear interpolation and colormap normalization to establish color scale (normalized to global density),
        for each surface.
        Returns:

        """
        min_density = maxint
        max_density = 0

        for gid in self.gated_ids:
            for side in self.gated_ids[gid]:
                pids = self.gated_ids[gid][side]['pids']
                if side == 'top' or side == 'bottom':
                    self.gated_ids[gid][side]['density'], self.gated_ids[gid][side]['positions'] = \
                        np.histogram(self.zplost[pids], 'fd')
                if side == 'left' or side == 'right':
                    self.gated_ids[gid][side]['density'], self.gated_ids[gid][side]['positions'] = \
                        np.histogram(self.xplost[pids], 'fd')

                if np.max(self.gated_ids[gid][side]['density']) > max_density:
                    max_density = np.max(self.gated_ids[gid][side]['density'])
                if np.min(self.gated_ids[gid][side]['density']) < min_density:
                    min_density = np.min(self.gated_ids[gid][side]['density'])
                try:
                    self.gated_ids[gid][side]['interpolation'] = interp1d(
                        np.arange(self.gated_ids[gid][side]['density'].size),
                        self.gated_ids[gid][side]['density'])
                except ValueError:
                    self.gated_ids[gid][side]['interpolation'] = None

        self.cmap_normalization = self.normalization(min_density, max_density)

    def generate_plots(self):
        """
        Creates scatter plots for each surface. Surfaces are composed of 1000 points each to mimic
        a gradient mapped on a line.
        Returns:

        """
        scatter_plots = []
        points = 1000
        # TODO: Assuming um scale for now. Need to generalize.
        for gid in self.gated_ids:
            for side in self.gated_ids[gid]:
                zmin, zmax = self.gated_ids[gid][side]['limits'][0:2]
                xmin, xmax = self.gated_ids[gid][side]['limits'][2:4]
                if side == 'top' or side == 'bottom':
                    size = self.gated_ids[gid][side]['density'].size
                    interp = self.gated_ids[gid][side]['interpolation']
                    try:
                        color_mapping = self.cmap(self.cmap_normalization(interp(np.linspace(0, size - 1, points))))
                    except:  # TODO: Need to find proper exception for zero particle case
                        color_mapping = 'k'
                    plot = self.ax.scatter(np.linspace(zmin, zmax, points) * self.scale,
                                           [(xmin + self.dx / 2.) * self.scale] * points,
                                           c=color_mapping,
                                           s=1,
                                           linewidths=0,
                                           # marker='|',
                                           zorder=50)
                    scatter_plots.append(plot)
                if side == 'left' or side == 'right':
                    size = self.gated_ids[gid][side]['density'].size
                    interp = self.gated_ids[gid][side]['interpolation']
                    try:
                        color_mapping = self.cmap(self.cmap_normalization(interp(np.linspace(0, size - 1, points))))
                    except:
                        color_mapping = 'k'
                    plot = self.ax.scatter([(zmin + self.dz / 2.) * self.scale] * points,
                                           np.linspace(xmin, xmax, points) * self.scale,
                                           c=color_mapping,
                                           s=1,
                                           linewidths=0,
                                           # marker='_',
                                           zorder=50)
                    scatter_plots.append(plot)

        ColorbarBase(self.ax_colorbar, cmap=self.cmap, norm=self.cmap_normalization)

    def generate_plots_2d(self):
        minS, maxS = maxint, 0
        contour_plots = []
        for cond in self.conductors.itervalues():

            for face in cond.generate_faces():
                x, y, z, s = face[0] * self.scale[0], face[1] * self.scale[1], \
                             face[2] * self.scale[2], face[3] * e / self.time * 1e-4
                print("m/m by face", np.min(s), np.max(s))
                if 0 <= np.min(s) < minS:  # -1 value indicates no particle anywhere on face
                    minS = np.min(s)
                if np.max(s) > maxS:
                    maxS = np.max(s)

                if np.min(s) < 0.0:
                    contour_plots.append(mlab.mesh(x, y, z, color=(0, 0, 0)))
                else:
                    contour_plots.append(mlab.mesh(x, y, z, scalars=s, colormap='viridis'))

    def generate_plots_3d(self):
        minS, maxS = maxint, 0
        contour_plots = []
        for cond in self.conductors.itervalues():

            for face in cond.generate_faces():
                x, y, z, s = face[0] * self.scale[0], face[1] * self.scale[1], \
                             face[2] * self.scale[2], face[3] * e / self.time * 1e-4
                print("m/m by face", np.min(s), np.max(s))
                if 0 <= np.min(s) < minS:  # -1 value indicates no particle anywhere on face
                    minS = np.min(s)
                if np.max(s) > maxS:
                    maxS = np.max(s)

                if isinstance(cond, conductor_type['Unstructured']):
                    pts = mlab.points3d(x, y, z, s, scale_mode='none', scale_factor=0.002)
                    mesh = mlab.pipeline.delaunay3d(pts)
                    contour_plots.append(mlab.pipeline.surface(mesh, colormap='viridis'))
                else:
                    if np.min(s) < 0.0:
                        contour_plots.append(mlab.mesh(x, y, z, color=(0, 0, 0), colormap='viridis'))
                    else:
                        contour_plots.append(mlab.mesh(x, y, z, scalars=s, colormap='viridis'))

        for cp in contour_plots:
            print(minS, maxS)
            cp.module_manager.scalar_lut_manager.trait_set(default_data_range=[minS * 0.95, maxS * 1.05])

        mlab.draw()
        mlab.colorbar(object=contour_plots[1], orientation='vertical')
        mlab.show()
