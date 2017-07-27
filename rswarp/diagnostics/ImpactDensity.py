from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from sys import maxint
from scipy.interpolate import interp1d
from warp.field_solvers.generateconductors import ZPlane
import numpy as np
import matplotlib.cm as cm


class PlotDensity(object):

    def __init__(self, ax, ax_colorbar, scraper, top, w3d):
        self.ax = ax
        self.ax_colorbar = ax_colorbar
        self.scraper = scraper
        self.top = top
        self.w3d = w3d

        self.gated_ids = {}
        self.zplost = self.top.zplost
        self.xplost = self.top.xplost
        self.dx = w3d.dx
        self.dz = w3d.dz

        self.cmap = cm.coolwarm
        self.normalization = Normalize
        self.cmap_normalization = None

    def __call__(self, *args, **kwargs):
        self.gate_scraped_particles()
        self.map_density()
        self.generate_plots()

    def gate_scraped_particles(self):

        xmmin = self.w3d.xmmin
        xmmax = self.w3d.xmmax
        zmmin = self.w3d.zmmin
        zmmax = self.w3d.zmmax

        for cond in self.scraper.conductors:
            self.gated_ids[cond.condid] = {}

            if isinstance(cond, ZPlane):
                zmin, zmax = cond.zcent - self.dz / 2., cond.zcent + self.dz / 2.
                xmin, xmax = xmmin, xmmax
                ids = np.where((zmin < self.zplost) & (self.zplost < zmax) &
                               (xmin < self.xplost) & (self.xplost < xmax))
                self.gated_ids[cond.condid]['left'] = {'pids': list(ids), 'limits': [zmin, zmax, xmin, xmax]}
                continue

            # top
            zmin, zmax = cond.zcent - cond.zsize / 2., cond.zcent + cond.zsize / 2.
            xmin, xmax = cond.xcent + cond.xsize / 2. - self.dx / 2., cond.xcent + cond.xsize / 2. + self.dx / 2.

            ids = np.where((zmin < self.zplost) & (self.zplost < zmax) & (xmin < self.xplost) & (self.xplost < xmax))
            self.gated_ids[cond.condid]['top'] = {'pids': list(ids), 'limits': [zmin, zmax, xmin, xmax]}

            # bottom
            zmin, zmax = cond.zcent - cond.zsize / 2., cond.zcent + cond.zsize / 2.
            xmin, xmax = cond.xcent - cond.xsize / 2. - self.dx / 2., cond.xcent - cond.xsize / 2. + self.dx / 2.

            ids = np.where((zmin < self.zplost) & (self.zplost < zmax) & (xmin < self.xplost) & (self.xplost < xmax))
            self.gated_ids[cond.condid]['bottom'] = {'pids': list(ids), 'limits': [zmin, zmax, xmin, xmax]}

            # left
            zmin, zmax = cond.zcent - cond.zsize / 2. - self.dz / 2., cond.zcent - cond.zsize / 2. + self.dz / 2.
            xmin, xmax = cond.xcent - cond.xsize / 2., cond.xcent + cond.xsize / 2.

            ids = np.where((zmin < self.zplost) & (self.zplost < zmax) & (xmin < self.xplost) & (self.xplost < xmax))
            self.gated_ids[cond.condid]['left'] = {'pids': list(ids), 'limits': [zmin, zmax, xmin, xmax]}

            # right
            zmin, zmax = cond.zcent + cond.zsize / 2. - self.dz / 2., cond.zcent + cond.zsize / 2. + self.dz / 2.
            xmin, xmax = cond.xcent - cond.xsize / 2., cond.xcent + cond.xsize / 2.

            ids = np.where((zmin < self.zplost) & (self.zplost < zmax) & (xmin < self.xplost) & (self.xplost < xmax))
            self.gated_ids[cond.condid]['right'] = {'pids': list(ids), 'limits': [zmin, zmax, xmin, xmax]}

    def map_density(self):
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
                    plot = self.ax.scatter(np.linspace(zmin, zmax, points) * 1e6,
                                           [(xmin + self.dx / 2.) * 1e6] * points,
                                           c=color_mapping,
                                           s=1,
                                           linewidths=1,
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
                    plot = self.ax.scatter([(zmin + self.dz / 2.) * 1e6] * points,
                                           np.linspace(xmin, xmax, points) * 1e6,
                                           c=color_mapping,
                                           s=1,
                                           linewidths=1,
                                           # marker='_',
                                           zorder=50)
                    scatter_plots.append(plot)

        ColorbarBase(self.ax_colorbar, cmap=self.cmap, norm=self.cmap_normalization)

    """
    Generates plots and a colorbar from lost particle data collected from Warp's scraper object.
    Plots and the colorbar are automatically created on the axes specified.
    It is the user's responsibility to make sure axes are suitably configured.

    Args:
        ax: matplotlib axis for overlaying scraper geometry
        ax_color: matplotlib axis for the colorbar
        scraper: Warp scraper object with lost particle statistics
        xplost: top.xplost array
        zplost: top.zplost array
        dx: x cell size (or desired discretization size in x)
        dz: z cell size (or desired discretization size in x)
        xmmin: Lower bound x-position of the simulation domain (w3d.xmmin in warp)
        xmmax: Upper bound x-position of the simulation domain (w3d.xmmax in warp)
        xmmin: Lower bound z-position of the simulation domain (w3d.zmmin in warp)
        xmmax: Upper bound z-position of the simulation domain (w3d.zmmax in warp)
    Returns:
        dictionary of gated data
    """






