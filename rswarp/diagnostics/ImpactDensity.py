from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from sys import maxint
from scipy.interpolate import interp1d
from warp.field_solvers.generateconductors import ZPlane
import numpy as np
import matplotlib.cm as cm

# TODO: Add attributes:
#   scatter points for surfaces?


class PlotDensity(object):

    """Plots density of scraped particles on conducting objects.

    """

    def __init__(self, ax, ax_colorbar, scraper, top, w3d):
        """
        Plots density of scraped particles on conducting objects.

        Can evaluate density on each surface of a Box or ZPlane separately and produce shaded density plots.
        To run automatically: call an initialized PlotDensity object.

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

        self.ax = ax
        self.ax_colorbar = ax_colorbar
        self.scraper = scraper
        self.top = top
        self.w3d = w3d

        self.gated_ids = {}
        self.dx = w3d.dx
        self.dz = w3d.dz
        self.scale = 1e6
        # categorize the number lost to avoid padded values at end of array
        self.numlost = top.npslost[0]
        self.zplost = self.top.zplost[:self.numlost]
        self.xplost = self.top.xplost[:self.numlost]

        self.cmap = cm.coolwarm
        self.normalization = Normalize
        self.cmap_normalization = None

    def __call__(self, *args, **kwargs):
        """
        Will produce produce matplotlib scatter plots and colorbar for density.
        Args:
            *args:
            **kwargs:

        Returns:

        """
        self.gate_scraped_particles()
        self.map_density()
        self.generate_plots()

    def gate_scraped_particles(self):
        """
        Isolate particle PIDs for each conductor surface.
        Returns:

        """
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
