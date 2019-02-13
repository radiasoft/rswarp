from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
import matplotlib.cm as cm
try:
    from mayavi import mlab
except ImportError:
    print("Mayavi not found. 3D plotting not enabled.")

from sys import maxint
from ConductorTemplates import conductor_type_2d, conductor_type_3d
import numpy as np
from scipy.constants import e


class PlotDensity(object):

    """Plots density of scraped particles on conducting objects.

    """

    def __init__(self, ax, ax_colorbar, scraper, top, w3d, interpolation='kde'):
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

        self.numlost = top.npslost[0]
        assert self.numlost > 1, "No particles lost in simulation. Nothing to plot."
        assert scraper.lcollectlpdata, "Flag 'lcollectlpdata' not enabled for scraper. No particle data to plot."
        assert interpolation == 'kde' or interpolation == 'cubic', "Interpolation must be either 'cubic' or 'kde'"
        self.interpolation = interpolation

        self.scraper = scraper
        self.top = top
        self.w3d = w3d

        self.gated_ids = {}
        self.conductors = {}  # will replace gated_ids

        if w3d.solvergeom == w3d.XYZgeom:
            conductor_type = conductor_type_3d
        else:
            conductor_type = conductor_type_2d
        for cond in scraper.conductors:
            try:
                self.conductors[cond.condid] = conductor_type[type(cond)](top, w3d, cond,
                                                                          interpolation=self.interpolation)
            except KeyError:
                print("{} not currently implemented. Falling back to Unstructured method.".format(type(cond)))
                self.conductors[cond.condid] = conductor_type['Unstructured'](top, w3d, cond,
                                                                          interpolation=self.interpolation)
        self.dx = w3d.dx
        self.dz = w3d.dz
        self.scale = [1e9, 1e9, 1e9]

        self.time = top.dt * top.it

        if w3d.solvergeom == w3d.XZgeom or w3d.solvergeom == w3d.RZgeom:
            self.ax = ax
            zoffset = 0.05 * np.amax([w3d.zmmin, w3d.zmmax])
            xoffset = 0.05 * np.amax([w3d.xmmin, w3d.xmmax])
            ax.set_xlim((w3d.zmmin - zoffset) * self.scale[2], (w3d.zmmax + zoffset) * self.scale[2])
            ax.set_ylim((w3d.xmmin - xoffset) * self.scale[0], (w3d.xmmax + xoffset) * self.scale[0])

            self.ax_colorbar = ax_colorbar
            self.cmap = cm.coolwarm
            self.normalization = Normalize
            self.cmap_normalization = None
        else:
            self.ax = mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(800, 600))
            self.clf = mlab.clf()

    def generate_plots_2d(self):
        minS, maxS = maxint, 0
        scatter_plots = []
        for cond in self.conductors.itervalues():
            print(cond)
            for face in cond.generate_faces_2d():
                x, z, s = face[0] * self.scale[0], \
                             face[1] * self.scale[1], face[2] * e / self.time * 1e-4
                print("m/m by face", np.min(s), np.max(s))
                if 0 <= np.min(s) < minS:  # -1 value indicates no particle anywhere on face
                    minS = np.min(s)
                if np.max(s) > maxS:
                    maxS = np.max(s)

                if np.min(s) < 0.0:
                    scatter_plots.append(self.ax.scatter(z, x, c=s, s=1, linewidths=0, zorder=50))
                else:
                    scatter_plots.append(self.ax.scatter(z, x, c=s, cmap=self.cmap, s=1, linewidths=0, zorder=50))
        self.cmap_normalization = self.normalization(minS, maxS)
        ColorbarBase(self.ax_colorbar, cmap=self.cmap, norm=self.cmap_normalization)

    def generate_plots_3d(self):
        minS, maxS = maxint, 0
        contour_plots = []
        for cond in self.conductors.itervalues():

            for face in cond.generate_faces_3d():
                x, y, z, s = face[0] * self.scale[0], face[1] * self.scale[1], \
                             face[2] * self.scale[2], face[3] * e / self.time * 1e-4
                print("m/m by face", np.min(s), np.max(s))
                if 0 <= np.min(s) < minS:  # -1 value indicates no particle anywhere on face
                    minS = np.min(s)
                if np.max(s) > maxS:
                    maxS = np.max(s)

                if isinstance(cond, conductor_type_3d['Unstructured']):
                    pts = mlab.points3d(x, y, z, s, scale_mode='none', scale_factor=0.002)
                    mesh = mlab.pipeline.delaunay3d(pts)
                    contour_plots.append(mlab.pipeline.surface(mesh, colormap='viridis'))
                else:
                    if np.min(s) < 0.0:
                        contour_plots.append(mlab.mesh(x, y, z, color=(0, 0, 0), colormap='coolwarm'))
                    else:
                        contour_plots.append(mlab.mesh(x, y, z, scalars=s, colormap='coolwarm'))

        for cp in contour_plots:
            print(minS, maxS)
            cp.module_manager.scalar_lut_manager.trait_set(default_data_range=[minS * 0.95, maxS * 1.05])

        mlab.draw()
        mlab.colorbar(object=contour_plots[1], orientation='vertical')
        mlab.show()
