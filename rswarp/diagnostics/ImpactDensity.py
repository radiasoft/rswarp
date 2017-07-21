from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from sys import maxint
from scipy.interpolate import interp1d
from warp.field_solvers.generateconductors import ZPlane
import matplotlib as mpl
import numpy as np
import matplotlib.cm as cm


def plot_impact_density(ax, ax_color, scraper, xplost, zplost, dx, dz, xmmin, xmmax, zmmin, zmmax):
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
    Returns:
        dictionary of gated data

    """
    cmap = cm.coolwarm

    gated_ids = {}
    for cond in scraper.conductors:
        gated_ids[cond.condid] = {}

        if isinstance(cond, ZPlane):
            zmin, zmax = cond.zcent - dz / 2., cond.zcent + dz / 2.
            xmin, xmax = xmmin, xmmax
            ids = np.where((zmin < zplost) & (zplost < zmax) & (xmin < xplost) & (xplost < xmax))
            gated_ids[cond.condid]['left'] = {'pids': list(ids), 'limits': [zmin, zmax, xmin, xmax]}
            continue

        # top
        zmin, zmax = cond.zcent - cond.zsize / 2., cond.zcent + cond.zsize / 2.
        xmin, xmax = cond.xcent + cond.xsize / 2. - dx / 2., cond.xcent + cond.xsize / 2. + dx / 2.

        ids = np.where((zmin < zplost) & (zplost < zmax) & (xmin < xplost) & (xplost < xmax))
        gated_ids[cond.condid]['top'] = {'pids': list(ids), 'limits': [zmin, zmax, xmin, xmax]}

        # bottom
        zmin, zmax = cond.zcent - cond.zsize / 2., cond.zcent + cond.zsize / 2.
        xmin, xmax = cond.xcent - cond.xsize / 2. - dx / 2., cond.xcent - cond.xsize / 2. + dx / 2.

        ids = np.where((zmin < zplost) & (zplost < zmax) & (xmin < xplost) & (xplost < xmax))
        gated_ids[cond.condid]['bottom'] = {'pids': list(ids), 'limits': [zmin, zmax, xmin, xmax]}

        # left
        zmin, zmax = cond.zcent - cond.zsize / 2. - dz / 2., cond.zcent - cond.zsize / 2. + dz / 2.
        xmin, xmax = cond.xcent - cond.xsize / 2., cond.xcent + cond.xsize / 2.

        ids = np.where((zmin < zplost) & (zplost < zmax) & (xmin < xplost) & (xplost < xmax))
        gated_ids[cond.condid]['left'] = {'pids': list(ids), 'limits': [zmin, zmax, xmin, xmax]}

        # right
        zmin, zmax = cond.zcent + cond.zsize / 2. - dz / 2., cond.zcent + cond.zsize / 2. + dz / 2.
        xmin, xmax = cond.xcent - cond.xsize / 2., cond.xcent + cond.xsize / 2.

        ids = np.where((zmin < zplost) & (zplost < zmax) & (xmin < xplost) & (xplost < xmax))
        gated_ids[cond.condid]['right'] = {'pids': list(ids), 'limits': [zmin, zmax, xmin, xmax]}

    min_density = maxint
    max_density = 0
    for id in gated_ids:
        for side in gated_ids[id]:
            pids = gated_ids[id][side]['pids']
            if side == 'top' or side == 'bottom':
                gated_ids[id][side]['density'], gated_ids[id][side]['positions'] = np.histogram(zplost[pids], 'fd')
            if side == 'left' or side == 'right':
                gated_ids[id][side]['density'], gated_ids[id][side]['positions'] = np.histogram(xplost[pids], 'fd')

            if np.max(gated_ids[id][side]['density']) > max_density:
                max_density = np.max(gated_ids[id][side]['density'])
            if np.min(gated_ids[id][side]['density']) < min_density:
                min_density = np.min(gated_ids[id][side]['density'])

            gated_ids[id][side]['interpolation'] = interp1d(np.arange(gated_ids[id][side]['density'].size),
                                                  gated_ids[id][side]['density'])

    cmap_normalization = Normalize(min_density, max_density)
    scatter_plots = []
    points = 1000
    # TODO: Assuming um scale for now. Need to generalize.
    for id in gated_ids:
        for side in gated_ids[id]:
            zmin, zmax = gated_ids[id][side]['limits'][0:2]
            xmin, xmax = gated_ids[id][side]['limits'][2:4]
            if side == 'top' or side == 'bottom':
                size = gated_ids[id][side]['density'].size
                interp = gated_ids[id][side]['interpolation']
                plot = ax.scatter(np.linspace(zmin, zmax, points) * 1e6,
                           [(xmin + dx / 2.) * 1e6] * points,
                           c=cmap(cmap_normalization(interp(np.linspace(0, size - 1, points)))),
                           s=1,
                           zorder=50)
                scatter_plots.append(plot)
            if side == 'left' or side == 'right':
                size = gated_ids[id][side]['density'].size
                interp = gated_ids[id][side]['interpolation']
                plot = ax.scatter([(zmin + dz / 2.) * 1e6] * points,
                           np.linspace(xmin, xmax, points) * 1e6,
                           c=cmap(cmap_normalization(interp(np.linspace(0, size - 1, points)))),
                           s=1,
                           zorder=50)
                scatter_plots.append(plot)

    ColorbarBase(ax_color, cmap=cmap, norm=cmap_normalization)

    return gated_ids




