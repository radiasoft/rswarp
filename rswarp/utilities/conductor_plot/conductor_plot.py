import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from warp import field_solvers
from warp import w3d


# TODO: Would be nice to have 'run_once' in a central repository location
def run_once(f):
    """
        Decorator to ensure decorated function can only be run once.
        Will return original output on subsequent calls.
    """
    def dec(*args, **kwargs):
        if not dec.run:
            dec.run = True
            dec.out = f(*args, **kwargs)
            return dec.out
        else:
            return dec.out

    dec.run = False

    return dec


class PlotConductors(object):
    # Supported conductor types
    conductor_types = ['Box']

    # Attributes template
    conductor_attributes = {'xcent': None,
                            'ycent': None,
                            'zcent': None,
                            'xsize': None,
                            'ysize': None,
                            'zsize': None,
                            'voltage': None,
                            'permeability': None,
                            'permittivity': None}

    def __init__(self, artist=None, xbounds=None, zbounds=None):
        try:
            self.xmin = w3d.xmmin
            self.xmax = w3d.xmmax
            self.zmin = w3d.zmmin
            self.zmax = w3d.zmmax
        except:
            self.xmin = xbounds[0]
            self.xmax = xbounds[1]
            self.zmin = zbounds[0]
            self.zmax = zbounds[1]

        # Try to guess an ideal scaling
        if abs(self.xmax - self.xmin) * 1e3 > 1.:
            self.scale = 1e3
        elif abs(self.xmax - self.xmin) * 1e6 > 1.:
            self.scale = 1e6
        elif abs(self.xmax - self.xmin) * 1e9 > 1.:
            self.scale = 1e9
        else:
            self.scale = 1.

        self.fig = None
        self.artist = artist
        self.conductors = []
        self.voltages = []
        self.dielectrics = []

        self.patch_colors = []

    @run_once
    def conductor_coordinates(self, solver):
        """
        Runs logic for finding which conductors can be plotted and run appropriate patch creation functions.
        Args:
            solver: Warp fieldsolver object containing conductors to be plotted.

        Returns:
                None
        """

        # Iterate through all conductor lists in the solver
        for key in solver.installedconductorlists:
            # Iterate through all conductor objects
            for conductor in solver.installedconductorlists[key]:
                # Perform check to make sure this is a conductor the code knows how to handle
                for obj_type in self.conductor_types:
                    if isinstance(conductor, getattr(field_solvers.generateconductors, obj_type)):
                        if conductor.voltage is not None:
                            self.conductors.append(self.set_rectangle_patch(conductor))
                            self.voltages.append(conductor.voltage)
                        else:
                            self.dielectrics.append(self.set_rectangle_patch(conductor))
                            # TODO: Will add permittivity when it becomes available

    def conductor_collection(self):
        for voltage in self.voltages:
            if voltage < 0.:
                self.patch_colors.append(plt.cm.seismic(240))
            elif voltage > 0.:
                self.patch_colors.append(plt.cm.seismic(15))
            elif voltage == 0.:
                self.patch_colors.append('grey')

        self.patches = PatchCollection(self.conductors + self.dielectrics)
        self.patches.set_color(self.patch_colors)
        self.artist.add_collection(self.patches)

    def set_rectangle_patch(self, conductor):
        """
        Creates a mpl.patches.Rectangle object to represent a box in the XZ plane.
        Args:
            conductor: Warp conductor object

        Returns:
            mpl.patches.Rectangle object

        """
        try:
            x = conductor.zcent
            y = conductor.xcent
            xlength = conductor.zsize
            ylength = conductor.xsize
        except:
            print "Conductor does not have correct attributes to plot: \n{}".format(conductor)
            return

        xcorner = x - xlength / 2.
        ycorner = y - ylength / 2.

        p = patches.Rectangle(
            (xcorner * self.scale, ycorner * self.scale),
            xlength * self.scale,
            ylength * self.scale)

        return p

    def create_artist(self):

        fig, ax1 = plt.subplots(1, 1)

        ax1.set_xlim(self.zmin * self.scale, self.zmax * self.scale)
        ax1.set_ylim(self.xmin * self.scale, self.xmax * self.scale)

        prefix = '($mm$)' * (self.scale == 1e3) + '($\mu m$)' * (self.scale == 1e6) + \
                 '($nm$)' * (self.scale == 1e9) + '($m$)' * (self.scale == 1.)

        ax1.set_xlabel('z ' + prefix)
        ax1.set_ylabel('x ' + prefix)

        self.fig = fig
        self.artist = ax1
