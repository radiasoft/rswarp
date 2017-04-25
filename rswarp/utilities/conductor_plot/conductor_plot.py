import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

    def __init__(self, artist=None):
        self.fig = None
        self.artist = artist

        self.positive_conductors = []
        self.negative_conductors = []
        self.ground_conductors = []
        self.dielectrics = []

        self.scale = 1.

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
                            if conductor.voltage > 0.:
                                self.positive_conductors.append(self.set_rectangle_patch(conductor))
                            elif conductor.voltage < 0.:
                                self.negative_conductors.append(self.set_rectangle_patch(conductor))
                            elif conductor.voltage == 0.:
                                self.ground_conductors.append(self.set_rectangle_patch(conductor))
                        else:
                            self.dielectrics.append(self.set_rectangle_patch(conductor))

    def conductor_collection(self):
        d

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
            return "FAILURE"

        xcorner = x - xlength / 2.
        ycorner = y - ylength / 2.

        p = patches.Rectangle(
            (xcorner * self.scale, ycorner * self.scale),
            xlength * self.scale,
            ylength * self.scale)

        return p

    def create_artist(self):
        try:
            xmin = w3d.xmmin
            xmax = w3d.xmmax
            zmin = w3d.zmmin
            zmax = w3d.zmmax
        except:
            return "No Grid Information"

        # Try to guess an ideal scaling
        if abs(xmax - xmin) * 1e3 > 1.:
            self.scale = 1e3
        elif abs(xmax - xmin) * 1e6 > 1.:
            self.scale = 1e6
        elif abs(xmax - xmin) * 1e9 > 1.:
            self.scale = 1e9
        else:
            self.scale = 1.

        fig, ax1 = plt.subplots(1, 1)
        ax1.set_xlim(zmin * self.scale, zmax * self.scale)
        ax1.set_ylim(xmin * self.scale, xmax * self.scale)

        self.fig = fig
        self.artist = ax1
