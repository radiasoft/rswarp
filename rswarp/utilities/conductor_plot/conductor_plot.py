import matplotlib.pyplot as plt
import matplotlib.patches as patches
from warp import field_solvers
from warp import w3d


class PlotConductors(object):
    conductor_types = {}
    conductor_types['ZPlane'] = ['xcent', 'ycent']
    conductor_types['Box'] = ['xcent', 'ycent', 'zcent', 'xsize', 'ysize', 'zsize']

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

    def conductor_coordinates(self, solver):

        # Iterate through all conductor lists in the solver
        for key in solver.installedconductorlists:
            # Iterate through all condcutor objects
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

    def set_rectangle_patch(self, conductor):
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
