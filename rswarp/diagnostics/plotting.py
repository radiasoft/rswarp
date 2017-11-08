# TODO: Move ImpactDensity and ConductorPlot into this central location
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import widgets, interactive
from collections import OrderedDict
from IPython.display import display

# TODO: Right now you need to provide a live FieldSolver object. Might want option to give bounds and cell numbers.


class FieldLineout():
    """
    Class can be used in IPython notebooks to create an interactive plot of field/potential lineouts of 2D simulations.
    To use: Instantiate an instance of FieldLineout and then call that instance.
    """

    def __init__(self, solver, field_data, potential_data):
        """
        Initialize instance of the interactive plotter
        Args:
            solver: Instance of a Warp FieldSolver object
            field_data: Array of 3D field data (vector component, nx, nz).
            potential_data: Array of field data (nx, nz).
        """
        self.solver = solver
        self.field_data = field_data
        self.potential_data = potential_data

    def __call__(self):

        init = 'x'
        self.position_options = {'x': self.solver.xmesh, 'z': self.solver.zmesh}
        self.position_formatter = widgets.Dropdown(options=self.position_options[init],
                                                   description='Lineout Intercept')

        z_axis_options = OrderedDict()
        z_axis_options['E_x'] = 0
        z_axis_options['E_z'] = 1
        z_axis_options['Potential'] = 2
        z_axis_formatter = widgets.Dropdown(options=z_axis_options, description='Field Data')

        x_axis_options = OrderedDict()
        x_axis_options['x'] = 0
        x_axis_options['z'] = 1
        x_axis_formatter = widgets.Dropdown(options=x_axis_options, description='Lineout Axis')

        p1 = interactive(self._plot_lineout, x=x_axis_formatter, position=self.position_formatter, y=z_axis_formatter)
        display(p1)

    def _plot_lineout(self, x, position, y):

        self.x = x
        self.position = float(position)
        self.y = y

        # Select correct x and y data sets from user input:
        if x == 0:
            mesh = self.solver.xmesh  # Find index in x
            plot_mesh = self.solver.zmesh  # Plot as function of z
            self.position_formatter.options = self.position_options['x']  # switch which position array to use
        elif x == 1:
            mesh = self.solver.zmesh
            plot_mesh = self.solver.xmesh
            self.position_formatter.options = self.position_options['z']

        if self.y == 0:
            data = self.field_data[0][()]
        elif self.y == 1:
            data = self.field_data[2][()]
        elif self.y == 2:
            data = self.potential_data[()]

        # We use the entire E or phi array which gives E or phi data on 2D grid, need to transpose
        # depending on desired axis so we can use a single plotting command
        if self.x == 0:
            data = np.transpose(data)

        # Coordinate defintions
        x_coordinates = ['z (nm)', 'x (nm)']
        y_coordinates = ['$E_x$ (V/m)', '$E_z$ (V/m)', 'Potential (V)']

        # Choose appropriate index for lineout
        index = return_index(np.min(mesh), np.max(mesh), np.size(mesh), self.position)

        # Construct plot
        fig, ax = plt.subplots(1, 1)

        ax.plot(plot_mesh * 1e9, data[:, index])

        ax.set_xlabel("{}".format(x_coordinates[self.x]))
        ax.set_ylabel("{}".format(y_coordinates[self.y]))


def return_index(lbound, ubound, cells, position):
    """
    Give the position of a node on a 1D mesh this function will return the corresponding index
        of that node in an array that holds the node positions.

    lbound: Lower bound of mesh domain.
    ubound: Upper bound of mesh domain.
    cells: Number of cells along axis in mesh domain.
    position: Position of mesh node to find corresponding index for.

    returns
     Integer
    """

    index = (position - lbound) * cells / (ubound - lbound)

    return int(index)
