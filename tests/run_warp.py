"""
Setting up test for efficiency calculation.
Parameters compared against Voesch et al Energy Technol 2017, 5, 1-11
Expected efficiency ~58% with no grid losses
"""

import sys
# Path for use on jupyter.radiasoft
sys.path.append('/home/vagrant/jupyter/repos/rswarp/rswarp/run_files/tec/')
# Path for use on local
sys.path.append('/Users/chall/research/github/rswarp/rswarp/run_files/tec/')
try:
    from gridded_tec_3d import main
    from tec_utilities import read_parameter_file
except ImportError:
    try:
        from rswarp.run_files.tec.gridded_tec_3d import main
        from tec_utilities import read_parameter_file
    except ImportError:
        raise ImportError, "Could not find rswarp.run_files.tec.gridded_tec_3d"

if __name__ == '__main__':
    """
    x_struts: Number of struts that intercept the x-axis.
    y_struts: Number of struts that intercept the y-axis
    V_grid: Voltage to place on the grid in Volts.
    gap_distance: Distance from the emitter to the collector in meters
    grid_height: Distance from the emitter to the grid normalized by gap distance.
    strut_width: Transverse extent of the struts in meters.
    strut_height: Longitudinal extent of the struts in meters.
    T_em: Temperature of emitter in Kelvin
    phi_em: Reistivity of the emitter side wiring in ohm*cm
    T_coll: Temperature of collector in Kelvin
    phi_cw: Resistivity of collector side wiring in ohm*cm
    run_id: Run id will be added to diagnostic folder name. Mainly used for parallel optimization.
    """

    run_attributes = read_parameter_file('example_run_attributes.yaml')

    main(**run_attributes)

