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
except ImportError:
    try:
        from rswarp.run_files.gridded_tec_3d import main
    except ImportError:
        raise ImportError, "Could not find rswarp.run_files.simple_tec_3d"

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

    # Values based on Voesch et al.
    x_struts = 1
    y_struts = 1
    V_grid = 15.0
    grid_height = 0.5
    strut_width = 2e-9
    strut_height = 2e-9
    rho_ew = 1.1984448e-03
    T_em = 1414 + 273.15
    phi_em = 2.174
    T_coll = 50 + 273.15
    phi_coll = 0.381
    rho_cw = 1.1984448e-03
    gap_distance = 1e-6
    run_id = 1

    main(x_struts, y_struts, V_grid, grid_height, strut_width, strut_height,
         rho_ew, T_em, phi_em, T_coll, phi_coll, rho_cw, gap_distance,
         run_id,
         injection_type=2, random_seed=True, install_grid=False, max_wall_time=1e9,
         particle_diagnostic_switch=True, field_diagnostic_switch=False, lost_diagnostic_switch=False)

    print 0
