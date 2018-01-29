import sys
# Path for use on jupyter.radiasoft
sys.path.append('/home/vagrant/jupyter/repos/rswarp/rswarp/run_files/')

try:
    from gridded_tec_3d import main
except ImportError:
    try:
        from rswarp.run_files.gridded_tec_3d import main
    except ImportError:
        raise ImportError, "Could not find rswarp.run_files.gridded_tec_3d"

if __name__ == '__main__':
    """
    nx: Number of struts that intercept the x-axis.
    ny: Number of struts that intercept the y-axis
    volts_on_grid: Voltage to place on the grid in Volts.
    grid_height: Distance from the emitter to the grid in meters.
    strut_width: Transverse extent of the struts in meters.
    strut_height: Longitudinal extent of the struts in meters.
    run_id: Run id. Mainly used for parallel optimization.
    
    Example values: 4 8 17.7265855605 1.07243108547e-07 1.97828380067e-09 1.7538111785e-08 1
    """
    args = sys.argv[1:]

    nx = int(args[0])
    ny = int(args[1])
    volts_on_grid = float(args[2])
    grid_height = float(args[3])
    strut_width = float(args[4])
    strut_height = float(args[5])

    run_id = int(args[6])

    main(nx, ny, volts_on_grid,
         grid_height, strut_width, strut_height, run_id,
         random_seed=False, install_grid=False,
         injection_type=2, cathode_temperature=1273.15,
         particle_diagnostic_switch=True, field_diagnostic_switch=False, lost_diagnostic_flag=True)

    print 0
