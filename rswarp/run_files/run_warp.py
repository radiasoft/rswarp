import sys
sys.path.append('/home/vagrant/jupyter/repos/rswarp/rswarp/run_files/')
from gridded_tec_3d import main

if __name__ == '__main__':
    """
    int, int,
    float,
    float, float, float
    str
    """
    args = sys.argv[1:]

    nx = int(args[0])
    ny = int(args[1])
    volts_on_grid = float(args[2])
    grid_height = float(args[3])
    strut_width = float(args[4])
    strut_height = float(args[5])

    id = int(args[6])

    run_directory = str(args[7])

    save_directory = run_directory

    main(nx, ny, volts_on_grid,
         grid_height, strut_width, strut_height, id,
         random_seed=False, install_grid=False,
         injection_type=2, cathode_temperature=1273.15,
         particle_diagnostic_switch=True, field_diagnostic_switch=False, lost_diagnostic_flag=True)

    print 0
