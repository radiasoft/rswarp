import sys
from new_run_tec_with_reflection import main
from rswarp.run_files.tec.tec_utilities import read_parameter_file

if __name__ == '__main__':
    config_file = sys.argv[1]
    run_attributes = read_parameter_file(config_file)

    main(**run_attributes)