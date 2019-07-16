from run_tec_reflections import main
import sys

if len(sys.argv) > 1:
    voltage = float(sys.argv[1])
    file_path = str(sys.argv[2])

run_options = {
        'dgap': 0.47e-3,
        'dt': 8e-12,
        'nsteps': 6000,
        'particles_per_step': 2400,
        'injection_type': 2,  # Thermionic injection without field enhancement
        'lambdaR': 0.5,       # Richardson constant AR = A0*lambdaR
        'srefprob': 0.0,      # probability of specular reflection
        'drefprob': 0.58,      # probability of diffuse reflection
        'cathode_temperature': 1050 + 273.15,
        'cathode_workfunction': 2.22, # in eV
        'anode_workfunction': 2.22 - 0.4,
        'anode_voltage': voltage,
        'gap_voltage': voltage + 0.23,
        'gate_voltage': 5.0,
        'beta': 27.,
        'reflection_scheme': "uniform",
        'reflections': True,
        'fieldperiod': 500,
        'particleperiod': 500,
        'file_path': file_path
    }


print("run simulation with: anode voltage = {}, gap voltage = {}, beta = {}, gap size = {}, dt = {}, nsteps = {}, particles_per_step = {}, reflection_scheme = {}".format(
        run_options['anode_voltage'], run_options['gap_voltage'], run_options['beta'], run_options['dgap'], run_options['dt'], run_options['nsteps'], run_options['particles_per_step'], run_options['reflection_scheme']))

main(**run_options)
