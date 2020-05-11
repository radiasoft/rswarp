# -*- coding: utf-8 -*-
u"""

:copyright: Copyright (c) 2020 RadiaSoft LLC.  All Rights Reserved.
:license: http://www.apache.org/licenses/LICENSE-2.0.html
"""
from __future__ import absolute_import, division, print_function
import time
import os
import yaml
import numpy as np


def record_time(func, time_list, *args, **kwargs):
    t1 = time.time()
    func(*args, **kwargs)
    t2 = time.time()
    time_list.append(t2 - t1)


class SteadyState:
    def __init__(self, top, cond, interval, tol=0.1):
        # Analyze if we are in steady-state regime by means of FFT.
        # Use estimated crossing time as the interval period and then look at
        self.top = top
        self.cond = cond
        self.tol = tol
        self.n = 0
        self.level = 1

        tstart = (self.top.it - interval) * self.top.dt
        _, collector_current = cond.get_current_history(js=None, l_lost=1, l_emit=0,
                                                        l_image=0, tmin=tstart, tmax=None, nt=top.it)
        self.y0 = np.abs(np.fft.rfft(collector_current)[1])

    def __call__(self, interval):
        tstart = (self.top.it - interval) * self.top.dt
        _, collector_current = self.cond.get_current_history(js=None, l_lost=1, l_emit=0,
                                                             l_image=0, tmin=tstart, tmax=None, nt=interval)
        y = np.abs(np.fft.rfft(collector_current)[1])
        self.level = y / self.y0

        if self.level <= self.tol:
            return 1
        else:
            return -1


class ExternalCircuit:
    def __init__(self, top, solver, rho, contact_potential, area, conductors, voltage_stride=20, debug=False):
        self.top = top
        self.solver = solver
        self.rho = rho
        self.contact_potential = contact_potential
        self.area = area  # in cm**2
        self.voltage_stride = voltage_stride
        self.debug = debug
        # TODO: these seem to remain empty when running warp in parallel
        self.voltage_history = []
        self.current_history = []

        try:
            conductors[0]
            self.conductors = conductors
        except TypeError:
            self.conductors = [conductors]

    def getvolt(self, t):
        # t is dummy variable
        if self.top.it % self.voltage_stride == 0:
            # This will tell the solver to update voltage on all conductors
            self.solver.gridmode = 0

        tmin = (self.top.it - self.voltage_stride) * self.top.dt
        for cond in self.conductors:
            times, current = cond.get_current_history(js=None, l_lost=1, l_emit=0,
                                                     l_image=0, tmin=tmin, tmax=None, nt=1)
            # Using many bins for the current sometimes gives erroneous zeros.
            # Using a single bin has consistently given a result ~1/2 expected current, hence the sum of the two values
            current = np.sum(current)
            voltage = self.contact_potential + current / self.area * self.rho

            self.voltage_history.append(voltage)
            self.current_history.append(current / self.area)
            if self.debug:
                print("Current/voltage at step: {} = {}, {}".format(self.top.it, current / self.area, voltage))

        return voltage


def write_parameter_file(pars, filename=None):
    path, filename = os.path.split(filename)
    if not filename:
        try:
            filename = 'run_attributes_{}.yaml'.format(pars['run_id'], pars['run_id'])
        except KeyError:
            print("No Filename or run_id, attributes will not be saved")
            return
        if path:
            filename = os.path.join(path, filename)
    else:
        if os.path.splitext(filename)[-1] != '.yaml':
            filename = os.path.splitext(filename)[0] + '.yaml'
        if path:
            filename = os.path.join(path, filename)
        else:
            filename = os.path.join(path, filename)

    tec_keys = ['x_struts', 'y_struts', 'V_grid', 'grid_height', 'strut_width', 'strut_height',
                'rho_ew', 'T_em', 'phi_em', 'T_coll', 'phi_coll', 'rho_cw', 'gap_distance', 'rho_load',
                'run_id']
    simulation_keys = ['injection_type', 'random_seed', 'install_grid', 'install_circuit', 'max_wall_time',
                       'particle_diagnostic_switch', 'field_diagnostic_switch', 'lost_diagnostic_switch', 'channel_width']

    tec_parameters = {}
    simulation_parameters = {}
    other_parameters = {}

    for key in pars:
        if key in tec_keys:
            tec_parameters[key] = pars[key]
        elif key in simulation_keys:
            simulation_parameters[key] = pars[key]
        else:
            other_parameters[key] = pars[key]

    pars = {'tec': tec_parameters, 'simulation': simulation_parameters, 'other': other_parameters}

    with open(filename, 'w') as outputfile:
        yaml.dump(pars, outputfile, default_flow_style=False)


def read_parameter_file(filename):
    parameters = yaml.safe_load(open(filename, 'r'))
    input_deck = {}
    for key0 in parameters:
        for key1, val in parameters[key0].items():
            input_deck[key1] = val

    return input_deck
