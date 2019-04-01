# -*- coding: utf-8 -*-
u"""Wrapper for `rswarp.run_files.tec.gridded_tec_3d.main`

:copyright: Copyright (c) 2019 RadiaSoft LLC.  All Rights Reserved.
:license: http://www.apache.org/licenses/LICENSE-2.0.html
"""
from __future__ import absolute_import, division, print_function

#: Parameters file for running at NERSC
DESIGN_FILE = 'tec_design_{run_id}.yaml'

#: How run_id is formatted
RUN_ID = '{generation}-{idn}'


def nersc(generation, idn):
    """Run `gridded_tec_3d` at NERSC.

    Reads parameters from ``tec_design_<generation>-<idn>.yaml``.  See
    `run_test` for parameter descriptions.

    Args:
        generation (str): generation
        idn (str): idn
    """
    from rswarp.run_files.tec import gridded_tec_3d
    from rswarp.run_files.tec import tec_utilities

    kw = dict(generation=generation, idn=idn)
    kw['run_id'] = RUN_ID.format(**kw)
    kw = tec_utilities.read_parameter_file(DESIGN_FILE.format(**kw))
    gridded_tec_3d.main(**kw)


def test(
    x_struts=1,
    y_struts=1,
    V_grid=15.0,
    grid_height=0.5,
    strut_width=2e-9,
    strut_height=2e-9,
    rho_ew=1.1984448e-03,
    T_em=1414 + 273.15,
    phi_em=2.174,
    T_coll=50 + 273.15,
    phi_coll=0.381,
    rho_cw=1.1984448e-03,
    gap_distance=1e-6,
    rho_load=0.01648048,
    run_id=0,
    injection_type=2,
    random_seed=True,
    install_grid=False,
    max_wall_time=1e9,
    particle_diagnostic_switch=True,
    field_diagnostic_switch=False,
    lost_diagnostic_switch=False,
):
    """Setup test for `gridded_tec_3d` efficiency calculation

    Parameters compared against Voesch et al Energy Technol 2017, 5, 1-11
    Expected efficiency ~58% with no grid losses

    Args:
        x_struts (int): Number of struts that intercept the x-axis [1]
        y_struts (int): Number of struts that intercept the y-axis [1]
        V_grid (float): Voltage to place on the grid in Volts [15.]
        grid_height (float): Distance from the emitter to the grid
            normalized by gap distance [.5]
        strut_width (float): Transverse extent of the struts in meters [2e-9]
        strut_height (float): Longitudinal extent of the struts in meters [2e-9]
        rho_ew (float): Emitter side wiring resistivity, ohms*cm [1.1984448e-03]
        T_em (float): Temperature of emitter in Kelvin [1414 + 273.15]
        phi_em (float): Resistivity of the emitter side wiring in ohm*cm [2.174]
        T_coll (float): Temperature of collector in Kelvin (50 + 273.15)
        phi_coll (float): Collector work function, eV [.381]
        rho_cw (float): Collector side wiring resistivity, ohms*cm [1.1984448e-03]
        gap_distance (float): Distance from the emitter to the collector in meters [1e-6]
        rho_load (float): Load resistivity, ohms*cm. Default value matched for
            `phi_em` = 2.174, `phi_coll` = 0.381, and
            `rho_cw` = `rho_ew` = 1.1984448e-03 [.01648048]
        run_id (str): Will be added to diagnostic folder name. Mainly
            used for parallel optimization [0]
        injection_type (int):
            1. For constant current emission with only thermal
               velocity spread in z and CL limited emission
            2: For true thermionic emission. Velocity spread
               along all axes [2]
        random_seed (bool): If True, will force a random seed to be
            used for emission positions [True]
        install_grid (bool): If False the grid will not be installed.
            Results in simple parallel plate setup. If True then
            `phi_em` - `phi_coll` specifies the voltage on the
            collector. [False]
        max_wall_time (float): Wall time to allow simulation to run for.
            Simulation is periodically checked and will halt if it appears
            the next segment of the simulation will exceed max_wall_time.
            This is not guaranteed to work since the guess is based on the
            run time up to that point. Intended to be used when running on
            system with job manager. [1e9]
        particle_diagnostic_switch (bool): Use openPMD compliant .h5 particle
            diagnostics [True]
        field_diagnostic_switch (bool): Use rswarp electrostatic .h5 field
            diagnostics (Maybe openPMD compliant?) [False]
        lost_diagnostic_switch (bool): Enable collection of lost particle coordinates
            with rswarp.diagnostics.parallel.save_lost_particles.[False]
    """
    from rswarp.run_files.tec import gridded_tec_3d

    gridded_tec_3d.main(
        x_struts,
        y_struts,
        V_grid,
        grid_height,
        strut_width,
        strut_height,
        rho_ew,
        T_em,
        phi_em,
        T_coll,
        phi_coll,
        rho_cw,
        gap_distance,
        rho_load,
        run_id,
        injection_type,
        random_seed,
        install_grid,
        max_wall_time,
        particle_diagnostic_switch,
        field_diagnostic_switch,
        lost_diagnostic_switch,
    )
