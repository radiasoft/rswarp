def main():
    Npart = 65536
    Nstep = 5000
    Nstep_tw = 3700  # number of loc-ns at which Courant-Snyder params are computed (interp-n in between)

    gamma0 = 42.66  # assumed exact
    sigma_gamma_over_gamma = 1.0e-3  # rms energy spread in the lab frame

    Z_ion = 79
    X_ion = np.array([0.0, 0.0, 0.0])
    # V_ion = np.array([0.0, 0.0, 0.0])
    coreSq = 1.0e-13  # m^2, a short-range softening parameter for the Coulomb potential: r^2 -> r^2 + coreSq

    L_mod = 3.7  # m, modulator section length in the lab frame

    I_el = 100.  # A, e-beam current
    z_min_lab = -2.0e-5  # m
    z_max_lab = 2.0e-5  # m

    z_min = gamma0 * z_min_lab  # in the beam frame
    z_max = gamma0 * z_max_lab  # beam frame

    # Initial Courant-Snyder parameters (beam at the waist initially):
    alpha_x_ini = 0.0
    alpha_y_ini = 0.0
    beta_x_ini = 4.5  # m
    beta_y_ini = 4.5  # m

    eps_n_rms_x = 5.0e-6  # m-rad, normalized rms emittance
    eps_n_rms_y = 5.0e-6  # m-rad, normalized rms emittance

    # separate RNG seeds for different phase space coords, so that increasing the number
    # of particles from N1 to N2 > N1 we get the same initial N1 particles among the N2
    rng_seed_x = 98765
    rng_seed_y = 87654
    rng_seed_z = 76543
    rng_seed_vx = 65432
    rng_seed_vy = 54321
    rng_seed_vz = 43210

    quad_grad = np.array([0.0, 0.0, 0.0, 0.0])  # quad field gradients (assuming 4 quads)

    # ------------------------------------------------------------------------------

    q_el = -1.6021766208e-19  # Coulombs
    m_el = 9.10938356e-31  # kg
    c0 = 299792458.  # m/s
    # r_cl_el = 2.8179403227e-15  # m, classical radius of electron
    q_o_m = q_el / m_el

    # ------------------------------------------------------------------------------

    t0 = time.time()

    beta0 = np.sqrt(1. - 1. / (gamma0 * gamma0))

    T_mod = L_mod / (gamma0 * beta0 * c0)  # sim time in the _beam_ frame
    dt = T_mod / np.float64(Nstep)

    ds_tw = L_mod / np.float64(Nstep_tw)
    s_tw, tw_bx_of_s, tw_by_of_s, tw_ax_of_s, tw_ay_of_s = compute_twiss_drift(L_mod, Nstep_tw, gamma0)

    Q_tot = -1.0 * I_el * (z_max - z_min) / (gamma0 * beta0 * c0)  # z_max, z_min in the beam frame; -1.0 b/c an e-beam
    Q_per_mp = Q_tot / float(Npart)

    gamma_x_ini = (1. + alpha_x_ini * alpha_x_ini) / beta_x_ini
    gamma_y_ini = (1. + alpha_y_ini * alpha_y_ini) / beta_y_ini

    eps_rms_x = eps_n_rms_x / (gamma0 * beta0)
    eps_rms_y = eps_n_rms_y / (gamma0 * beta0)

    x_rms_ini = np.sqrt(eps_rms_x * beta_x_ini)
    y_rms_ini = np.sqrt(eps_rms_y * beta_y_ini)
    xp_rms_ini = np.sqrt(eps_rms_x * gamma_x_ini)  # in the lab frame, of course
    yp_rms_ini = np.sqrt(eps_rms_y * gamma_y_ini)
    vx_rms_ini = gamma0 * beta0 * c0 * xp_rms_ini  # in the _beam_ frame
    vy_rms_ini = gamma0 * beta0 * c0 * yp_rms_ini  # in the _beam_ frame

    vz_rms_ini = beta0 * c0 * sigma_gamma_over_gamma  # in the _beam_ frame

    rand1 = np.random.RandomState(rng_seed_x)
    rand2 = np.random.RandomState(rng_seed_y)
    rand3 = np.random.RandomState(rng_seed_z)
    rand4 = np.random.RandomState(rng_seed_vx)
    rand5 = np.random.RandomState(rng_seed_vy)
    rand6 = np.random.RandomState(rng_seed_vz)

    # The ion E-field is normalized outside the function that computes it
    k_Eion = 29.9792458 * Z_ion * np.abs(q_el)  # (1/c) * (1/(4 pi eps0)) *Z*|e| == (c0/10^7) *Z*|e|

    # ------------------------------------------------------------------------------

    # x, y, z: positions at steps n; xp1, yp1, zp1: positions at steps n + 1

    x = rand1.normal(0., x_rms_ini, Npart)
    x -= np.mean(x)
    x *= x_rms_ini / np.std(x)
    y = rand2.normal(0., y_rms_ini, Npart)
    y -= np.mean(y)
    y *= y_rms_ini / np.std(y)
    # z = rand3.uniform(z_min, z_max, Npart)
    dz = (z_max - z_min) / np.float64(Npart)
    z = np.linspace(z_min + 0.5 * dz, z_max - 0.5 * dz, Npart)

    # _beam-frame_ velocity components normalized to the speed of light (i.e., betas):
    # beam_x_minus, beam_y_minus, beam_z_minus at steps n - 1/2,  beam_x_plus, beam_y_plus, beam_z_plus at steps n + 1/2

    beam_x_minus = rand4.normal(0., vx_rms_ini, Npart)
    beam_x_minus -= np.mean(beam_x_minus)
    beam_x_minus *= vx_rms_ini / np.std(beam_x_minus)
    beam_x_minus /= c0
    beam_y_minus = rand5.normal(0., vy_rms_ini, Npart)
    beam_y_minus -= np.mean(beam_y_minus)
    beam_y_minus *= vy_rms_ini / np.std(beam_y_minus)
    beam_y_minus /= c0
    beam_z_minus = rand6.normal(0., vz_rms_ini, Npart)
    beam_z_minus -= np.mean(beam_z_minus)
    beam_z_minus *= vz_rms_ini / np.std(beam_z_minus)
    beam_z_minus /= c0
    np.save('initial_distribution.npy', [x, y , z, beam_x_minus, beam_y_minus, beam_z_minus])

    beam_x_plus = 0.0 * beam_x_minus
    beam_y_plus = 0.0 * beam_y_minus
    beam_z_plus = 0.0 * beam_z_minus

    # delta-f weights (one per delta-f particle): weight_minus at steps n - 1/2,  weight_plus at steps n + 1/2
    # the weights are initially zero because we use f and f_0 such that f(t=0) = f_0(t=0)
    weight_minus = 0.0 * x
    weight_plus = 0.0 * x

    # TODO: Need this advance
    # this may not be a great choice, but in this script t = 0 <--> n = -1/2, so n = 0 <--> t = 0.5*dt
    x += 0.5 * dt * c0 * beam_x_minus
    y += 0.5 * dt * c0 * beam_y_minus
    z += 0.5 * dt * c0 * beam_z_minus
    # periodic BCs in the z direction:
    z[z > z_max] = z_min + (z[z > z_max] - z_max)
    z[z < z_min] = z_max - (z_min - z[z < z_min])

    t = 0.5 * dt  # and n = 0

    t1 = time.time()

    print('Finished initialization. Time to initialize: ', t1 - t0, ' sec.\n')

    # ------------------------------------------------------------------------------

    print('Initial distribution parameters:')
    print('x_rms = ', x_rms_ini, 'm,  y_rms = ', y_rms_ini, ' m')
    print('vx_rms = ', vx_rms_ini, 'm/s  ', 'vx_rms_ini*T_mod / rms_x_ini = ', vx_rms_ini * T_mod / x_rms_ini)
    print('vy_rms = ', vy_rms_ini, 'm/s  ', 'vy_rms_ini*T_mod / rms_y_ini = ', vy_rms_ini * T_mod / y_rms_ini)
    print('vz_rms = ', vz_rms_ini, 'm/s  ', 'vz_rms_ini*T_mod / (z_max -z_min) = ', \
          vz_rms_ini * T_mod / (z_max - z_min))

    # keep record of the beam parameters as a function of time
    s_plot = gamma0 * beta0 * c0 * np.linspace(0.5 * dt, T_mod - 0.5 * dt, Nstep)
    x_rms = np.zeros(Nstep, dtype=np.float64)
    y_rms = 0.0 * x_rms
    x_rms[0] = np.std(x)
    y_rms[0] = np.std(y)

    s_vz = 0.0 * s_plot
    bz_mean_of_s = 0.0 * s_plot
    bz_rms_of_s = 0.0 * s_plot

    # ------------------------------------------------------------------------------

    #  leapfrog: the main loop
    np.save('distribution_pre_step_0.npy', [x, y , z, beam_x_minus, beam_y_minus, beam_z_minus])
    for istep in np.arange(Nstep - 1):

        # ION FIELDS
        Eix, Eiy, Eiz = E_ion(Npart, x, y, z, X_ion, coreSq)

        # velocities are non-relativistic in the beam frame
        s = gamma0 * beta0 * c0 * t  # s into the modulator, lab frame

        # UPDATE ELECTRON VELOCITIES
        beam_x_plus = beam_x_minus + dt * q_o_m * (-gamma0 * k_quad(s, quad_grad) * x * (beta0 + beam_z_minus) + k_Eion * Eix)
        beam_y_plus = beam_y_minus + dt * q_o_m * (gamma0 * k_quad(s, quad_grad) * y * (beta0 + beam_z_minus) + k_Eion * Eiy)
        beam_z_plus = beam_z_minus + dt * q_o_m * (gamma0 * k_quad(s, quad_grad) * (x * beam_x_minus - y * beam_y_minus) + k_Eion * Eiz)

        # interpolate the Twiss functions to the current location
        i_tw = int(np.modf(s / ds_tw)[1])
        fr_tw = np.modf(s / ds_tw)[0]

        tw_ax = fr_tw * tw_ax_of_s[i_tw + 1] + (1. - fr_tw) * tw_ax_of_s[i_tw]
        tw_ay = fr_tw * tw_ay_of_s[i_tw + 1] + (1. - fr_tw) * tw_ay_of_s[i_tw]
        tw_bx = fr_tw * tw_bx_of_s[i_tw + 1] + (1. - fr_tw) * tw_bx_of_s[i_tw]
        tw_by = fr_tw * tw_by_of_s[i_tw + 1] + (1. - fr_tw) * tw_by_of_s[i_tw]

        # FIRST WEIGHT UPDATE
        weight_plus += dt * q_o_m * (1. - weight_minus) * (tw_ax * x + tw_bx * beam_x_minus / (gamma0 * beta0)) * k_Eion * Eix \
              / (gamma0 * beta0 * eps_rms_x)
        weight_plus += dt * q_o_m * (1. - weight_minus) * (tw_ay * y + tw_by * beam_y_minus / (gamma0 * beta0)) * k_Eion * Eiy \
              / (gamma0 * beta0 * eps_rms_y)
        bzm_mean = np.mean(beam_z_minus)
        bzm_rms = np.std(beam_z_minus)
        s_vz[istep + 1] = s
        bz_mean_of_s[istep + 1] = bzm_mean
        bz_rms_of_s[istep + 1] = bzm_rms

        # weight_plus += dt*q_o_m*(1.-weight_minus)* (c0/vz_rms_ini)*(c0/vz_rms_ini)*(beam_z_minus-bzm_mean) *k_Eion*Eiz
        weight_plus += dt * q_o_m * (1. - weight_minus) * \
                       (1. / bzm_rms) * (1. / bzm_rms) * (beam_z_minus - bzm_mean) * k_Eion * Eiz

        weight_minus = 1.0 * weight_plus

        xp1 = x + dt * c0 * beam_x_plus
        yp1 = y + dt * c0 * beam_y_plus
        zp1 = z + dt * c0 * beam_z_plus

        beam_x_minus = 1.0 * beam_x_plus
        beam_y_minus = 1.0 * beam_y_plus
        beam_z_minus = 1.0 * beam_z_plus
        x = 1.0 * xp1
        y = 1.0 * yp1
        z = 1.0 * zp1

        # periodic BCs in the z direction:
        z[z > z_max] = z_min + (z[z > z_max] - z_max)
        z[z < z_min] = z_max - (z_min - z[z < z_min])

        x_rms[istep + 1] = np.std(x)
        y_rms[istep + 1] = np.std(y)

        t += dt
        if istep == 0:
            np.save('disitribution_end_of step_{}.npy'.format(istep),
                    [x, y, z, beam_x_minus, beam_y_minus, beam_z_minus])

    # final (half-)step to advance from t = T_mod -0.5*dt to t = T_mod
    Eix, Eiy, Eiz = E_ion(Npart, x, y, z, X_ion, coreSq)
    s = gamma0 * beta0 * c0 * t
    beam_x_plus = beam_x_minus + dt * q_o_m * (-gamma0 * k_quad(s, quad_grad) * x * (beta0 + beam_z_minus) + k_Eion * Eix)
    beam_y_plus = beam_y_minus + dt * q_o_m * (gamma0 * k_quad(s, quad_grad) * y * (beta0 + beam_z_minus) + k_Eion * Eiy)
    beam_z_plus = beam_z_minus + dt * q_o_m * (gamma0 * k_quad(s, quad_grad) * (x * beam_x_minus - y * beam_y_minus) + k_Eion * Eiz)

    i_tw = int(np.modf(s / ds_tw)[1])
    fr_tw = np.modf(s / ds_tw)[0]
    tw_ax = fr_tw * tw_ax_of_s[i_tw + 1] + (1. - fr_tw) * tw_ax_of_s[i_tw]
    tw_ay = fr_tw * tw_ay_of_s[i_tw + 1] + (1. - fr_tw) * tw_ay_of_s[i_tw]
    tw_bx = fr_tw * tw_bx_of_s[i_tw + 1] + (1. - fr_tw) * tw_bx_of_s[i_tw]
    tw_by = fr_tw * tw_by_of_s[i_tw + 1] + (1. - fr_tw) * tw_by_of_s[i_tw]

    weight_plus += dt * q_o_m * (1. - weight_minus) * (tw_ax * x + tw_bx * beam_x_minus / (gamma0 * beta0)) * k_Eion * Eix \
          / (gamma0 * beta0 * eps_rms_x)
    weight_plus += dt * q_o_m * (1. - weight_minus) * (tw_ay * y + tw_by * beam_y_minus / (gamma0 * beta0)) * k_Eion * Eiy \
          / (gamma0 * beta0 * eps_rms_y)
    bzm_mean = np.mean(beam_z_minus)
    bzm_rms = np.std(beam_z_minus)
    weight_plus += dt * q_o_m * (1. - weight_minus) * (1. / bzm_rms) * (1. / bzm_rms) * (beam_z_minus - bzm_mean) * k_Eion * Eiz
    weight_minus = 1.0 * weight_plus

    dt *= 0.5
    xp1 = x + dt * c0 * beam_x_plus
    yp1 = y + dt * c0 * beam_y_plus
    zp1 = z + dt * c0 * beam_z_plus
    beam_x_minus = 1.0 * beam_x_plus
    beam_y_minus = 1.0 * beam_y_plus
    beam_z_minus = 1.0 * beam_z_plus
    x = 1.0 * xp1
    y = 1.0 * yp1
    z = 1.0 * zp1
    # periodic BCs in the z direction:
    z[z > z_max] = z_min + (z[z > z_max] - z_max)
    z[z < z_min] = z_max - (z_min - z[z < z_min])
    t += dt
    # print 's = ', gamma0*beta0*c0*t  # = L_mod = 3.7

    t2 = time.time()
    print('Finished tracking. Time to track: ', t2 - t1, ' sec\n')

    # ----------------------------------------------------------------

    print('Final x and y rms sizes (in m): ', x_rms[Nstep - 1], y_rms[Nstep - 1], '\n')

    # plt.figure(figsize=(12.5, 10))
    #
    # plt.subplot(2, 2, 3)
    # plt.plot(s_tw, tw_bx_of_s, 'g-', label=r'$\beta_x$')
    # plt.plot(s_tw, tw_by_of_s, 'b-', label=r'$\beta_y$')
    # plt.legend(loc='upper right')
    # plt.xlim(0.0, L_mod)
    # plt.xlabel(r'$s (m)$')
    # plt.ylabel(r'$\beta_x, \beta_y (m)$')
    # plt.title(r'Twiss $\beta_x$, $\beta_y$')
    #
    # plt.subplot(2, 2, 4)
    # plt.plot(s_tw, tw_ax_of_s, 'r-', label=r'$\alpha_x$')
    # plt.plot(s_tw, tw_ay_of_s, 'g-', label=r'$\alpha_y$')
    # plt.legend(loc='upper right')
    # plt.xlim(0.0, L_mod)
    # plt.xlabel(r'$s (m)$')
    # plt.ylabel(r'$\alpha_x, \alpha_y$')
    # plt.title(r'Twiss $\alpha_x$, $\alpha_y$')
    #
    # Nbins = 40
    # bins = np.linspace(z_min, z_max, Nbins + 1)
    # bw = (z_max - z_min) / np.float64(Nbins)
    # binsZ = (bins[0:bins.size - 1] + 0.5 * bw)  # / ( 0.5*(z_max -z_min) )
    #
    # wHist = np.histogram(z, bins, weights=weight_minus)[0]
    # partCount = np.histogram(z, bins, weights=None)[0]
    #
    # plt.subplot(2, 2, 1)
    # # plt.xlim(-0.6*(z_max -z_min), 0.6*(z_max -z_min))
    # plt.xlabel(r'$z (m)$')
    # plt.ylabel(r'$\delta \rho / \rho$')
    # plt.title(r'Final longitudinal modulation')
    # plt.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
    # plt.plot(binsZ, wHist / partCount, 'r-')
    # plt.plot(binsZ, 0.0 * wHist / partCount, 'k-')
    #
    # plt.subplot(2, 2, 2)
    # # plt.xlim(-0.6*(z_max -z_min), 0.6*(z_max -z_min))
    # plt.xlabel(r'$z (m)$')
    # plt.ylabel(r'N per bin')
    # plt.title(r'Final longit. un-weighed part. count')
    # plt.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
    # plt.plot(binsZ, partCount, 'b-')
    # plt.plot(binsZ, 0.0 * partCount, 'k-')

    # plt.show()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def k_quad(s, quad_grad):  # s in the lab frame
    kq = 0.0
    return -kq


def E_ion(Np, x, y, z, X_ion, coreSq=1.0e-13):
    # Np = np.shape(x)[0]

    Ei_ion = np.zeros((Np, 3), dtype=np.float64)
    Ei_ion[:, 0] = x[:] - X_ion[0]  # positive ion
    Ei_ion[:, 1] = y[:] - X_ion[1]
    Ei_ion[:, 2] = z[:] - X_ion[2]
    r3 = np.zeros(Np, dtype=np.float64)
    r3[:] = np.power(np.sqrt(Ei_ion[:, 0] * Ei_ion[:, 0] + Ei_ion[:, 1] * Ei_ion[:, 1] \
                             + Ei_ion[:, 2] * Ei_ion[:, 2] + coreSq), 3)
    Ei_ion[:, 0] = Ei_ion[:, 0] / r3[:]
    Ei_ion[:, 1] = Ei_ion[:, 1] / r3[:]
    Ei_ion[:, 2] = Ei_ion[:, 2] / r3[:]
    # return Ei_ion  #  NB: un-normalized
    return Ei_ion[:, 0], Ei_ion[:, 1], Ei_ion[:, 2]


def compute_twiss_drift(L_mod, Nstep_tw, gamma0):
    q_el = -1.6021766208e-19  # Coulombs
    m_el = 9.10938356e-31  # kg
    c0 = 299792458.  # m/s
    beta0 = np.sqrt(1. - 1. / (gamma0 * gamma0))

    invBrho = q_el / (gamma0 * beta0 * m_el * c0)
    k2 = invBrho * np.array([0.0, 0.0, 0.0, 0.0])  # (this is kx[s], assuming 4 quads)

    s = np.linspace(0.0, L_mod, Nstep_tw + 1)
    dsmax = 0.0001

    # Initial conditions: same for x and y, assuming a round beam; a0 = 0 if at a waist:
    a0 = 0.0
    b0 = 4.5
    y0 = [a0, b0]

    # Bundle parameters for the ODE solver:
    # x trace space:
    params = (k2[0], k2[1], k2[2], k2[3])
    soln_x = odeint(dfdt_drift, y0, s, args=(params,), hmax=dsmax)

    # y trace space:
    params = (-k2[0], -k2[1], -k2[2], -k2[3])
    soln_y = odeint(dfdt_drift, y0, s, args=(params,), hmax=dsmax)

    return s, soln_x[:, 1], soln_y[:, 1], soln_x[:, 0], soln_y[:, 0]


def dfdt_drift(y, s, params):
    a, b = y

    # kk2 = params
    # v0, v1, v2, v3 = params
    quadgr = 0.0

    # emittance-dominated beam
    derivs = np.array([-(1. + a * a) / b + quadgr * b, -2. * a])

    return derivs


if __name__ == "__main__":
    import numpy as np
    # import matplotlib.pyplot as plt
    import time
    from scipy.integrate import odeint

    import cProfile

    main()

    # cProfile.run('main()')

