import unittest
import numpy as np
import rswarp.run_files.delta_f.delta_f_tools as dft
from scipy.constants import c as c0

Npart = 65536
gamma0 = 42.66  # assumed exact
sigma_gamma_over_gamma = 1.0e-3  # rms energy spread in the lab frame
z_min_lab = -2.0e-5  # m
z_max_lab = 2.0e-5  # m
z_min = gamma0 * z_min_lab  # in the beam frame
z_max = gamma0 * z_max_lab  # beam frame

alpha_x_ini = 0.0
alpha_y_ini = 0.0
beta_x_ini = 4.5  # m
beta_y_ini = 4.5  # m

eps_n_rms_x = 5.0e-6  # m-rad, normalized rms emittance
eps_n_rms_y = 5.0e-6  # m-rad, normalized rms emittance

beta0 = np.sqrt(1. - 1. / (gamma0 * gamma0))

eps_rms_x = eps_n_rms_x / (gamma0 * beta0)
eps_rms_y = eps_n_rms_y / (gamma0 * beta0)

gamma_x_ini = (1. + alpha_x_ini * alpha_x_ini) / beta_x_ini
gamma_y_ini = (1. + alpha_y_ini * alpha_y_ini) / beta_y_ini

x_rms_ini = np.sqrt(eps_rms_x * beta_x_ini)
y_rms_ini = np.sqrt(eps_rms_y * beta_y_ini)
xp_rms_ini = np.sqrt(eps_rms_x * gamma_x_ini)  # in the lab frame, of course
yp_rms_ini = np.sqrt(eps_rms_y * gamma_y_ini)
vx_rms_ini = gamma0 * beta0 * c0 * xp_rms_ini  # in the _beam_ frame
vy_rms_ini = gamma0 * beta0 * c0 * yp_rms_ini  # in the _beam_ frame
vz_rms_ini = beta0 * c0 * sigma_gamma_over_gamma  # in the _beam_ frame
transverse_sigmas = [x_rms_ini, y_rms_ini, vx_rms_ini, vy_rms_ini]


class TestDistribution(unittest.TestCase):
    # Test distribution before any updates have been applied
    def setUp(self):
        self.ip_distribution = np.load('./delta_f_files/initial_distribution.npy')
        self.measured_sigmas = np.std(self.ip_distribution, axis=1)

    def test_x(self):
        check = np.isclose(self.measured_sigmas[0], transverse_sigmas[0])
        self.assertTrue(check)

    def test_y(self):
        check = np.isclose(self.measured_sigmas[1], transverse_sigmas[1])
        self.assertTrue(check)

    def test_z(self):
        dz = (z_max - z_min) / np.float64(Npart)
        z = np.linspace(z_min + 0.5 * dz, z_max - 0.5 * dz, Npart)
        check = np.isclose(self.ip_distribution[2, :], z)
        self.assertTrue(np.all(check))

    def test_vx(self):
        check = np.isclose(self.measured_sigmas[3], transverse_sigmas[2] / c0)
        self.assertTrue(check)

    def test_vy(self):
        check = np.isclose(self.measured_sigmas[4], transverse_sigmas[3] / c0)
        self.assertTrue(check)

    def test_vz(self):
        check = np.isclose(self.measured_sigmas[5], vz_rms_ini / c0)
        self.assertTrue(check)


class TestDistributionCreation(unittest.TestCase):
    # Verify rswarp distribution matches up against the I.P.'s delta-f benchmark
    def setUp(self):
        self.ip_distribution = np.load('./delta_f_files/initial_distribution.npy').T
        seeds = [98765, 87654, 76543 ,65432 ,54321 ,43210]
        self.new_distr = dft.create_distribution(Npart=Npart, transverse_sigmas=transverse_sigmas,
                                                 length=z_max-z_min, z_sigma=vz_rms_ini, seeds=seeds)

    def test_x(self):
        check = self.ip_distribution[:, 0] - self.new_distr[:, 0]
        self.assertFalse(np.all(check))

    def test_y(self):
        check = self.ip_distribution[:, 1] - self.new_distr[:, 1]
        self.assertFalse(np.all(check))

    def test_z(self):
        check = self.ip_distribution[:, 2] - self.new_distr[:, 2]
        self.assertFalse(np.all(check))

    def test_vx(self):
        check = self.ip_distribution[:, 3] - self.new_distr[:, 3]
        self.assertFalse(np.all(check))

    def test_vy(self):
        check = self.ip_distribution[:, 4] - self.new_distr[:, 4]
        self.assertFalse(np.all(check))

    def test_vz(self):
        check = self.ip_distribution[:, 5] - self.new_distr[:, 5]
        self.assertFalse(np.all(check))

