import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

Nt = 1000000
Nr = 128
Nz = 64

pipe_radius = 0.1524 / 2. * 100.
pipe_length = 200.

dr = pipe_radius / Nr
dz = pipe_length / Nz

dens = np.empty((Nr, Nz))
tot_e_dens_r = np.empty((Nr))
tot_e_dens_z = np.empty((Nz))

density_plot, ax = plt.subplots(1, 2, figsize=(12, 6))
# exlicitly make density_plot the active figure:
plt.figure(density_plot.number)
plt.suptitle('Particle densitites after ' + str(Nt) + ' time steps with drain electrodes at +20 V')
#plt.subplots_adjust(hspace=0.4)

f = h5py.File('diags/hdf5/data0' + str(Nt) + '.h5', 'r')

species = ['Electron', 'emitted e-', 'H2+']

for n in range(3):

    x = f['/data/' + str(Nt) + '/particles/' + species[n] + '/position/x'][:] * 1.0e2
    y = f['/data/' + str(Nt) + '/particles/' + species[n] + '/position/y'][:] * 1.0e2
    z = f['/data/' + str(Nt) + '/particles/' + species[n] + '/position/z'][:] * 1.0e2
    w = f['/data/' + str(Nt) + '/particles/' + species[n] + '/weighting'][:]

    print species[n], ' weight mean (and standard deviation): ', np.mean(w), ' (', np.std(w), ')'

    dens[:, :] = 0.
    for i in range(x.size):
        ir = int(math.sqrt(x[i] * x[i] + y[i] * y[i]) / dr)
        if ir > Nr - 1:
            ir = Nr - 1
        iz = int(z[i] / dz)
        if iz > Nz - 1:
            iz = Nz - 1
        dens[ir, iz] += w[i]
        if i % 1000000 == 0:
            print i, ir, iz, x[i], y[i], z[i]
    for ir in range(Nr):
        dens[ir, :] /= 2. * math.pi * (ir + .5) * dr * dr * dz
#        for iz in range(Nz):
#            dens[ir, iz] /= 2. * math.pi * (ir + .5) * dr * dr * dz
    dens[:, :] *= 1.e-6
    dens[0, 0] = 0.

    #ax[0].set_title(species[n])
    ax[0].set_xlabel('radial location (cm)')
    ax[0].set_ylabel('density (10^6/cm^3)')
    ax[0].set_xlim(0., pipe_radius)
    ax[0].plot(np.linspace(0., pipe_radius, Nr), dens[:, Nz / 2], label = species[n])

    if n == 0:
        tot_e_dens_r[:] = dens[:, Nz / 2]
        neb = np.sum(tot_e_dens_r)
    if n == 1:
        tot_e_dens_r[:] += dens[:, Nz / 2]
        net = np.sum(tot_e_dens_r)
        ax[0].plot(np.linspace(0., pipe_radius, Nr), tot_e_dens_r, label = 'electrons total')
    if n == 2:
        ni = np.sum(dens[:, Nz / 2])

    #ax[1].set_title(species[n])
    ax[1].set_xlabel('axial location (cm)')
    ax[1].set_ylabel('density (10^6/cm^3)')
    ax[1].set_xlim(0., pipe_length)
    ax[1].plot(np.linspace(0., pipe_length, Nz), dens[0, :], label = species[n])

    if n == 0:
        tot_e_dens_z[:] = dens[0, :]
    if n == 1:
        tot_e_dens_z[:] += dens[0, :]
        ax[1].plot(np.linspace(0., pipe_length, Nz), tot_e_dens_z, label = 'electrons total')

#density_plot.tight_layout()
ax[0].legend(prop = {'size': 10}, loc = 'upper right')
ax[1].legend(prop = {'size': 10}, loc = 'upper left')
density_plot.savefig('density_plot-' + str(Nt) + '.png')
plt.close()

print 'charge neutralization efficiency is ', 100. * (1. - (net - ni) / neb), '%'
