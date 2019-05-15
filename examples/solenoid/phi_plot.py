import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

Nt = 60000

f = h5py.File('diags/fields/electric/data' + str(Nt) + '.h5', 'r')
#print f.keys()
phi = f['/data/' + str(Nt) + '/meshes/phi'][0, :, :]
Er = f['/data/' + str(Nt) + '/meshes/E/r'][0, :, :] * 1.0e2 * 1.0e-3
Ez = f['/data/' + str(Nt) + '/meshes/E/z'][0, :, :] * 1.0e2 * 1.0e-3

Nr = 40
Nz = 40000

#pipe_radius = 0.1524 / 2. * 100.
#pipe_length = 750.
pipe_radius = 3.
pipe_length = 3000.

dr = pipe_radius / Nr
dz = pipe_length / Nz

r = np.linspace(0.0, pipe_radius, Nr + 1)
#z = np.linspace(0.5 * dz, pipe_length - 0.5 * dz, Nz)
z = np.linspace(0.0, pipe_length, Nz + 1)

phi_plot, ax = plt.subplots(3, 1, figsize=(8, 12))
# exlicitly make phi_plot the active figure:
plt.figure(phi_plot.number)

#ax[0].set_title('particle end losses vs. time')
ax[0].set_xlabel('axial location (cm)')
ax[0].set_ylabel('electrostatic potential (V)')
ax[0].plot(z, phi[0, :], 'r', label = 'on axis')
ax[0].plot(z, phi[20, :], 'b', label = 'halfway out')
ax[0].plot(z, phi[40, :], 'g', label = 'at edge')
ax[0].plot(z, phi[38, :], 'c', label = 'almost at edge')
ax[0].legend(prop = {'size': 10}, loc = 'upper center')

ax[1].set_xlabel('axial location (cm)')
ax[1].set_ylabel('axial electric field (kV/cm)')
ax[1].plot(z, Ez[0, :], 'r', label = 'on axis')
ax[1].plot(z, Ez[20, :], 'b', label = 'halfway out')
ax[1].plot(z, Ez[40, :], 'g', label = 'at edge')
ax[1].plot(z, Ez[38, :], 'c', label = 'almost at edge')
ax[1].legend(prop = {'size': 10}, loc = 'upper center')

ax[2].set_xlabel('radial location (cm)')
ax[2].set_ylabel('radial electric field (kV/cm)')
ax[2].plot(r, Er[:, Nz//2], 'r', label = 'middle')
ax[2].plot(r, Er[:, 1], 'b', label = 'left')
ax[2].plot(r, Er[:, -2], 'g', label = 'right')
ax[2].legend(prop = {'size': 10}, loc = 'lower right')

phi_plot.tight_layout()
phi_plot.savefig('phi_plot-' + str(Nt) + '.png')
#plt.show()
plt.close()
