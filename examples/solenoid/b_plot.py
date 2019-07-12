import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

bx = np.load('bx.npy')
bx_shape = bx.shape
by = np.load('by.npy')
bz = np.load('bz.npy')

print bx_shape

#sys.exit(0)

pipe_radius = 3.
pipe_length = 3000.

Nr = bx_shape[0]
Nz = bx_shape[2]

dr = pipe_radius / Nr
dz = pipe_length / Nz

#r = np.linspace(0.0, pipe_radius, Nr + 1)
#z = np.linspace(0.0, pipe_length, Nz + 1)
r = np.linspace(0.5 * dr - pipe_radius, pipe_radius - 0.5 * dr, Nr)
z = np.linspace(0.5 * dz, pipe_length - 0.5 * dz, Nz)

b_plot, ax = plt.subplots(3, 2, figsize=(8, 12))
# exlicitly make b_plot the active figure:
plt.figure(b_plot.number)

#ax[0, 0].set_title('particle end losses vs. time')
ax[0, 0].set_xlabel('axial location (cm)')
ax[0, 0].set_ylabel(r'$B_x$ (T)')
ax[0, 0].plot(z[:100], bx[Nr // 2, Nr // 2, :100], 'r', label = 'on axis')
ax[0, 0].plot(z[:100], bx[Nr // 4, Nr // 2, :100], 'b', label = 'halfway out in x')
#ax[0, 0].plot(z[:100], bx[Nr // 2, Nr // 4, :100], 'g', label = 'halfway out in y')
ax[0, 0].plot(z[:100], bx[0, Nr // 2, :100], 'c', label = 'at x edge')
#ax[0, 0].plot(z[:100], bx[Nr // 2, 0, :100], 'm', label = 'at y edge')
ax[0, 0].legend(prop = {'size': 10}, loc = 'lower right')

ax[0, 1].set_xlabel('radial location (cm)')
ax[0, 1].set_ylabel(r'$B_x$ (T)')
ax[0, 1].plot(r, bx[:, Nr // 2, 15], 'r', label = 'vs x')
ax[0, 1].plot(r, bx[13, :, 15], 'b', label = 'vs y at x = 13')
ax[0, 1].plot(r, bx[19, :, 15], 'g', label = 'vs y at x = 19')
ax[0, 1].legend(prop = {'size': 10}, loc = 'upper left')

ax[1, 0].set_xlabel('axial location (cm)')
ax[1, 0].set_ylabel(r'$B_y$ (T)')
ax[1, 0].plot(z[:100], by[Nr // 2, Nr // 2, :100], 'r', label = 'on axis')
ax[1, 0].plot(z[:100], by[Nr // 4, Nr // 2, :100], 'b', label = 'halfway out in x')
ax[1, 0].plot(z[:100], by[0, Nr // 2, :100], 'c', label = 'at x edge')
ax[1, 0].legend(prop = {'size': 10}, loc = 'lower right')

ax[1, 1].set_xlabel('radial location (cm)')
ax[1, 1].set_ylabel(r'$B_y$ (T)')
ax[1, 1].plot(r, by[:, Nr // 2, 15], 'r', label = 'vs x')
ax[1, 1].plot(r, by[13, :, 15], 'b', label = 'vs y at x = 13')
ax[1, 1].plot(r, by[19, :, 15], 'g', label = 'vs y at x = 19')
ax[1, 1].legend(prop = {'size': 10}, loc = 'upper left')

ax[2, 0].set_xlabel('axial location (cm)')
ax[2, 0].set_ylabel(r'$B_z$ (T)')
ax[2, 0].plot(z[:100], bz[Nr // 2, Nr // 2, :100], 'r', label = 'on axis')
ax[2, 0].plot(z[:100], bz[Nr // 4, Nr // 2, :100], 'b', label = 'halfway out in x')
ax[2, 0].plot(z[:100], bz[0, Nr // 2, :100], 'c', label = 'at x edge')
ax[2, 0].legend(prop = {'size': 10}, loc = 'lower right')

ax[2, 1].set_xlabel('radial location (cm)')
ax[2, 1].set_ylabel(r'$B_z$ (T)')
ax[2, 1].plot(r, bz[:, Nr // 2, 15], 'r', label = 'vs x')
ax[2, 1].plot(r, bz[13, :, 15], 'b', label = 'vs y at x = 13')
ax[2, 1].plot(r, bz[19, :, 15], 'g', label = 'vs y at x = 19')
ax[2, 1].legend(prop = {'size': 10}, loc = 'upper left')

b_plot.tight_layout()
b_plot.savefig('b_plot.png')
#plt.show()
plt.close()
