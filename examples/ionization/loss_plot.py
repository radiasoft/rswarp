import h5py as h5
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, physical_constants, m_e, e

loss_plot, ax1 = plt.subplots(1, 1, figsize=(8, 4))
# exlicitly make loss_plot the active figure:
plt.figure(loss_plot.number)
ax1.set_title('particle end losses vs. time')
ax1.set_xlabel('time step')
ax1.set_ylabel('number of lost particles')

f = h5.File('/home/vagrant/jupyter/runs/rswarp002/diags/crossing_record.h5', 'r')
#print list(f.keys())
dset = f['left']['e']
#dump_times = map(int, list(dset.keys()))
dump_times = [int(i) for i in list(dset.keys())]
dump_times.sort()
#print dump_times
number_of_lost = []
for n in dump_times:
    number_of_lost.append(dset[str(n)].size)

ax1.plot(dump_times, number_of_lost, 'r', label = 'electrons left')

dset = f['left']['h']
i = 0
for n in dump_times:
    number_of_lost[i] = dset[str(n)].size
    i += 1
ax1.plot(dump_times, number_of_lost, 'b', label = 'ions left')

dset = f['right']['e']
i = 0
for n in dump_times:
    number_of_lost[i] = dset[str(n)].size
    i += 1
ax1.plot(dump_times, number_of_lost, 'm', label = 'electrons right')

dset = f['right']['h']
i = 0
for n in dump_times:
    number_of_lost[i] = dset[str(n)].size
    i += 1
ax1.plot(dump_times, number_of_lost, 'g', label = 'ions right')

loss_plot.legend(prop = {'size': 10}, loc = 'upper right')
loss_plot.tight_layout()
loss_plot.savefig('loss_plot.png')
#plt.show()
plt.close()

f.close()

#f = h5py.File('diags/hdf5/data00003000.h5', 'r')
#x = f['data/3000/particles/emitted e-/position/x'][:] * 1.0e2
#x = f['data/3000/particles/H2+/position/x'][:]
