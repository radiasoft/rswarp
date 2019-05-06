import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

loss_plot, ax1 = plt.subplots(1, 1, figsize=(8, 4))
# exlicitly make loss_plot the active figure:
plt.figure(loss_plot.number)
ax1.set_title('particle end losses vs. time')
ax1.set_xlabel('time step')
ax1.set_ylabel('number of lost particles')

with open('loss_hist.txt', 'r') as flh:
    columns = flh.readline().strip().split()
    nhist = int(columns[0])
    data = np.empty((5, nhist))
    # Read data
    for n in range(nhist):
        columns = flh.readline().strip().split()
        for i in range(5):
            data[i, n] = int(columns[i])

ax1.plot(data[0, :], data[1, :], 'r', label = 'electrons left')
ax1.plot(data[0, :], data[2, :], 'b', label = 'ions left')
ax1.plot(data[0, :], data[3, :], 'm', label = 'electrons right')
ax1.plot(data[0, :], data[4, :], 'g', label = 'ions right')

#ax1.legend(prop = {'size': 10}, loc = 'center')
ax1.legend(prop = {'size': 10}, loc = 'upper right')
ax1.grid(True)

#loss_plot.legend(prop = {'size': 10}, loc = 'upper left')
loss_plot.tight_layout()
loss_plot.savefig('loss_plot.png')
#plt.show()
plt.close()
