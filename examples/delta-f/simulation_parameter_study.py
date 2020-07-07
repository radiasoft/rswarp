#!/usr/bin/env python
# coding: utf-8

# In[2]:


from plasma import electric_field, normalized_velocity_kick, debye_length, plasma_frequency
import numpy as np
from scipy.constants import c, e, m_e, epsilon_0
from scipy.constants import Boltzmann as k_b


# ### Beam specifications

# In[3]:


# Relatistic quantities
gamma0 = 42.66
beta0 = np.sqrt(1. - 1. / (gamma0 * gamma0))


# #### Transverse Specs

# In[4]:


# Initial Courant-Snyder parameters (beam at the waist initially):
alpha_x_ini = 0.0
beta_x_ini = 4.5  # m
gamma_x_ini = (1. + alpha_x_ini * alpha_x_ini) / beta_x_ini

eps_n_rms_x = 5.0e-6  # m-rad, normalized rms emittance
eps_rms_x = eps_n_rms_x / (gamma0 * beta0)  # m-rad, geometric emittance


# In[ ]:


x_rms_ini = np.sqrt(eps_rms_x * beta_x_ini)  # m
xp_rms_ini = np.sqrt(eps_rms_x * gamma_x_ini)  # rms angular divergence, lab frame value
v_rms_transverse = gamma0 * beta0 * c * xp_rms_ini  # velocity spread in beam frame


# #### Longitudinal Specs

# In[5]:


# Lab frame quantities
slice_current = 100. # Amps from CeC PoP Exp
slice_length = 4e-6  # We only simulate a small slice around one ion, full bunch length would be ~100 mm


# In[6]:


sigma_gamma_over_gamma = 1.0e-3  # rms energy spread in the lab frame
vz_rms_ini = beta0 * c * sigma_gamma_over_gamma  # m/s


# ### Derived Beam Quantities

# In[8]:


electrons_in_slice = slice_current * (slice_length / c) / e  # True number (NOT macroparticles)


# In[37]:


# Use the same density everywhere right now
transverse_temperature_rms = 0.5 * m_e * (c*beta0*gamma0)**2 * np.sqrt(eps_rms_x / beta_x_ini)**2 / k_b
transverse_density_rms = electrons_in_slice / ((2 * np.pi)**(1.5) * x_rms_ini * x_rms_ini * (slice_length * gamma0) )

longitudinal_temperature_rms = 0.5 * m_e * c**2 * beta0**2 * sigma_gamma_over_gamma**2 / k_b
longitudinal_density_rms = transverse_density_rms


# In[38]:


transverse_debye_length = debye_length(transverse_temperature_rms, transverse_density_rms)
transverse_plasma_frequency = plasma_frequency(transverse_density_rms)
longitudinal_debye_length = debye_length(longitudinal_temperature_rms, longitudinal_density_rms)
longitudinal_plasma_frequency = plasma_frequency(longitudinal_density_rms)


# Longitudinal Debye length calculated in https://www.bnl.gov/isd/documents/86221.pdf
# is $\lambda_D$ = 42 $\mu m$

# In[62]:


print('Transverse Debye length (um): {:3.1f}'.format(transverse_debye_length * 1e6))
# print('Transverse Plasma period (ns): {:3.1f}'.format(2 * np.pi / transverse_plasma_frequency * 1e9))
print('Longitudinal Debye length (um): {:3.1f}'.format(longitudinal_debye_length * 1e6))
print('Plasma period (ns): {:3.1f}'.format(2 * np.pi / longitudinal_plasma_frequency * 1e9))


# In[70]:


avg_ee_distance = 1. / transverse_density_rms**(1/3.)


# In[71]:


print('Average inter-particle distance (um): {:3.1f}'.format(avg_ee_distance * 1e6))


# ### Test Modulator Parameters

# In[11]:


# Simulation Steps
Nsteps = 5000
L_mod = 3.7  # m, modulator section length in the lab frame
T_mod = L_mod / (gamma0 * beta0 * c)  # sim time in the _beam_ frame


# # Plotting

# In[12]:


import matplotlib.pyplot as plt
from matplotlib import ticker, colors


# In[13]:


get_ipython().run_line_magic('matplotlib', 'widget')


# In[72]:


plt.figure()

current_dt = T_mod / np.float64(Nsteps)
current_v_rms = v_rms_transverse
current_Q = 79
plasma_freq_dt = (2 * np.pi / longitudinal_plasma_frequency) / 20.


plt.title(r'$ \frac{dv_{e-}}{v_{e-}^{rms}} = \frac{q_e E_{ion}(r_0)}{v_{e-}^{rms} m_{e-}} dt$', fontsize=24)
# Values I have simulated
r_0_vals = np.linspace(1e-14, 1e-11, )

# Plot Gold for Warp timestep and value based on fraction of plasma period
plt.plot(r_0_vals * 1e12, normalized_velocity_kick(r_0_vals, current_dt, current_v_rms, current_Q), label=r'$Q_{ion}$=79,' +'dt={:2.1e}'.format(current_dt))
plt.plot(r_0_vals * 1e12, normalized_velocity_kick(r_0_vals, plasma_freq_dt, current_v_rms, current_Q), label=r'$Q_{ion}$=79,'+' dt={:2.1e}'.format(plasma_freq_dt))
plt.scatter(np.array([1e-13, 5e-15]) * 1e12, normalized_velocity_kick(np.array([1e-13, 5e-15]), current_dt, current_v_rms, current_Q))

# Protons for comparison
plt.plot(r_0_vals * 1e12, normalized_velocity_kick(r_0_vals, current_dt, current_v_rms, 1), label=r'$Q_{ion}$=1')
plt.yscale('log')
plt.ylabel(r'$dv_{e-}$ / $v_{e-}^{rms}$', fontsize=16)
plt.xlabel('$r_0$ (pm)', fontsize=16)
plt.legend()
plt.annotate('Current Warp dt={:2.2e} s'.format(current_dt), xy=(0.59, 0.68), xycoords='figure fraction')
plt.annotate(r'$v_{e-}^{rms}$='+'{:2.2e} m/s'.format(current_v_rms), xy=(0.59, 0.63), xycoords='figure fraction')
plt.show()


# # Quick, Order-of-magnitude Check on delta-f weight update limitation
# Not sure if this is correct or relevant. Assumes prior step weight update was 'small'. This calculation right now suggests weight update will always be small (<<1) for any sane time step.

# In[44]:


v_norm = v_rms_transverse * (1 / gamma0) / c / (gamma0 * beta0)


# In[47]:


(e / m_e) * (1 * x_rms_ini + v_norm * 1) * electric_field(1e-13, 79)*e / (gamma0 * beta0 * eps_n_rms_x)


# # Look at timestep / r0 limitations for drift case assuming typical cell crossing limits
# Order of magnitude study. Assumes $\beta_x$ ~ 20 m and uses a safety factor on max velocity of 6*$\sigma_{vx}$

# In[73]:


# cell size if Twiss beta goes up to 20
number_of_cells = 32
cell_size = np.sqrt(eps_rms_x * 20) / number_of_cells
# dt_min = cell_size / v_max


# Fix softening value $r_0$ then for a given timestep $dt$ the minimum time for any particle  to cross a cell of width $dx$ is $dt_{min}=\frac{dx}{6\sigma_{vx}+v_{max}^{\rm ion kick}(r_0, dt)}$
# 
# Plot $dt_{min}$ / $dt$ as we scan $dt$. Want this to stay less than 1 to prevent too many particles from crossing a cell in less than 1 timestep

# In[75]:


fig, ax1 = plt.subplots(1, 1)

current_dt = T_mod / np.float64(Nsteps)
current_v_rms = v_rms_transverse
current_Q = 79

dt_vals = np.linspace(10e-14, 3e-12, 500)

dt_ratio = cell_size / (current_v_rms * 6 + current_v_rms *normalized_velocity_kick(1e-13, dt_vals, current_v_rms, current_Q)) / dt_vals
ax1.plot(1e12*dt_vals, 1/dt_ratio, label=r'$r_0$ = 100 fm')

dt_ratio = cell_size / (current_v_rms * 6 + current_v_rms*normalized_velocity_kick(5e-15, dt_vals, current_v_rms, current_Q)) / dt_vals
ax1.plot(1e12*dt_vals, 1/dt_ratio, label=r'$r_0$ = 5 fm')

ax1.set_xlabel('dt (ps)')
ax1.set_ylabel(r'$\frac{dx}{v_{max}(r_0)}$ / dt')
ax1.legend()
                 
plt.show()


# Sweep over time step $dt$ and $r_0$ and look at requirement that $dv$ * $dt$ < $\lambda_D$
# 
# use longitudinal $\lambda_D$ as it is the smaller of transverse and longitudinal values

# In[78]:


fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

ax1.set_title(r'dv * dt  / $\lambda_D$')
current_dt = T_mod / np.float64(Nsteps)
current_v_rms = v_rms_transverse
current_Q = 79

r_0_vals = np.linspace(1e-14, 1e-11, 500)
dt_vals = np.linspace(3e-14, 1e-12, 500)
X, Y = np.meshgrid(r_0_vals, dt_vals)
Z = normalized_velocity_kick(X, Y, current_v_rms, current_Q) * current_v_rms * Y / longitudinal_debye_length
r0_contour = ax1.contourf(X * 1e12, Y * 1e12, Z, levels=64, locator=ticker.LogLocator(subs='all'))
ax1.set_xlabel(r'$r_0$ (pm)')
ax1.set_ylabel('dt (ps)')
fig.colorbar(r0_contour)

                 
plt.show()


# ## Work in progress  - logarithmic scales are not nice with contourf

# In[37]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

current_dt = T_mod / np.float64(Nsteps)
current_v_rms = v_rms_transverse
current_Q = 79

r_0_vals = np.linspace(1e-15, 1e-13, 500)
dt_vals = np.linspace(1e-15, 1e-12, 500)
X, Y = np.meshgrid(r_0_vals, dt_vals)

r0_contour = ax1.contourf(X, Y, normalized_velocity_kick(X, Y, current_v_rms, current_Q), levels=64, locator=ticker.LogLocator(subs='all'))
ax1.set_xlabel(r'$r_0$')
ax1.set_ylabel('time step')
fig.colorbar(r0_contour)
dt_ratio = dx / (current_v_rms * 6 + current_v_rms*normalized_velocity_kick(X, Y, current_v_rms, current_Q)) / Y
Z = 1/dt_ratio
lev_exp = np.linspace(np.floor(np.log10(Z.min())-1),
                   np.ceil(np.log10(Z.max())+1), 10)
level = np.power(10, lev_exp)
dt_contour = ax2.contourf(X, Y, 1/dt_ratio, level, norm=colors.LogNorm())
# dt_contour = ax2.contourf(X, Y, 1/dt_ratio, levels=64, locator=ticker.LogLocator(subs='all'))
fig.colorbar(dt_contour)
ax2.set_xlabel(r'$r_0$')
ax2.set_ylabel('time step')
                 
plt.show()


# In[38]:


lev_exp


# In[39]:


np.linspace(np.floor(np.log10(Z.min())-1),
                   np.ceil(np.log10(Z.max())+1), 10)


# In[ ]:




