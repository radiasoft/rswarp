'''
Electron Column
February 26, 2018
Ben Freemire
freeben@fnal.gov
'''

import numpy as np
from tables import *
from warp import *
from warp.run_modes.egun_like import *
from warp.utils.timedependentvoltage import TimeVoltage
from warp.particles.ionization import *
from warp.particles.extpart import ZCrossingParticles
from warp.data_dumping import PWpyt
import sys
#from Secondaries import *


# which geometry to use 2d or 3d
w3d.solvergeom = w3d.XYZgeom

#define some strings that go into the output file
top.pline1     = "Electron Column"
top.pline2     = "Save Beam Distribution"
top.runmaker   = "Ben Freemire (freeben@fnal.gov)"

# --- Invoke setup routine for the plotting
setup(makepsfile=1, cgmlog=1)

q = 1.6e-19 # define electric charge

# --- Set basic beam parameters
ibeaminit      =  8e-3   # in A, 8mA from HINS
ekininit       =  2.5e6  # in eV, 2.5 MeV HINS proton beam

# Scan parameters
Vbias = -10.   # Should be negative
Vbias_left  = -10.
Vbias_right = -10.

Bsol  = 0.1    # in Tesla

# --- Set input parameters describing the 3d simulation
top.dt = 350.*ps # 357 ps cyclotron period
numProt = 1e3
top.npinject = numProt # Number of macroparticles injected per time step

# --- weight factor for emitted macroparticles
# The Ionization function does not take into account Species weight when
# creating daughter products. We will adjust the daughter products' weights
# and cross section accordingly
Proton_weight = ibeaminit*top.dt/q/top.npinject # Number of real particles per
# macroparticle for protons
Wfact = Proton_weight/1000 # Number of real particles per macroparticle for
# electrons & ions = Proton_weight*Wfact

# relativistic correction
beta   = 0.0728557
fselfb_option = clight*beta

# --- define Species: Species class (defined in scripts/species.py).
protons = Species(type=Hydrogen,charge_state=+1,name='p',weight=Proton_weight)
electrons = Species(type=Electron,name='e-',weight=Proton_weight*Wfact)
h2plus = Species(type=Dihydrogen,charge_state=+1,name='H2+',
		 weight=Proton_weight*Wfact)
# This accounts for the increased weight of electrons and ions

#finject: species fraction for each source
top.finject[0, protons.jslist[0]] = 1
protons.ibeam = ibeaminit

# --- starting conditions for the ion and electron beam
top.ekin     = ekininit   # in eV
derivqty()

w3d.l4symtry = false  # four-fold symmetry
w3d.l2symtry = false  # two-fold symmetry

vz = top.vbeam # proton velocity

# --- Set boundary conditions

# ---   for field solve
w3d.bound0  = neumann #dirichlet
w3d.boundnz = neumann
w3d.boundxy = dirichlet #neumann

# ---   for particles
top.pbound0  = absorb
top.pboundnz = absorb
top.prwall   = 25.e-3 # 25 mm

le = 1.0 # m, length of the ecolumn length

zcross_pos=le
targetz_particles=ZCrossingParticles(zz=zcross_pos,laccumulate=1)

# --- Set field grid size
w3d.xmmin = -top.prwall
w3d.xmmax = +top.prwall
w3d.ymmin = -top.prwall
w3d.ymmax = +top.prwall
w3d.zmmin =  0.0-0.1
w3d.zmmax  = le +0.1  # 1-m-long ecolumn length

w3d.nx = 100
w3d.ny = 100
w3d.nz = 100

step_ini=int((w3d.zmmax/vz)/top.dt)

if w3d.l4symtry: w3d.xmmin = 0.
if w3d.l2symtry or w3d.l4symtry: w3d.ymmin = 0.

# set grid spacing
w3d.dx = (2.*top.prwall)/w3d.nx
w3d.dy = (2.*top.prwall)/w3d.ny
w3d.dz = (w3d.zmmax - w3d.zmmin)/w3d.nz

# --- Specify injection of the particles
top.inject = 1 # 0: no injection, 1: constant current, 2: space-charge limited injection

top.linj_efromgrid = true  # Turn on transverse E-fields near emitting surface
top.zinject = w3d.zmmin    # initial z of particle injection
top.ibpush   = 1           # Specifies type of B-advance; 0 - none, 1 - fast

source_radius = 5.5e-3 # 5.5 mm from MAD

top.ainject = source_radius
top.binject = source_radius
w3d.l_inj_user_particles_v = true

def nonlinearsource():
    if w3d.inj_js == protons.jslist[0]:
        nprot = top.npinject
        r = source_radius*random.random(nprot)
        theta = 2.*pi*random.random(nprot)
        x = r*cos(theta)
        y = r*sin(theta)
        w3d.npgrp = nprot
        gchange('Setpwork3d')
        w3d.xt[:] = x
        w3d.yt[:] = y
        w3d.uxt[:] = 0.
        w3d.uyt[:] = 0.
        w3d.uzt[:] = vz

installuserparticlesinjection(nonlinearsource)

# --- Select plot intervals, etc.
#top.nhist = 1 # Save history data every time step
#top.itplfreq[0:4]=[0,1000000,25,0] # Make plots every 25 time steps
#top.itmomnts[0:4]=[0,1000000,top.nhist,0] # Calculate moments every step

# --- Save time histories of various quantities versus z.
top.lhcurrz  = true
top.lhrrmsz  = true
top.lhxrmsz  = true
top.lhyrmsz  = true
top.lhepsnxz = true
top.lhepsnyz = true
top.lhvzrmsz = true

# --- Set up fieldsolver - 7 means the multigrid solver
top.fstype     = 7
f3d.mgtol      = 1.e-1 # Poisson solver tolerance, in volts
f3d.mgparam    =  1.5
f3d.downpasses =  2
f3d.uppasses   =  2

# --- Generate the PIC code (allocate storage, load ptcls, t=0 plots, etc.)
package("w3d")
generate()

#################################################################################################################

beampipe = ZCylinderOut(radius=top.prwall,zlower=w3d.zmmin, zupper=w3d.zmmax, 
                        voltage= 0.0, xcent=0, ycent=0, zcent=0)
electrode_left = ZCylinderOut(radius=top.prwall*0.9, zlower=0.-0.05, 
                              zupper=0.+0.05, voltage=Vbias_left, xcent=0, 
                              ycent=0, zcent=0)
electrode_right = ZCylinderOut(radius=top.prwall*0.9, zlower=le-0.05, 
                               zupper=le+0.05, voltage=Vbias_right, xcent=0, 
                               ycent=0, zcent=0)

installconductors(beampipe)
installconductors(electrode_left + electrode_right)
addsolenoid(zi=0., zf = le, ri=top.prwall, maxbz= Bsol) 

# --- Recalculate the fields
fieldsolve(-1)

target_density = 3.54e19 #1.e-3 torr @ 293K

# --- setup the charge exchange
ioniz = Ionization(stride=1, zmin=0, zmax=le)

# ------------p ionziation: p + H2 -> p + e + H2+
ioniz.add(protons, emitted_species=[h2plus,electrons], 
	  cross_section=1.82e-21/Wfact, ndens=target_density)
# The cross section is adjusted based on the weight of the electrons & ions

# ------ electron ionization e + H2 -> H2+ + e + e
#ioniz.add_ionization(incident_species=electrons, emitted_species=[h2plus,electrons], cross_section=1.3e-20, ndens = target_density)

#nx integer: mesh points are 0,...,nx
ixcenter = int(w3d.nx/2)
iycenter = int(w3d.ny/2)
izcenter = int(w3d.nz/2)

xcoord = zeros(w3d.nx+1,float)
for k in range(0, w3d.nx+1):
    xcoord[k] = w3d.xmmin + k * w3d.dx

zcoord = zeros(w3d.nz+1,float)
for k in range(0, w3d.nz+1):
    zcoord[k] = w3d.zmmin + k * w3d.dz

eden_time = []
col_time = []

save_repetition = 60
plot_repetition = 600
final_iter = 6000


for iter in range(0, final_iter+1):

    ###########################################################################
    #   plotting in ps format
    ###########################################################################
    if(iter % plot_repetition == 0):

        protons.ppzx(color=red, titles = 0, view=9, 
                     pplimits=(w3d.zmmin,w3d.zmmax,-top.prwall,top.prwall))
        pfzx(fill=1,filled=0, plotsg=0,titles = 0, cond=0,view=9)

        h2plus.ppzx(color=blue, titles = 0,msize=100, view=10)
        electrons.ppzx(color=green, titles = 0, msize=100, view=10, 
                       pplimits=(w3d.zmmin, w3d.zmmax, -top.prwall, top.prwall))
        fma()

        iden  = protons.get_density()
        eden  = electrons.get_density()
        hden  = h2plus.get_density()
        
        #ptitles(): draw plot titles on the current frame
        ptitles(titlet="Density (#/m3) at x=y=0", titleb="Z (m)", titlel=" ")
        limits(w3d.zmmin, w3d.zmmax)
        pla(iden[ixcenter,iycenter,0:], zcoord, color = red) #protons=ions=red
        pla(eden[ixcenter,iycenter,0:], zcoord, color = green) #electrons=green
        pla(hden[ixcenter,iycenter,0:], zcoord, color = blue) #h2+=blue
        fma()

        ptitles(titlet="Density (#/m3) at ecolumn center", titleb="X (m)", 
                titlel=" ")
        limits(w3d.xmmin, w3d.xmmax)
        pla(iden[0:,iycenter,izcenter], xcoord, color = red)
        pla(eden[0:,iycenter,izcenter], xcoord, color = green)
        pla(hden[0:,iycenter,izcenter], xcoord, color = blue)
        fma()

        # longitudinal electric fields
        ez = getselfe(comp="z", ix = ixcenter,  iy = iycenter)
        ptitles(titlet="Self electric fields (V/m) along the beam", 
                titleb="Z (m)", titlel=" ")
        limits(w3d.zmmin, w3d.zmmax)
        pla(ez, zcoord,  color= red)
        fma()

        # transverse electric fields
        ex = getselfe(comp="x", iy = iycenter,  iz = izcenter)
        ptitles(titlet="Self electric fields (V/m) along the beam", 
                titleb="X (m)", titlel =" ")
        limits(w3d.xmmin, w3d.xmmax)
        pla(ex, xcoord,  color= red)
        fma()

        eden_time.append((eden[ixcenter,iycenter,izcenter]+
                        eden[ixcenter,iycenter,izcenter-1]+
                        eden[ixcenter,iycenter,izcenter+1])/3.)
        col_time.append( top.time )


    ###########################################################################
    #   save particle, density, and efields data in pkl format
    ###########################################################################
    if (iter % save_repetition == 0):
        h5file= PWpickle.PW("timestep_%06d.pkl" % iter)

        #### Protons
        nprotons = protons.getn()
        xprotons = protons.getx()
        yprotons = protons.gety()
        zprotons = protons.getz()
        vxprotons = protons.getvx()
        vyprotons = protons.getvy()
        vzprotons = protons.getvz()
        pidprotons = protons.getpid()
	      
        protons_r = []
        for i in range(nprotons):
            protons_r.append([i,xprotons[i],yprotons[i],zprotons[i],
                              pidprotons[i],vxprotons[i],vyprotons[i],
                              vzprotons[i]])
	      
        protons_np= np.array(protons_r)
        h5file.protons= protons_np

        #### Electrons
        nelectrons = electrons.getn()
        xelectrons = electrons.getx()
        yelectrons = electrons.gety()
        zelectrons = electrons.getz()
        vxelectrons = electrons.getvx()
        vyelectrons = electrons.getvy()
        vzelectrons = electrons.getvz()
        pidelectrons = electrons.getpid()
        print(nelectrons)
	      
        electrons_r = []
        for i in range(nelectrons):
            electrons_r.append([i,xelectrons[i],yelectrons[i],zelectrons[i],
                                pidelectrons[i],vxelectrons[i],vyelectrons[i],
                                vzelectrons[i]])
	      
        electrons_np= np.array(electrons_r)
        h5file.electrons= electrons_np
        
        #### H2+
        nh2plus = h2plus.getn()
        xh2plus = h2plus.getx()
        yh2plus = h2plus.gety()
        zh2plus = h2plus.getz()
        vxh2plus = h2plus.getvx()
        vyh2plus = h2plus.getvy()
        vzh2plus = h2plus.getvz()
        pidh2plus = h2plus.getpid()
	      
        h2plus_r = []
        for i in range(nh2plus):
            h2plus_r.append([i,xh2plus[i],yh2plus[i],zh2plus[i],pidh2plus[i],
                             vxh2plus[i],vyh2plus[i],vzh2plus[i]])
	      
        h2plus_np= np.array(h2plus_r)
        h5file.h2plus= h2plus_np
	      
    if iter*top.dt > 1.77*us:
	top.inject = 0

    step(1)

    # Close remaining open files
h5file.close()

save_data = PWpickle.PW("OutFile.pkl")
out_x = protons.getx()
out_y = protons.gety()
out_z = protons.getz()
out_vx = protons.getvx()
out_vy = protons.getvy()
out_vz = protons.getvz()
out_pid = protons.getpid()
out_n = protons.getn()
write_data=[]
for ii in range(out_n):
    write_data.append([ii,out_x[ii],out_y[ii],out_z[ii],out_vx[ii],out_vy[ii],
                       out_vz[ii],out_pid[ii]])
write_data_np=np.array(write_data)
save_data.protons=write_data_np
save_data.close()

zcross_data = PWpickle.PW("ZCross.pkl")
out_x = targetz_particles.getx()
out_y = targetz_particles.gety()
out_t = targetz_particles.gett()
out_vx = targetz_particles.getvx()
out_vy = targetz_particles.getvy()
out_vz = targetz_particles.getvz()
out_pid = targetz_particles.getpid()
out_n = targetz_particles.getn()
write_data=[]
for ii in range(out_n):
    write_data.append([ii,out_x[ii],out_y[ii],out_t[ii],out_vx[ii],out_vy[ii],
                       out_vz[ii],out_pid[ii]])
write_data_np=np.array(write_data)
zcross_data.targetz_particles=write_data_np
zcross_data.close()

#time history of electron desntiy 
ptitles(titlet="Electron density (#/m3) at center", titleb="Time (s)", 
        titlel =" ")
pla(eden_time, col_time,  color= red)
fma()

# phase-space plot to see any instability
#protons.ppzvtheta (view = 3,color=red)
#protons.ppzvr(view = 4,color=red)
#protons.ppxvx(view = 5,color=red)
#protons.ppzvz(view = 6,color=red)
#fma()
