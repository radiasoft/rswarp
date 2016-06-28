import numpy as np
import os
import h5py as h5
import matplotlib.pyplot as plt


class ElectrostaticFields:
	"""
		Produce an HDF5 file with electric fields and potential.
		File tree:
		/mesh
			/x
			/y
			/z
		attributes: nx, ny, nz

		/potential
			/electric - [xdim,ydim,zdim]
		/fields
			/electric - [fieldaxis,xdim,ydim,zdim]

		Parameters:
			solver: Must be a Multigrid3D object.
			top: Object representing Warp's top package.
	"""

	def __init__(self, solver, top):

		self.solver = solver
		self.top = top


	def gatherfields(self):
		self.efield = self.solver.getselfe()

		return self.efield

	def gatherpotential(self):
		self.phi = self.solver.getphi()


	def gathermesh(self):
		self.nx = self.solver.nx
		self.ny = self.solver.ny
		self.nz = self.solver.nz

		self.xmesh = self.solver.xmesh
		self.ymesh = self.solver.ymesh
		self.zmesh = self.solver.zmesh

	def write(self,filename = 'efield.h5'):

		self.gatherfields()
		self.gatherpotential()
		self.gathermesh()

		filename = os.path.splitext(filename)
		step = str(self.top.it)
		filename = '%s%s%s' % (filename[0],step.zfill(5),filename[1]) 

		f = h5.File(filename, 'w')

		f.create_dataset('mesh/x', data=self.xmesh)
		f.create_dataset('mesh/y', data=self.ymesh)
		f.create_dataset('mesh/z', data=self.zmesh)

		f.attrs['nx'] = self.nx
		f.attrs['ny'] = self.ny
		f.attrs['nz'] = self.nz

		f.create_dataset('fields/electric', data=self.efield)
		f.create_dataset('potential/electric', data=self.phi)


if __name__ == '__main__':
	f0 = h5.File('efield00051.h5', 'r')

	xi = f0['mesh/x']
	yi = f0['mesh/y']
	zi = f0['mesh/z']

	phi = f0['potential/electric']

	plt.contourf(xi,yi,phi[:,:,10],cmap=plt.cm.viridis)

	plt.colorbar()

	plt.show()




