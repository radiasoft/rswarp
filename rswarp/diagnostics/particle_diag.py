"""
This file defines the class ParticleDiagnostic
It modifies the `get_quantity` method to get particle_weight 
and not species_weight * particle_weight
"""
import os
import h5py
import numpy as np
from scipy import constants
from warp.data_dumping.openpmd_diag.generic_diag import OpenPMDDiagnostic
from warp_parallel import gatherarray, mpiallgather, comm_world
from warp.data_dumping.openpmd_diag.data_dict import macro_weighted_dict, weighting_power_dict, \
     particle_quantity_dict

class ParticleDiagnostic(OpenPMDDiagnostic) :
    """
    Class that defines the particle diagnostics to be done.

    Usage
    -----
    After initialization, the diagnostic is called by using the
    `write` method.
    """

    def __init__(self, period, top, w3d, comm_world=None,
                 species = {"electrons": None},
                 iteration_min=None, iteration_max=None,
                 particle_data=["position", "momentum", "weighting"],
                 select=None, write_dir=None, lparallel_output=False,
                 write_metadata_parallel=False, sub_sample=None) :
        """
        Initialize the field diagnostics.

        Parameters
        ----------
        period : int
            The period of the diagnostics, in number of timesteps.
            (i.e. the diagnostics are written whenever the number
            of iterations is divisible by `period`)

        top : the object representing the `top` package in Warp
            Contains information on the time.

        w3d : the object representing the `w3d` package in Warp
            Contains the dimensions of the grid.

        comm_world : a communicator object
            Either an mpi4py or a pyMPI object, or None (single-proc)

        species : a dictionary of Species objects
            The Species object that is written (e.g. elec)
            is assigned to the particleName of this species.
            (e.g. "electrons")

        iteration_min: int, optional
            iteration at which the diagnostic starts to be active

        iteration_max: int, optional
            iteration at which the diagnostic stops being active

        particle_data : a list of strings, optional
            A list indicating which particle data should be written.
            The list can contain any of the following strings:
            "position", "momentum", "E", "B", "id", "weighting" or "dfweighting"

        select : dict, optional
            Either None or a dictionary of rules
            to select the particles, of the form
            'x' : [-4., 10.]   (Particles having x between -4 and 10 microns)
            'ux' : [-0.1, 0.1] (Particles having ux between -0.1 and 0.1 mc)
            'uz' : [5., None]  (Particles with uz above 5 mc)

        write_dir : a list of strings, optional
            The POSIX path to the directory where the results are
            to be written. If none is provided, this will be the path
            of the current working directory.

        lparallel_output : boolean
            Switch to set output mode (parallel or gathering)
            If "True" : Parallel output

        write_metadata_parallel : boolean
            If "True" : file metadata are written in parallel

        sub_sample : integer
            If "None" : all particles are dumped
            If not None: every sub_sample particle is dumped
        """
        # General setup
        OpenPMDDiagnostic.__init__(self, period, top, w3d, comm_world,
                    iteration_min=iteration_min, iteration_max=iteration_max,
                    lparallel_output=lparallel_output, write_dir=write_dir,
                    write_metadata_parallel=write_metadata_parallel )

        # This check is needed when running in parallel. The order of the
        # particle_data needs to be the same on all processors and this is
        # only gauranteed with either list or tuple. A set for example won't work.
        assert isinstance(particle_data, list) or isinstance(particle_data, tuple),\
               Exception('ParticleDiagnostic: particle_data must be either a list or tuple')

        # Register the arguments
        self.particle_data = particle_data
        self.species_dict = species
        self.select = select
        self.sub_sample = sub_sample
        # Correct the bounds in momenta (since the momenta in Warp
        # are not unitless, but have the units of a velocity)
        for momentum in ['ux', 'uy', 'uz']:
            if (self.select is not None) and (momentum in self.select):
                if self.select[momentum][0] is not None:
                    self.select[momentum][0] *= constants.c
                if self.select[momentum][1] is not None:
                    self.select[momentum][1] *= constants.c

    def setup_openpmd_species_group( self, grp, species, N=None ) :
        """
        Set the attributes that are specific to the particle group

        Parameter
        ---------
        grp : an h5py.Group object
            Contains all the species

        species : a Warp species.Species object

        N: int, optional
            The global number of particles (if known)
            (used in order to store constant records)
        """
        # If the number of particles is unknown, set it to 1
        if N is None:
            N = 1

        # Generic attributes
        grp.attrs["particleShape"] = float( self.top.depos_order[0][0] )
        grp.attrs["currentDeposition"] = np.string_("Esirkepov")
        grp.attrs["particleSmoothing"] = np.string_("none")
        # Particle pusher
        if self.top.pgroup.lebcancel_pusher==True :
            grp.attrs["particlePush"] = np.string_("Vay")
        else :
            grp.attrs["particlePush"] = np.string_("Boris")
        # Particle shape
        if np.all( self.top.efetch==1 ) :
            grp.attrs["particleInterpolation"]= np.string_("momentumConserving")
        elif np.all( self.top.efetch==4 ) :
            grp.attrs["particleInterpolation"]= np.string_("energyConserving")

        # Setup constant datasets
        for quantity in ["charge", "mass", "positionOffset"] :
            grp.require_group(quantity)
            self.setup_openpmd_species_record( grp[quantity], quantity )
        for quantity in ["charge", "mass", "positionOffset/x",
                            "positionOffset/y", "positionOffset/z"] :
            grp.require_group(quantity)
            self.setup_openpmd_species_component( grp[quantity] )
            grp[quantity].attrs["shape"] = np.array([N], dtype=np.uint64)

        # Set the corresponding values
        grp["charge"].attrs["value"] = species.charge
        grp["mass"].attrs["value"] = species.mass
        grp["positionOffset/x"].attrs["value"] = 0.
        grp["positionOffset/y"].attrs["value"] = 0.
        grp["positionOffset/z"].attrs["value"] = 0.

    def setup_openpmd_species_record( self, grp, quantity ):
        """
        Set the attributes that are specific to a species record

        Parameter
        ---------
        grp : an h5py.Group object or h5py.Dataset
            The group that correspond to `quantity`
            (in particular, its path must end with "/<quantity>")

        quantity : string
            The name of the record being setup
            e.g. "position", "momentum"
        """
        if quantity == "dfweighting":
            return
        # Generic setup
        self.setup_openpmd_record( grp, quantity )

        # Weighting information
        grp.attrs["macroWeighted"] = macro_weighted_dict[quantity]
        grp.attrs["weightingPower"] = weighting_power_dict[quantity]

    def setup_openpmd_species_component( self, grp ) :
        """
        Set the attributes that are specific to a species component

        Parameter
        ---------
        grp : an h5py.Group object or h5py.Dataset

        quantity : string
            The name of the component
        """
        self.setup_openpmd_component( grp )

    def write_hdf5( self, iteration ) :
        """
        Write an HDF5 file that complies with the OpenPMD standard

        Parameter
        ---------
        iteration : int
             The current iteration number of the simulation.
        """
        # Before opening the file, select the particles that
        # need to be written for each species
        # (This allows to know the number of particles to be written,
        # which is needed when setting up the file)
        select_array_dict = {}
        selected_nlocals_dict = {}
        selected_nglobal_dict = {}
        # Loop over the different species, select the particles and fill
        # select_array_dict, selected_nlocals_dict, selected_nglobal_dict
        for species_name in sorted(self.species_dict.keys()):
            # Select the particles that will be written
            species = self.species_dict[species_name]
            select_array_dict[species_name] = self.apply_selection( species )
            # Get their total number
            n = select_array_dict[species_name].sum()
            if comm_world is not None :
                # In MPI mode: gather and broadcast an array containing
                # the number of particles on each process
                selected_nlocals_dict[species_name] = mpiallgather( n )
                selected_nglobal_dict[species_name] = \
                    sum(selected_nlocals_dict[species_name])
            else:
                # Single-proc output
                selected_nlocals_dict[species_name] = None
                selected_nglobal_dict[species_name] = n

        # Find the file name
        filename = "data%08d.h5" %iteration
        fullpath = os.path.join( self.write_dir, "hdf5", filename )

        # Create the file and setup its attributes
        # (can be done by one proc or in parallel)
        self.create_file_empty_particles( fullpath, self.top.it,
                    self.top.time, self.top.dt, selected_nglobal_dict )

        # Open the file again (possibly in parallel)
        f = self.open_file( fullpath, parallel_open=self.lparallel_output )
        # (f is None if this processor does not participate in writing data)

        # Loop over the different species and write the requested quantities
        for species_name in sorted(self.species_dict.keys()) :

            # Get the HDF5 species group
            if f is not None:
                species_path = "/data/%d/particles/%s"%(iteration,species_name)
                species_grp = f[species_path]
            else:
                species_grp = None

            # Get the relevant species object and selection array
            species = self.species_dict[species_name]
            select_array = select_array_dict[species_name]
            n_rank = selected_nlocals_dict[species_name]

            # Write the datasets for each particle datatype
            self.write_particles( species_grp, species, n_rank, select_array )

        # Close the file
        if f is not None:
            f.close()

    def write_particles( self, species_grp, species, n_rank, select_array ):
        """
        Write all the particle data sets for one given species

        species_grp : an h5py.Group
            The group where to write the species considered

        species : a warp Species object
            The species object to get the particle data from

        n_rank: an array with dtype = int of size = n_procs
            Contains the local number of particles for each process

        select_array : 1darray of bool
            An array of the same shape as that particle array
            containing True for the particles that satify all
            the rules of self.select
        """
        for particle_var in self.particle_data :

            # Vector quantity
            if particle_var in ["position", "momentum", "E", "B"] :
                for coord in ["x", "y", "z"] :
                    quantity = "%s%s" %( particle_quantity_dict[particle_var],
                                        coord)
                    quantity_path = "%s/%s" %(particle_var, coord)
                    self.write_dataset( species_grp, species, quantity_path,
                                        quantity, n_rank, select_array )

            # Scalar quantities
            elif particle_var=="weighting" :
                quantity = "w"
                quantity_path = "weighting"
                self.write_dataset( species_grp, species, quantity_path,
                                    quantity, n_rank, select_array )
            elif particle_var=="id" :
                quantity = "id"
                quantity_path = "id"
                self.write_dataset( species_grp, species, quantity_path,
                                    quantity, n_rank, select_array )
            elif particle_var == "dfweighting":
                quantity = "dfw"
                quantity_path = "dfweighting"
                self.write_dataset( species_grp, species, quantity_path,
                                    quantity, n_rank, select_array )

            else :
                raise ValueError("Invalid string in %s of species"
                                     %(particle_var))

    def apply_selection( self, species ) :
        """
        Apply the rules of self.select to determine which
        particles should be written

        Parameters
        ----------
        species : a Species object

        Returns
        -------
        A 1d array of the same shape as that particle array
        containing True for the particles that satify all
        the rules of self.select
        """
        # Initialize an array filled with True
        npart = species.getn(gather = 0)
        select_array = np.ones(npart, dtype='bool')

        # Apply the rules successively
        if self.select is not None :
            # Go through the quantities on which a rule applies
            for quantity in self.select.keys() :
                quantity_array = self.get_quantity( species, quantity )
                # Lower bound
                if self.select[quantity][0] is not None :
                    select_array = np.logical_and(
                        quantity_array > self.select[quantity][0],
                        select_array )
                # Upper bound
                if self.select[quantity][1] is not None :
                    select_array = np.logical_and(
                        quantity_array < self.select[quantity][1],
                        select_array )
        if self.sub_sample is not None:
            subsamp_array = np.zeros(npart, dtype='bool')
            subsamp_array[::self.sub_sample]=1
            # Subsample particle array
            select_array=np.logical_and(subsamp_array,select_array)
        return( select_array )

    def create_file_empty_particles( self, fullpath, iteration,
                                   time, dt, select_nglobal_dict=None ):
        """
        Create an openPMD file with empty meshes and setup all its attributes

        Parameters
        ----------
        fullpath: string
            The absolute path to the file to be created

        iteration: int
            The iteration number of this diagnostic

        time: float (seconds)
            The physical time at this iteration

        dt: float (seconds)
            The timestep of the simulation

        select_nglobal_dict: dictionary or None
            A dictionary whose keys are strings with the names of the
            species, and whose values are integers representing the global
            number of particles (across all MPI proc) that have been selected
            (according to the rules of self.select)
            If `select_nglobal_dict` is None, then the datasets are considered
            appendable, instead of having a fixed size.
        """
        # Create the file (can be done by one proc or in parallel)
        f = self.open_file( fullpath,
            parallel_open=self.write_metadata_parallel )

        # Setup the different layers of the openPMD file
        # (f is None if this processor does not participate is writing data)
        if f is not None:

            # Setup the attributes of the top level of the file
            self.setup_openpmd_file( f, iteration, time, dt )
            # Setup the meshes group (contains all the particles)
            f.attrs["particlesPath"] = np.string_("particles/")
            particle_path = "/data/%d/particles/" %iteration
            particle_grp = f.require_group(particle_path)
            # Loop through all particle species
            for species_name in sorted(self.species_dict.keys()):
                species = self.species_dict[species_name]

                # Check the number of particles to write
                if select_nglobal_dict is not None:
                    N = select_nglobal_dict[species_name]
                else:
                    N = None

                # Create and setup the h5py.Group species_grp
                species_path = particle_path+"%s/" %(species_name)
                species_grp = f.require_group( species_path )
                self.setup_openpmd_species_group( species_grp, species, N=N )

                # Loop over the different quantities that should be written
                # and setup the corresponding datasets
                for particle_var in self.particle_data:

                    # Vector quantities
                    if particle_var in ["position", "momentum", "E", "B"]:
                        # Setup the dataset
                        quantity_path=species_path+ "%s/" %particle_var
                        quantity_grp = f.require_group(quantity_path)
                        for coord in ["x","y","z"]:
                            # Create the dataset (fixed size or appendable)
                            if N is not None:
                                dset = quantity_grp.create_dataset(
                                    coord, (N,), dtype='f8')
                            else:
                                dset = quantity_grp.create_dataset(
                                    coord, (0,), maxshape=(None,), dtype='f8')
                            self.setup_openpmd_species_component( dset )
                        self.setup_openpmd_species_record( quantity_grp,
                                                           particle_var)

                    # Scalar quantity
                    elif particle_var in ["weighting", "id", "t", "dfweighting"]:
                        # Choose the type of the output
                        if particle_var == "id":
                            dtype = 'uint64'
                        else:
                            dtype = 'f8'
                        # Create the dataset (fixed size or appendable)
                        if N is not None:
                            dset = species_grp.create_dataset(
                                particle_var, (N,), dtype=dtype )
                        else:
                            dset = species_grp.create_dataset( particle_var,
                                (0,), maxshape=(None,), dtype=dtype)
                        self.setup_openpmd_species_component( dset )
                        self.setup_openpmd_species_record( dset, particle_var )

                    # Unknown field
                    else:
                        raise ValueError(
                        "Invalid string in particletypes: %s" %particle_var)

            # Close the file
            f.close()

    def write_dataset( self, species_grp, species, path, quantity,
                       n_rank, select_array ) :
        """
        Write a given dataset

        Parameters
        ----------
        species_grp : an h5py.Group
            The group where to write the species considered

        species : a warp Species object
            The species object to get the particle data from

        path : string
            The relative path where to write the dataset, inside the species_grp

        quantity : string
            Describes which quantity is written
            x, y, z, ux, uy, uz, w, dfw, ex, ey, ez,
            bx, by or bz

        n_rank: an array with dtype = int of size = n_procs
            Contains the local number of particles for each process

        select_array : 1darray of bool
            An array of the same shape as that particle array
            containing True for the particles that satify all
            the rules of self.select
        """
        # Get the dataset and setup its attributes
        if species_grp is not None:
            dset = species_grp[path]

        # Fill the dataset with the quantity
        # (Single-proc operation, when using gathering)
        if not self.lparallel_output:
            quantity_array = self.get_dataset( species,
                    quantity, select_array, gather=True )
            if self.rank == 0:
                dset[:] = quantity_array
        # Fill the dataset with these quantities with respect
        # to the global position of the local domain
        # (truly parallel HDF5 output)
        else :
            quantity_array = self.get_dataset( species,
                    quantity, select_array, gather=False )
            # Calculate last index occupied by previous rank
            nold = sum(n_rank[0:self.rank])
            # Calculate the last index occupied by the current rank
            nnew = nold+n_rank[self.rank]
            # Write the local data to the global array
            dset[nold:nnew] = quantity_array

    def get_dataset( self, species, quantity, select_array, gather ) :
        """
        Extract the array that satisfies select_array

        species : a Particles object
            The species object to get the particle data from

        quantity : string
            The quantity to be extracted (e.g. 'x', 'uz', 'w')

        select_array : 1darray of bool
            An array of the same shape as that particle array
            containing True for the particles that satify all
            the rules of self.select

        gather : bool
            Whether to gather the fields on the first processor
        """

        # Extract the quantity
        quantity_array = self.get_quantity( species, quantity )

        # Apply the selection
        quantity_array = quantity_array[ select_array ]

        # If this is the momentum, mutliply by the proper factor
        if quantity in ['ux', 'uy', 'uz']:
            quantity_array *= species.mass

        # Gather the data if required
        if gather==False :
            return( quantity_array )
        else :
            return(gatherarray( quantity_array, root=0 ))

    def get_quantity( self, species, quantity ) :
        """
        Get a given quantity

        Parameters
        ----------
        species : a Species object
            Contains the species object to get the particle data from

        quantity : string
            Describes which quantity is queried
            Either "x", "y", "z", "ux", "uy", "uz", "w", "ex", "ey", "ez",
            "bx", "by" or "bz"
            or even a formula involving particle quantities ("x+y")
        """

        #NB (for developers): this is a very simple implementation
        #one must take care of the order of search of particle quantities
        #'ux','uy','uz' etc. so that the routine effectively works by just
        #using the replace method on strings. In addition, double character
        #quantities containing other quantitities (e.g 'ex' contains 'x')
        #are replaced with numbered quantities 'e1' to avoid collisions
        #with replace().

        dict_keys_val=dict()
        if "ux" in quantity :
            dict_keys_val["u1"]=species.getux(gather=False)
            quantity=quantity.replace("ux","u1")
        if "uy" in quantity :
            dict_keys_val["u2"]=species.getuy(gather=False)
            quantity=quantity.replace("uy","u2")
        if "uz" in quantity :
            dict_keys_val["u3"]=species.getuz(gather=False)
            quantity=quantity.replace("uz","u3")
        if "ex" in quantity :
            dict_keys_val["e1"]=species.getex(gather=False)
            quantity=quantity.replace("ex","e1")
        if "ey" in quantity :
            dict_keys_val["e2"]=species.getey(gather=False)
            quantity=quantity.replace("ey","e2")
        if "ez" in quantity :
            dict_keys_val["e3"]=species.getez(gather=False)
            quantity=quantity.replace("ez","e3")
        if "bx" in quantity :
            dict_keys_val["b1"]=species.getbx(gather=False)
            quantity=quantity.replace("bx","b1")
        if "by" in quantity :
            dict_keys_val["b2"]=species.getby(gather=False)
            quantity=quantity.replace("by","b2")
        if "bz" in quantity :
            dict_keys_val["b3"]=species.getbz(gather=False)
            quantity=quantity.replace("bz","b3")
        if "dfw" in quantity:
            dict_keys_val["df"]=species.getdfweight(gather=False)
            quantity=quantity.replace("dfw", "df")
        if "w" in quantity  :
            dict_keys_val["w"]=species.getw(gather=False)
        if "x" in quantity :
            dict_keys_val["x"]=species.getx(gather=False)
        if "y" in quantity :
            dict_keys_val["y"]=species.gety(gather=False)
        if "z" in quantity :
            dict_keys_val["z"]=species.getz(gather=False)
        if "id" in quantity:
            # The ssnid is stored in Warp as a float. Thus, it needs
            # to be converted to the nearest integer (rint)
            dict_keys_val["id"] = \
            (np.rint( species.getssn(gather=False) )).astype('uint64')

        string_to_exec=quantity
        for keys in dict_keys_val.keys():
            string_to_exec=string_to_exec.replace(keys,"dict_keys_val[\'"+keys+"\']")
        string_to_exec='quantity_array = '+string_to_exec
        local_dict = {'dict_keys_val':dict_keys_val, 'quantity_array':None}
        exec(string_to_exec, globals(), local_dict)

        return( local_dict['quantity_array'] )
