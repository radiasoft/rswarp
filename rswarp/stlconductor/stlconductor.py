from __future__ import division
import pymesh
from warp import *
from copy import deepcopy
import warnings

class LineSegment:
    """
    class to manage line segment instance
    """
    def __init__(self, points):
        if np.asarray(points).shape != (2,3):
            raise TypeError("Input vertices must be a container having 2 points")
        self._points = np.array(points)

    def point(self, idx):
        return self._points[idx]

    def points(self):
        return self._points

# end of class LineSegment

class Ray:
    """
    class to manage ray instance
    """
    def __init__(self, lineseg):
        if not isinstance(lineseg, LineSegment):
            raise TypeError("Illegal input for Ray initialization")
        self.orig = lineseg.point(0)
        self.dir = lineseg.point(1) - self.orig

# end of class Ray

class Triangle:
    """
    class to manage triangle instance
    """

    def __init__(self, vertices):
        if np.asarray(vertices).shape != (3,3):
            raise TypeError("Input vertices must be a container having 3 points")

        self._vertices = np.array(vertices)

        self._normal = np.cross(self._vertices[1] - self._vertices[0],
                                self._vertices[2] - self._vertices[0])
        self._normal /= np.linalg.norm(self._normal)

    def vertex(self, idx = 0):
        return self._vertices[idx]

    def vertices(self):
        return self._vertices

    def normal(self):
        return self._normal

# end of class Triangle

class STLconductor(Assembly):
    """
    Conductor defined in a STL file.
    Input: filename - STL file
           voltage=0. - conductor voltage
           xcent=0., ycent=0., zcent=0. - not necessarily needed for defining STLconductor, but required by Assembly
           precision_decimal=4 - precision (in the format of scientific expression) kept for vertices read from stl file
           normalization_factor=None - normalize coordinates of mesh by normalization_factor to reduce round-off errors
           fuzz=None - (small) fuzzy number
           disp="none", "auto", or (delta_x, delta_y, delta_z) - move the conductor by this small amount of displacement for purpose of effectively finding intercepts
           condid="next" - conductor id, must be integer, or can be 'next' in which case a unique ID is chosen
    """
    def __init__(self, filename, voltage=0.,
                       xcent=0., ycent=0., zcent=0.,
                       raytri_scheme="watertight", precision_decimal=None, normalization_factor=None, fuzz=1e-14,
                       disp="none", verbose="off", condid="next", **kw):
        usenewconductorgeneration()
        kwlist = []
        Assembly.__init__(self, voltage, xcent, ycent, zcent, condid, kwlist,
                          self.conductorf, self.conductord, self.intercept,
                          self.conductorfnew,
                          kw=kw)
        self.filename = filename
        self.installedintercepts = {}
        self._precision_decimal = precision_decimal
        self._normalization_factor = normalization_factor
        self._winding_fuzz = fuzz           # fuzzy for small scale winding number
        self._fuzz = fuzz
        self._eps = 1e-6                    # _eps*dx: small number used for ray-tri parallel judgement
        self._disp = disp
        self._infinity = np.inf
#         self._infinity = largepos

        if verbose == "off": self._verbose = False
        elif verbose == "on": self._verbose = True
        else:
            err_msg = "Illegal verbose input value: "
            err_msg += "[legal options are \"off\" (default) or \"on\"]."
            raise StandardError(err_msg)

        if raytri_scheme == "watertight":
            self._intersect3D_ray_triangle = self._intersect3D_ray_triangle_watertight
            if self._verbose:
                print(" ---  Ray-triangle intersection finding scheme selected: Watertight")
        elif raytri_scheme == "moller":
            self._intersect3D_ray_triangle = self._intersect3D_ray_triangle_moller
            if self._verbose:
                print(" ---  Ray-triangle intersection finding scheme selected: Moller-Trumbore")
        else:
            err_msg = "Illegal raytri_scheme input value: "
            err_msg += "[legal options are \"watertight\" (default) or \"moller\"]."
            raise StandardError(err_msg)

        self.surface = pymesh.load_mesh(filename)

        self._surface_mesh_quality_check()

        if disp != "none" and disp != "auto":
            try:
                if len(np.asarray(disp)) != 3: raise StandardError("")
            except:
                err_msg = "Illegal disp input type/values: "
                err_msg += "[legal options are \"none\" (default), \"auto\", or an array with 3 entries]."
                raise StandardError(err_msg)

            # make a copy of mesh read from pymesh because it is read-only
            vertices = deepcopy(self.surface.vertices)

            for i in range(len(disp)): vertices[:, i] += disp[i]

            self.surface = pymesh.form_mesh(vertices, self.surface.faces)

        self._surface_boxlo = np.amin(self.surface.vertices, axis=0)
        self._surface_boxhi = np.amax(self.surface.vertices, axis=0)

    # end of __init__ function

    def getextent(self):
        """
        return the extent of a stl conductor
        The extent is considered to be the minimum rectangle box
        that just encloses the conductor used in  Grid.getdatanew for gridintercepts calculation
        """
        return ConductorExtent([self._surface_boxlo[0], self._surface_boxlo[1], self._surface_boxlo[2]],
                               [self._surface_boxhi[0], self._surface_boxhi[1], self._surface_boxhi[2]],
                               [self.xcent, self.ycent, self.zcent])
#         return ConductorExtent([-largepos, -largepos, -largepos],
#                                [+largepos, +largepos, +largepos],
#                                [self.xcent, self.ycent, self.zcent])

    # end of getextent function
    
    def conductorf(self):
        raise Exception("This function should never be called")

    # end of conductorf function

    def conductord(self, xcent, ycent, zcent, n, x, y, z, distance):
        """
        Input: x - a list of x-coordinates
               y - a list of y-coordinates
               z - a list of z-coordinates
               (requirement: x, y and z must have the same length)
        Output: distance - returned list of distance
                           (points outside conductor have positive values,
                            points inside have negative values)
        (Note: other parameters are not used, but kept here to be compatible with warp)
        """
        nx = len(x)
        if nx != len(y) or nx != len(z):
            raise Exception("input lists of coordinates must have same length")

        pts = np.array([x, y, z]).transpose()
        winding_number = pymesh.compute_winding_number(self.surface, pts)
        winding_number[np.abs(winding_number) < self._winding_fuzz] = 0.0
        pts_inside = winding_number > 0.
        distance[:] = np.sqrt(pymesh.distance_to_mesh(self.surface, pts)[0])
        distance[pts_inside] *= -1

    # end of conductord function

    def intercept(self, xx, yy, zz, vx, vy, vz):
        """
        calculates the location were particles with the given velocities
        most recently intersected the conductor
        Input : xx, yy, zz - a list of x, y, and z coordinates of particles' positions
                vx, vy, vz - a list of x, y, and z components of particles' velocities
        Output : ??
        """
        raise Exception("STLconductor intercept not yet implemented")

    # end of intercept function

    def conductorfnew(self, xcent, ycent, zcent, intercepts, fuzz):
        """
        interface of generating x, y, z intercepts
        Input: intercepts - Fortran data structure ConductorInterceptType
                            for storing intercepts relevant info
                fuzz - fuzzy number (required by Warp, but not used in STLconductor.
                                     STLconductor uses its own fuzz self._fuzz)
        Output: intercepts - ready to be used by "conductordelfromintercepts"
        """

        # Warning: STLconductor caches xi, yi, and zi to save computational time
        # by assuming that CAD conductors, once installed, never change
        # during a simulation.
        # Warp code recomputes intercepts for the MG field solver every PIC step.
        # This is inefficient but tolerable for Warp's built-in geometries.
        # However, computing intercepts for CAD conductors is very time consuming,
        # and recomputing these intercepts every PIC step is unacceptable.
        mglevel = intercepts.mglevel
        if mglevel == 0:
            self._dx0 = np.array([intercepts.dx, intercepts.dy, intercepts.dz])

        if not mglevel in self.installedintercepts:
            if self._verbose:
                print " ---  generating intercepts for mglevel = {}".format(mglevel)
            xi, yi, zi = self._conductorfnew_impl(intercepts)
            self.installedintercepts[mglevel] = (xi, yi, zi)
        else:
            xi, yi, zi = self.installedintercepts[mglevel]

        intercepts.nxicpt = xi.shape[0]
        intercepts.nyicpt = yi.shape[0]
        intercepts.nzicpt = zi.shape[0]
        intercepts.gchange()
        intercepts.xintercepts[...] = xi
        intercepts.yintercepts[...] = yi
        intercepts.zintercepts[...] = zi

        return xi, yi, zi

    # end of conductorfnew function
    
    def _round_float(self, f, n, expr):
        return float(('%.' + str(n) + expr) % f)

    def _surface_mesh_quality_check(self):
        
        # extent of surface
        surface_boxlo = np.amin(self.surface.vertices, axis=0)
        surface_boxhi = np.amax(self.surface.vertices, axis=0)

        # get a statistics of angles of all elements
        self._surface_elem_angles = np.zeros((self.surface.num_faces, 3))
        iface = -1
        for face in self.surface.faces:
            vert = self.surface.vertices[face]
            iface += 1
            for i in range(3):
                a = vert[(i+1)%3] - vert[i]
                b = vert[(i+2)%3] - vert[i]
                theta = np.arccos( np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)) )
                self._surface_elem_angles[iface, i] = theta
        self._surface_elem_angles = np.degrees(self._surface_elem_angles)
        self._surface_elem_min_angle = np.amin(self._surface_elem_angles)
        self._surface_elem_max_angle = np.amax(self._surface_elem_angles)
        self._surface_elem_median_angle = np.median(self._surface_elem_angles)

        print(" ---  STL conductor surface mesh: # of vertices and elements = ({}, {})".format(
            self.surface.num_vertices, self.surface.num_elements))
        print(" ---  STL conductor surface mesh: extent -> (xmin,ymin,zmin)=({},{},{}), (xmax,ymax,zmax)=({},{},{})".format(
            surface_boxlo[0], surface_boxlo[1], surface_boxlo[2], surface_boxhi[0], surface_boxhi[1], surface_boxhi[2]))
        print(" ---  STL conductor surface mesh: triangle element angles -> (min,max,median)=({},{},{}) deg".format(
            self._surface_elem_min_angle, self._surface_elem_max_angle, self._surface_elem_median_angle))


    def _conductorfnew_impl(self, intercepts):
        """
        actual function that calculates and returns x, y, z intercepts
        Input: intercepts - Fortran data structure ConductorInterceptType
                            for storing intercepts relevant info
        Output: xi[0:nxicpt, ny, nz] - xintercepts
                yi[0:nyicpt, nx, nz] - yintercepts
                zi[0:nzicpt, nx, ny] - zintercepts
        """

        # number of mesh nodes of simulation box
        nx = intercepts.nx; ny = intercepts.ny; nz = intercepts.nz

        # mesh size of simulation box
        dx = np.array([intercepts.dx, intercepts.dy, intercepts.dz])

        # lower and higher corners of simulation box
        boxlo = np.array([intercepts.xmmin, intercepts.ymmin, intercepts.zmmin])
        boxhi = np.array([intercepts.xmmin + nx*dx[0], intercepts.ymmin + ny*dx[1], intercepts.zmmin + nz*dx[2]]) 

        if not self._normalization_factor:
            # minimal box that can enclose the surface
            surface_boxlo = self._surface_boxlo 
            surface_boxhi = self._surface_boxhi
            dx0 = self._dx0     # cell size at mglevel = 0
            # surface mesh vertices
            vertices = deepcopy(self.surface.vertices)
        else:
            dx /= self._normalization_factor
            boxlo /= self._normalization_factor
            boxhi /= self._normalization_factor
            dx0 = self._dx0/self._normalization_factor  # cell size at mglevel = 0
            # minimal box that can enclose the surface
            surface_boxlo = self._surface_boxlo / self._normalization_factor
            surface_boxhi = self._surface_boxhi / self._normalization_factor
            # surface mesh vertices
            vertices = self.surface.vertices / self._normalization_factor
            
        if self._precision_decimal:
            if not self._normalization_factor: expr = 'e'
            else: expr = 'f'
            for i in range(3):  # cell size at mglevel = 0
                surface_boxlo[i] = self._round_float(surface_boxlo[i], self._precision_decimal, expr)
                surface_boxhi[i] = self._round_float(surface_boxhi[i], self._precision_decimal, expr)
                m, n = vertices.shape
                for i in range(m):
                    for j in range(n):
                        vertices[i,j] = self._round_float(vertices[i,j], self._precision_decimal, expr)

        self._eps *= np.amin(dx)

        x = np.linspace(boxlo[0], boxhi[0], nx+1)
        y = np.linspace(boxlo[1], boxhi[1], ny+1)
        z = np.linspace(boxlo[2], boxhi[2], nz+1)

        surface_box_extent = surface_boxhi - surface_boxlo

        if self._verbose:
            print " ---  (nx,ny,nz)=({},{},{}), (dx,dy,dz)=({},{},{})".format(nx,ny,nz,dx[0],dx[1],dx[2])
            print " ---  boxlo=({},{},{}), boxhi=({},{},{})".format(boxlo[0],boxlo[1],boxlo[2],boxhi[0],boxhi[1],boxhi[2])
            print " ---  surface_boxlo=({},{},{}), surface_boxhi=({},{},{})".format(surface_boxlo[0],surface_boxlo[1],surface_boxlo[2],
                                                                                    surface_boxhi[0],surface_boxhi[1],surface_boxhi[2])

        eps = 5e-4*dx0

        if self._verbose:
            print " ---  generating xintercepts ..."
        xmin = surface_boxlo[0]-0.5*surface_box_extent[0]; xmax = surface_boxhi[0]+0.5*surface_box_extent[0]
        imp = (0, 1, 2)
        directions = [(0, 0, 0), (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1)]
        successful = True
        for direction in directions:
            disp = eps*np.asarray(direction)

            if not successful:
                msg = "Failed to generate xintercepts.\n"
                msg += "This happens when the segment lies in surface element planes or intersects with element edges "
                msg += "so that intercepts cannot be perfectly resolved due to round-off errors.\n"
                if self._disp == "auto":
                    msg += "Displacing the object by [{}, {}, {}], and starting over.\n".format(disp[0], disp[1], disp[2])
                    warnings.warn(msg)
                else:
                    msg += "A possible solution is to move the object by a tiny amount of displacement through \"disp=[x, y, z]\" option so segment is off the element edge.\n"
                    msg += "Normalizing through \"normalization_factor=dx\" option is also recommended."
                    raise StandardError(msg)

            xintercepts = self._gen_intercepts(imp, xmin, xmax, [nx, ny, nz], [x, y, z], dx, boxlo, boxhi, vertices+disp)
            xi, successful = self._produce_intercepts(xintercepts)

            if successful: break

        if self._verbose:
            print " ---  generating yintercepts ..."
        ymin = surface_boxlo[1]-0.5*surface_box_extent[1]; ymax = surface_boxhi[1]+0.5*surface_box_extent[1]
        imp = (1, 0, 2)
        directions = [(0, 0, 0), (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1)]
        successful = True
        for direction in directions:
            disp = eps*np.asarray(direction)

            if not successful:
                msg  = "Failed to generate yintercepts.\n"
                msg += "This happens when the segment lies in surface element planes or intersects with element edges "
                msg += "so that intercepts cannot be perfectly resolved due to round-off errors.\n"
                if self._disp == "auto":
                    msg += "Displacing the object by [{}, {}, {}], and starting over.\n".format(disp[0], disp[1], disp[2])
                    warnings.warn(msg)
                else:
                    msg += "A possible solution is to move the object by a tiny amount of displacement through \"disp=[x, y, z]\" option so segment is off the element edge.\n"
                    msg += "Normalizing through \"normalization_factor=dx\" option is also recommended."
                    raise StandardError(msg)

            yintercepts = self._gen_intercepts(imp, ymin, ymax, [nx, ny, nz], [x, y, z], dx, boxlo, boxhi, vertices+disp)
            yi, successful = self._produce_intercepts(yintercepts)

            if successful: break

        if self._verbose:
            print " ---  generating zintercepts ..."
        zmin = surface_boxlo[2]-0.5*surface_box_extent[2]; zmax = surface_boxhi[2]+0.5*surface_box_extent[2]
        imp = (2, 0, 1)
        directions = [(0, 0, 0), (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0)]
        successful = True
        for direction in directions:
            disp = eps*np.asarray(direction)

            if not successful:
                msg  = "Failed to generate zintercepts.\n"
                msg += "This happens when the segment lies in surface element planes or intersects with element edges "
                msg += "so that intercepts cannot be perfectly resolved due to round-off errors.\n"
                if self._disp == "auto":
                    msg += "Displacing the object by [{}, {}, {}], and starting over.\n".format(disp[0], disp[1], disp[2])
                    warnings.warn(msg)
                else:
                    msg += "A possible solution is to move the object by a tiny amount of displacement through \"disp=[x, y, z]\" option so segment is off the element edge.\n"
                    msg += "Normalizing through \"normalization_factor=dx\" option is also recommended."
                    raise StandardError(msg)

            zintercepts = self._gen_intercepts(imp, zmin, zmax, [nx, ny, nz], [x, y, z], dx, boxlo, boxhi, vertices+disp)
            zi, successful = self._produce_intercepts(zintercepts)

            if successful: break

        if self._normalization_factor:
            xi *= self._normalization_factor
            yi *= self._normalization_factor
            zi *= self._normalization_factor

        return xi, yi, zi

    # end of _conductorfnew_impl function

    def _intersect3D_lineseg_triangle(self, lineseg, triangle):
        """
        find intersection of a line segment with a triangle
        """
        ray = Ray(lineseg)
        t = self._intersect3D_ray_triangle(ray, triangle)
        if t is None or t > 1.0:
            return None
        return ray.orig + t*ray.dir

    # end of _intersect3D_lineseg_triangle function

    def _intersect3D_ray_triangle_moller(self, ray, triangle):
        """
        Moller-Trumbore's ray-triangle intersection finding scheme
        """
        # See [Tomas Moller and Ben Trumbore, "Fast, Minimum Storage Ray-Triangle Intersection".
        # Jounal of Graphics Tools. 2: 21-28. doi:10.1080/10867651.1997.10487468].
        # Current python implementation is based on the c code publicly available through
        # http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/raytri/
        # Performance analysis can also be found.

        rorig = ray.orig
        rdir = ray.dir
        edge1 = triangle.vertex(1) - triangle.vertex(0)
        edge2 = triangle.vertex(2) - triangle.vertex(0)
        pvec = np.cross(rdir, edge2)
        det = np.dot(edge1, pvec)

        if det > self._eps:
            # calculate U parameter and test bounds
            tvec = rorig - triangle.vertex(0)
            u = np.dot(tvec, pvec)
            if u < 0.0 or u > det: return None
            qvec = np.cross(tvec, edge1)
            # calculate V parameter and test bounds
            v = np.dot(rdir, qvec)
            if v < 0.0 or u+v > det: return None
        elif det < -self._eps:
            # calculate U parameter and test bounds
            tvec = rorig - triangle.vertex(0)
            u = np.dot(tvec, pvec)
            if u > 0.0 or u < det: return None
            # calculate V parameter and test bounds
            qvec = np.cross(tvec, edge1)
            v = np.dot(rdir, qvec)
            if v > 0.0 or u+v < det: return None
        else:
            return None         # ray is parallel to the plane of the triangle

        inv_det = 1.0/det
        return np.dot(edge2, qvec)*inv_det

    # end of _intersect3D_ray_triangle_moller function

    def _intersect3D_ray_triangle_watertight(self, ray, triangle):
        """
        Watertight ray-triangle intersection finding scheme
        """
        # See [Sven Woop, Carsten Benthin and Ingo Wald, "Watertight Ray/Triangle Intersection".
        # Jounal of Computer Graphics Techniques. Vol.2, No.1: 65-82 (2013). http://jcgt.org/published/0002/01/05/].
        # Another helpful discussion on this algorithm can also be found in
        # https://www.allthingsphi.com/blog/2016/12/18/watertight-ray-triangle-intersection.html.

        rorig = ray.orig
        rdir = ray.dir

        # get the index where the ray direction is maximal
        kz = np.argmax(np.abs(rdir))
        kx = kz+1
        if kx == 3: kx = 0
        ky = kx+1
        if ky == 3: ky = 0
        if rdir[kz] < 0.0:
            # swap(kx, ky)
            ky, kx = kx, ky
        Sx = rdir[kx]/rdir[kz]
        Sy = rdir[ky]/rdir[kz]
        Sz = 1.0/rdir[kz]

        # vertices relative to ray origin
        A = triangle.vertex(0) - rorig
        B = triangle.vertex(1) - rorig
        C = triangle.vertex(2) - rorig

        # perform shear and scale of vertices
        Ax = A[kx] - Sx*A[kz]
        Ay = A[ky] - Sy*A[kz]
        Bx = B[kx] - Sx*B[kz]
        By = B[ky] - Sy*B[kz]
        Cx = C[kx] - Sx*C[kz]
        Cy = C[ky] - Sy*C[kz]

        # calculate scaled barycentric coordinates
        U = Cx*By - Cy*Bx
        V = Ax*Cy - Ay*Cx
        W = Bx*Ay - By*Ax

        # perform edge tests
        if (U < 0.0 or V < 0.0 or W < 0.0) and (U > 0.0 or V > 0.0 or W > 0.0): return None

        # determinant
        det = U+V+W
        if det == 0.0: return None

        # calculate scaled z-coordinates of vertices
        # use them to calculate the hit distance
        Az = Sz*A[kz]
        Bz = Sz*B[kz]
        Cz = Sz*C[kz]
        T = U*Az + V*Bz + W*Cz

        # if (det > 0.0 and T < 0.0) or (det < 0.0 and T > 0.0) is equivalent to
        # if (xorf(T, det_sign) < 0.0f), where det_sign = sign_mask(det) in Woop's paper
        # please refer https://www.allthingsphi.com/blog/2016/12/18/watertight-ray-triangle-intersection.html
        # for detailed information on "sign_mask" and "xorf"
        if (det > 0.0 and T < 0.0) or (det < 0.0 and T > 0.0): return None

        inv_det = 1.0/det
        return T*inv_det

    # end of _intersect3D_ray_triangle_watertight function

    def _gen_intercepts(self, imp, rmin, rmax, ncells, coord, dx, boxlo, boxhi, vertices):
        """
        generate the x-, y-, or z-intercepts based on vertices.
        If successful, the generated intercepts will be passed
        to "produce" intercepts array ready for use by warp
        """
        m = ncells[imp[1]]; n = ncells[imp[2]]
        ax1 = coord[imp[1]]; ax2 = coord[imp[2]]

        # create (m+1)by(n+1) list
        # for each element: intercepts[m][n] = [ [], [] ]
        intercepts = [ [ [ [], [] ] for _ in range(n+1) ] for _ in range(m+1) ]
        
        r0 = [0, 0, 0]; r1 = [0, 0, 0]
        r0[imp[0]] = rmin; r1[imp[0]] = rmax
        faces = self.surface.faces
        for face in faces:
            T = Triangle(vertices[face])
            T_boxlo = np.amin(T.vertices(), axis=0)
            T_boxhi = np.amax(T.vertices(), axis=0)

            T_boxlo_idx = np.amax([[0, 0, 0], np.int_(np.floor((T_boxlo - boxlo)/dx))], axis=0)
            T_boxhi_idx = np.amin([ncells,    np.int_(np.ceil ((T_boxhi - boxlo)/dx))], axis=0)

            for i in range(T_boxlo_idx[imp[1]], T_boxhi_idx[imp[1]]+1):
                r0[imp[1]] = ax1[i]; r1[imp[1]] = ax1[i]
                for j in range(T_boxlo_idx[imp[2]], T_boxhi_idx[imp[2]]+1):
                    r0[imp[2]] = ax2[j]; r1[imp[2]] = ax2[j]
                    L = LineSegment([r0, r1])
                    I = self._intersect3D_lineseg_triangle(L, T)

                    if isinstance(I, np.ndarray): intercepts[i][j][0].append(I[imp[0]])
                    elif isinstance(I, list):                   # R lies in T plane
                        if len(I) > 0: intercepts[i][j][1].append( (I[0][imp[0]], I[1][imp[0]]) )
                    elif I == -1:
                        return -1

        return intercepts

    # end of _gen_intercepts function

    def _produce_intercepts(self, intercepts):
        """
        produce the intercepts array ready for use by warp
        the returned array is column major in favor of Fortran.
        """

        if intercepts == -1: return None, False

        decimal = self._precision_decimal
        if not decimal: decimal = 4
        if not self._normalization_factor: expr = 'e'
        else: expr = 'f'

        m = len(intercepts)
        n = len(intercepts[0])
        data_list = []

        nicpt_max = 0
        for i in range(m):
            for j in range(n):
                pos = i*n + j
                
                d = np.unique([self._round_float(icpt, decimal, expr) for icpt in intercepts[i][j][0]])
                if len(d) % 2 == 1:
                    return None, False

                data_list.append([a for a in d])
                for k in range(len(intercepts[i][j][1])):
                    data_list[pos].extend(intercepts[i][j][1][k])

                nicpt_max = max(nicpt_max, len(data_list[pos]))

        # copy items stored in data_list to the clean intercepts array, which is to be returned
        if nicpt_max > 1:
            c_intercepts = np.full((nicpt_max, m, n), self._infinity)
            for i in range(m):
                for j in range(n):
                    pos = i*n + j
                    data_list[pos].sort()

                    icpt = 0
                    for d in data_list[pos]:
                        c_intercepts[icpt, i, j] = d
                        icpt += 1
        else:
            # if no intercepts, make an array with 2 items whose value being infinity on each mesh point
            # so warp gridintercept function can handle this
            c_intercepts = np.full((2, m, n), self._infinity)

        return c_intercepts, True

    # end of _produce_intercepts function
  
# end of class STLconductor
