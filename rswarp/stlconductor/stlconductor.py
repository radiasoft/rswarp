from __future__ import division
import pymesh
from warp import *
from copy import deepcopy
import warnings

class Ray:
    """
    class to manage ray and line segment instance
    """
    def __init__(self, points):
        if np.asarray(points).shape != (2,3):
            raise TypeError("Input vertices must be a container having 2 points")
        self._points = np.array(points)

    def point(self, idx):
        return self._points[idx]

    def points(self):
        return self._points

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
                       precision_decimal=None, normalization_factor=None, fuzz=None,
                       disp="none", condid='next', **kw):
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
        self._fuzz = fuzz
        self._winding_fuzz = 1e-14          # fuzzy for small scale winding number
        self._disp = disp
        self._infinity = np.inf
#         self._infinity = largepos
        if fuzz: self._fuzz2 = fuzz*fuzz

        self.surface = pymesh.load_mesh(filename)

        self._surface_mesh_quality_check()

        if disp != "none" and disp != "auto":
            try:
                if len(np.asarray(disp)) != 3: raise RuntimeError("")
            except:
                err_msg = "invalid type/values passed to argument [disp].\n"
                err_msg += "options for disp can be \"none\", \"auto\", or an array with 3 entries."
                raise RuntimeError(err_msg)

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
    
    def round_float(self, f, n, expr):
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
                surface_boxlo[i] = self.round_float(surface_boxlo[i], self._precision_decimal, expr)
                surface_boxhi[i] = self.round_float(surface_boxhi[i], self._precision_decimal, expr)
                m, n = vertices.shape
                for i in range(m):
                    for j in range(n):
                        vertices[i,j] = self.round_float(vertices[i,j], self._precision_decimal, expr)

        if not self._fuzz :
            self._fuzz = np.amin(dx)*1e-6
            self._fuzz2 = self._fuzz*self._fuzz
#         print "self._fuzz =", self._fuzz

        x = np.linspace(boxlo[0], boxhi[0], nx+1)
        y = np.linspace(boxlo[1], boxhi[1], ny+1)
        z = np.linspace(boxlo[2], boxhi[2], nz+1)

        surface_box_extent = surface_boxhi - surface_boxlo

        print " ---  (nx,ny,nz)=({},{},{}), (dx,dy,dz)=({},{},{})".format(nx,ny,nz,dx[0],dx[1],dx[2])
        print " ---  boxlo=({},{},{}), boxhi=({},{},{})".format(boxlo[0],boxlo[1],boxlo[2],boxhi[0],boxhi[1],boxhi[2])
        print " ---  surface_boxlo=({},{},{}), surface_boxhi=({},{},{})".format(surface_boxlo[0],surface_boxlo[1],surface_boxlo[2],
                                                                          surface_boxhi[0],surface_boxhi[1],surface_boxhi[2])

        eps = 5e-4*dx0

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
                    raise RuntimeError(msg)

            xintercepts = self._gen_intercepts(imp, xmin, xmax, [nx, ny, nz], [x, y, z], dx, boxlo, boxhi, vertices+disp)
            xi, successful = self._produce_intercepts(xintercepts, 'x')

            if successful: break

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
                    raise RuntimeError(msg)

            yintercepts = self._gen_intercepts(imp, ymin, ymax, [nx, ny, nz], [x, y, z], dx, boxlo, boxhi, vertices+disp)
            yi, successful = self._produce_intercepts(yintercepts, 'y')

            if successful: break

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
                    raise RuntimeError(msg)

            zintercepts = self._gen_intercepts(imp, zmin, zmax, [nx, ny, nz], [x, y, z], dx, boxlo, boxhi, vertices+disp)
            zi, successful = self._produce_intercepts(zintercepts, 'z')

            if successful: break

        if self._normalization_factor:
            xi *= self._normalization_factor
            yi *= self._normalization_factor
            zi *= self._normalization_factor

        return xi, yi, zi

    # end of _conductorfnew_impl function


    def _intersect3D_ray_triangle(self, ray, triangle):
        """
        compute the intersection points of a ray and triangle
        """

        # get triangle edge vectors and plane normal
        u = triangle.vertex(1) - triangle.vertex(0)
        v = triangle.vertex(2) - triangle.vertex(0)
        n = triangle.normal()

        if np.all(n == np.zeros(3)): return None                # triangle is degenerate

        rd = ray.point(1) - ray.point(0)                        # ray direction vector
        w0 = ray.point(0) - triangle.vertex(0)
        a  = -np.dot(n, w0)
        b  = np.dot(n, rd)

#         if np.abs(b) < self._fuzz:                              # ray is parallel to triangle plane
#             return None
        if np.abs(b) < self._fuzz2:                              # ray is parallel to triangle plane
            if np.abs(a) < self._fuzz2:                          # ray lies in triangle plane
                return self._intersect_triangle_edges(ray, triangle)
            else:                                               # ray disjoint from plane
                return None

        # get intersect point of ray w ith triangle
        r = a/b
        r = self.round_float(r, 10, 'f')

        if r < 0: return None                                   # no intersect with this ray
        if r > 1: return None                                   # no intersect with this line segment

        I = ray.point(0) + r*rd                                 # intersect point of ray and plane

        # test if intersect inside triangle
        uu = np.dot(u, u)
        uv = np.dot(u, v)
        vv = np.dot(v, v)
        w  = I - triangle.vertex(0)
        wu = np.dot(w, u)
        wv = np.dot(w, v)
        D  = uv*uv - uu*vv

        s  = (uv*wv - vv*wu)/D
        if s < 0 or s > 1: return None                          # I is outside triangle
        t  = (uv*wu - uu*wv)/D
        if t < 0 or (s+t) > 1: return None                      # I is outside triangle

        return I                                                # line segment intersects triangle at I

    # end of _intersect3D_ray_triangle function

    def _intersect_triangle_edges(self, ray, triangle):
        """
        compute the possible intersection points of a ray and all the 3 triangle edges
        when the ray is in the same plane with the triangle
        """
        intercepts = []
        for i in range(3):
            tri_edge = Ray([triangle.vertex(i), triangle.vertex((i+1) % 3)])
            I = self._intersect_2segments(ray, tri_edge)
            if I is not None:
                if isinstance(I, list):
                    for i in I: intercepts.append(i)
                else:
                    intercepts.append(I)

        nicpt = len(intercepts)

        # nicpt == 0 - no intercepts
        # ray and triangle are parallel but do not lie in the same plane

        # nicpt == 1 - should never happen
        if nicpt == 1:
            return -1
#             err_msg  = "Failed to generate intercepts for segment[{}, {}].\n".format(ray.point(0), ray.point(1))
#             err_msg += "This happens when the segment lies in the plane of surface element, and cannot be perfectly resolved due to round-off errors.\n"
#             err_msg += "A possible solution is to move the object by a tiny amount of displacement through \"disp=[x, y, z]\" option so segment is off the plane.\n"
#             err_msg += "Normalizing through \"normalization_factor=dx\" option is also recommended."
#             raise RuntimeError(err_msg)

        # nicpt == 2 should be the most common situation
        # just return, nothing to do

        # nicpt == 3 - rare situation
        # this happens when ray intersects triangle with
        # one of its vertices and the opposite side
        # nicpt == 4 - furthre rare situation
        # this happens when ray and one of the triangle edegs are collinear
        # for nicpt == 3 or 4, remove the duplicated intercepts
        if nicpt == 3 or nicpt == 4:
            n = self._precision_decimal
            if not n: n = 4     # default value if precision_decimal is not defined

            if not self._normalization_factor: expr = 'e'
            else: expr = 'f'

            for i in range(len(intercepts)) :
                for j in range(3) :
                    intercepts[i][j] = self.round_float(intercepts[i][j], n, expr)

            I = np.unique(np.array(intercepts), axis=0)
            intercepts = [i for i in I]
#             print "intercepts = {}".format(intercepts)

        if len(intercepts) > 2:
            return -1
#             err_msg  = "Failed to generate intercepts for segment[{}, {}].\n".format(ray.point(0), ray.point(1))
#             err_msg += "This happens when the segment lies in the plane of surface element, and cannot be perfectly resolved due to round-off errors.\n"
#             err_msg += "A possible solution is to move the object by a tiny amount of displacement through \"disp=[x, y, z]\" option so segment is off the plane.\n"
#             err_msg += "Normalizing through \"normalization_factor=dx\" option is also recommended."
#             raise RuntimeError(err_msg)

        return intercepts

    # end of _intersect_triangle_edges function

    def _intersect_2segments(self, seg1, seg2):
        """
        compute intersection point of two line segments.
        (Note: the two input segments are already on the same plane
                so this function can be called. )
        """

        # project the plane formed by seg1 and seg2
        # to xy-, yz- or zx-plane
        e1 = seg1.point(1) - seg1.point(0)
        e2 = seg2.point(1) - seg2.point(0)

        n = np.cross(e1, seg2.point(0)-seg1.point(0))
        if np.linalg.norm(n) == 0.:      # seg1.point(0), seg1.point(1) and seg2.point(0) are collinear
            n = np.cross(e1, seg2.point(1)-seg1.point(0))

        if np.linalg.norm(n) == 0.:     # seg1 and seg2 are collinear
            for j in range(3):
                if e1[j] != 0.: break
            k = (j+1)%3
        else:
            if np.abs(np.dot(n, [0., 0., 1.])) != 0.:           # not perpendicular to xy-plane
                j=0; k=1
            else:
                if np.abs(np.dot(n, [1., 0., 0.])) != 0.:       # not perpendicular to yz-plane
                    j=1; k=2
                else:
                    j=2; k=0

#         print "n1={}, n2={}".format(n, np.cross(e1, seg2.point(0)-seg1.point(0)))

        u = np.array([0.,0.])
        v = np.array([0.,0.])
        w = np.array([0.,0.])
        u[0] = seg1.point(1)[j] - seg1.point(0)[j]
        u[1] = seg1.point(1)[k] - seg1.point(0)[k]
        v[0] = seg2.point(1)[j] - seg2.point(0)[j]
        v[1] = seg2.point(1)[k] - seg2.point(0)[k]
        w[0] = seg1.point(0)[j] - seg2.point(0)[j]
        w[1] = seg1.point(0)[k] - seg2.point(0)[k]

        D1 = self._perp2D(u, v)
        is_parallel = np.abs(D1) < self._fuzz2
#         print "n={}, j={}, k={}, u={}, v={}, w={}, D1={}, is_parallel={}".format(n, j, k, u, v, w, D1, is_parallel)

        if is_parallel:
            if self._perp2D(u, w) != 0 or self._perp2D(v, w) != 0:
                return None                                     # seg1 and seg2 are not collinear
            
            # seg1 and seg2 are collinear or degenerate
            # first check if they are degenerate points
            du = np.dot(u, u)
            dv = np.dot(v, v)

            if du == 0 and dv == 0:                             # both are points
                if np.all(u[0] == v[0]):                        # they are distinct points
                    return None
                else:                                           # they are the same point
                    return u[0]

            if du == 0:                                         # seg1 is a single point
                # this cannot happen
                return None

            if dv == 0:
                # this cannot happen
                return None

            # seg1 and seg2 are collinear (not degenerate points)
            # get their overlap or not
            w2 = np.array([0.,0.])
            w2[0] = seg1.point(1)[j] - seg2.point(0)[j]
            w2[1] = seg1.point(1)[k] - seg2.point(0)[k]
            if v[0] != 0:
                t0 = w[0]/v[0]; t1 = w2[0]/v[0]
            else:
                t0 = w[1]/v[1]; t1 = w2[1]/v[1]

            if t0 > t1:                                    # t0 must be smaller than t1
                t = t0; t0 = t1; t1 = t0                   # swap if not

            if t0 > 1 or t1 < 0:                           # no overlap
                return None

            t0 = max(0, t0)                                # clip to min 0
            t1 = min(1, t1)                                # clip to max 1
            if t0 == t1:                                   # intersect is a point
                return seg2.point(0) + t0*e2

            # they overlap in a valid subsegment
            return [seg2.point(0) + t0*e2, seg2.point(0) + t1*e2]

        # the segments are skew and may intersect in a point

        # get the intersect parameter for seg1
        D2 = self._perp2D(v, w)
#         print "u={}, v={}, w={}, D1={}, D2={}".format(u, v, w, D1, D2)
        sI = D2/D1

        if sI < 0 or sI > 1:
            return None

        # get the intersect parameter for S2
        D3 = self._perp2D(u, w)

#         print "D3={}".format(D3)
        tI = D3/D1

        if tI < 0 or tI > 1:
            return None

        return seg1.point(0) + sI*e1

    #end of _intersect_2segments function

    # perp product
    def _perp2D(self, u, v):
        return u[0]*v[1] - u[1]*v[0]

    def _in_segment(self, p, s):
        """
        check if point p is in a collinear segment s:
        """
        if s.point(0)[0] != s.point(1)[0]:                 # s is not parallel to x-axis
            if s.point(0)[0] <= p[0] and p[0] <= s.point(1)[0]:
                return True
            if s.point(1)[0] <= p[0] and p[0] <= s.point(0)[0]:
                return True
        elif s.point(0)[1] != s.point(1)[1]:               # s is not parallel to y-axis
            if s.point(0)[1] <= p[1] and p[1] <= s.point(1)[1]:
                return True
            if s.point(1)[1] <= p[1] and p[1] <= s.point(0)[1]:
                return True
        elif s.point(0)[2] != s.point(1)[2]:               # s is not parallel to z-axis
            if s.point(0)[2] <= p[2] and p[2] <= s.point(1)[2]:
                return True
            if s.point(1)[2] <= p[2] and p[2] <= s.point(0)[2]:
                return True

        return False

    # end of _in_segment function

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
                    R = Ray([r0, r1])
                    I = self._intersect3D_ray_triangle(R, T)

                    if isinstance(I, np.ndarray): intercepts[i][j][0].append(I[imp[0]])
                    elif isinstance(I, list):                   # R lies in T plane
                        if len(I) > 0: intercepts[i][j][1].append( (I[0][imp[0]], I[1][imp[0]]) )
                    elif I == -1:
                        return -1

        return intercepts
    # end of _gen_intercepts function

    def _produce_intercepts(self, intercepts, info):
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
                
                d = np.unique([self.round_float(icpt, decimal, expr) for icpt in intercepts[i][j][0]])
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
