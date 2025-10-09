''' Discrete boundary element code 

Implementation in this file is for the 2-D (planar) heat equation.

See also prototyping notebooks:
  * DBEM_operator.ipynb

  
Issues:

The matrix multiplication L@u is computed on-the-fly, so that the dense matrix
does not need to be stored. This is very slow, even for the forward
multiplication. This may be remedied by off-loading the work to C or performing
matrix multiplication by blocks instead of row-wise for improved vectorization.

For matrix inverse, consider MINRES with Jacobi preconditioner (effective due to
the singular part (1/r terms, where r is scalar distance) being large compraed
to the regular part).


'''

import numpy as np
import scipy.optimize
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib
from dataclasses import dataclass

# Meshing dependency
import gmsh
# Mesh reader (for .msh files)
import meshio

def generate_annulus_mesh_to_file(msh_file_name,
    dx=0.2, annulus_inner_radius=0.25, annulus_outer_radius=1.5,
    show_gmsh_window=False) -> None:
  ''' Generates mesh and saves to file name `msh_file_name`.
  This mesh is a 2D annulus with specified inner radius and outer radius.
  Parameter dx specificies the (approximate) mesh size.
  Depends on module gmsh. '''

  gmsh.initialize()

  # Define collection of points
  points = [
    # Bounding box vertices
    gmsh.model.geo.addPoint(-0.5, -0.5, 0, dx),
    gmsh.model.geo.addPoint(0.5,  -0.5, 0, dx),
    gmsh.model.geo.addPoint(0.5,  0.5,  0, dx),
    gmsh.model.geo.addPoint(-0.5, 0.5,  0, dx),
    # Inner circle points
    gmsh.model.geo.addPoint(0.0, 0, 0, dx),
    gmsh.model.geo.addPoint(annulus_inner_radius, 0, 0, dx),
    gmsh.model.geo.addPoint(-annulus_inner_radius, 0, 0, dx),
    # Outer circle points
    gmsh.model.geo.addPoint(0.0, 0.0, 0, dx),
    gmsh.model.geo.addPoint(annulus_outer_radius, 0.0, 0, dx),
    gmsh.model.geo.addPoint(-annulus_outer_radius, 0.0, 0, dx),
  ]
  # Define collection of straight lines (unused if only circles are needed)
  lines = [
    # Box lines
    # gmsh.model.geo.add_line(points[0], points[1]),
    # gmsh.model.geo.add_line(points[1], points[2]),
    # gmsh.model.geo.add_line(points[2], points[3]),
    # gmsh.model.geo.add_line(points[3], points[0]),
  ]
  # Define collection of circular arcs
  arcs = [
    # Inner circle
    gmsh.model.geo.addCircleArc(points[5], points[4], points[6]),
    gmsh.model.geo.addCircleArc(points[6], points[4], points[5]),
    # Outer circle
    gmsh.model.geo.addCircleArc(points[8], points[7], points[9]),
    gmsh.model.geo.addCircleArc(points[9], points[7], points[8]),
  ]
  # Define curve loops
  loops = [
    # Bounding box loop
    # gmsh.model.geo.add_curve_loop(lines),
    # Inner circle loop
    gmsh.model.geo.add_curve_loop(arcs[0:2]),
    # Outer circle loop
    gmsh.model.geo.add_curve_loop(arcs[2:4]),
  ]
  # Define 2D surface
  surfaces = [
    # Bounding box and inner circle
    # gmsh.model.geo.add_plane_surface([loops[0], loops[1]]),
    # Outer circle and inner circle
    gmsh.model.geo.add_plane_surface([loops[1], loops[0]]),
  ]

  # Define physical groups
  surfaces_physical = [gmsh.model.addPhysicalGroup(2, surfaces, name="surface")]
  # lines_physical = [gmsh.model.addPhysicalGroup(1, lines, name=f"box")]
  inner_arcs_physical = [gmsh.model.addPhysicalGroup(1, arcs[0:2], name=f"circle_in")]
  outer_arcs_physical = [gmsh.model.addPhysicalGroup(1, arcs[2:4], name=f"circle_out")]

  gmsh.model.geo.synchronize()
  gmsh.model.mesh.generate()
  gmsh.write(msh_file_name)

  if show_gmsh_window:
    import sys
    if 'close' not in sys.argv:
        gmsh.fltk.run()

  gmsh.finalize()

def generate_network_mesh_to_file(msh_file_name,
    dx=0.1, 
    chamber_radius=0.2,
    outer_radius=5.0,
    chamber_spacing_scale=7.5,
    N_chambers=4,
    seed=None,
    show_gmsh_window=False) -> np.array:
  ''' Generates mesh and saves to file name `msh_file_name`.
  This mesh is a 2D disk with circular holes cut out of it.
  Parameter dx specificies the (approximate) mesh size.
  Depends on module gmsh.
  
  Returns array of chamber centers (x,y).
  '''

  # Chamber radius

  # Outer domain boundary radius
  a = chamber_radius
  b = outer_radius

  # Generate random centers
  if seed is None:
    ''' No seeding '''
  else:
    np.random.seed(6)
  # Array of (x,y) coordinates
  xy = chamber_spacing_scale * (
    np.random.rand(N_chambers, 2,) - 0.5)

  gmsh.initialize()

  bdry_points = [
    # Bounding box vertices
    gmsh.model.geo.addPoint(-0.5, -0.5, 0, dx),
    gmsh.model.geo.addPoint(0.5,  -0.5, 0, dx),
    gmsh.model.geo.addPoint(0.5,  0.5,  0, dx),
    gmsh.model.geo.addPoint(-0.5, 0.5,  0, dx),
    # Outer circle points
    gmsh.model.geo.addPoint(0.0, 0.0, 0, dx),
    gmsh.model.geo.addPoint(b, 0.0, 0, dx),
    gmsh.model.geo.addPoint(-b, 0.0, 0, dx), 
  ]

  chamber_points = []
  for i in range(xy.shape[0]):
    chamber_points.extend((
      gmsh.model.geo.addPoint(xy[i,0], xy[i,1], 0, dx),
      gmsh.model.geo.addPoint(xy[i,0] + a, xy[i,1], 0, dx),
      gmsh.model.geo.addPoint(xy[i,0] - a, xy[i,1], 0, dx),
    ))

  lines = [
    # Box lines
    # gmsh.model.geo.add_line(points[0], points[1]),
    # gmsh.model.geo.add_line(points[1], points[2]),
    # gmsh.model.geo.add_line(points[2], points[3]),
    # gmsh.model.geo.add_line(points[3], points[0]),
  ]

  arcs_outer = [
    # Cirle arcs for outer circle
    gmsh.model.geo.addCircleArc(bdry_points[5], bdry_points[4], bdry_points[6]),
    gmsh.model.geo.addCircleArc(bdry_points[6], bdry_points[4], bdry_points[5]),
  ]

  arcs_chambers = []
  for i in range(xy.shape[0]):
    arcs_chambers.extend((
      gmsh.model.geo.addCircleArc(chamber_points[3*i+1], chamber_points[3*i], chamber_points[3*i+2]),
      gmsh.model.geo.addCircleArc(chamber_points[3*i+2], chamber_points[3*i], chamber_points[3*i+1]),
    ))

  loops = [
    # Bounding box loop
    # gmsh.model.geo.add_curve_loop(lines),
    # Outer circle loop
    gmsh.model.geo.add_curve_loop(arcs_outer),
  ]
  # Magma chamber boundary loops
  for i in range(xy.shape[0]):
    loops.append(gmsh.model.geo.add_curve_loop(arcs_chambers[2*i:2*i+2]))

  surfaces = [
    # Bounding box and inner circle
    # gmsh.model.geo.add_plane_surface([loops[0], loops[1]]),
    # Outer circle and inner circle
    gmsh.model.geo.add_plane_surface([loops[0], *loops[1:]]),
  ]

  # Physical groups
  surfaces_physical = [gmsh.model.addPhysicalGroup(2, surfaces, name="surface")]
  # lines_physical = [gmsh.model.addPhysicalGroup(1, lines, name=f"box")]
  outer_arcs_physical = [gmsh.model.addPhysicalGroup(1, arcs_outer, name=f"circle_out")]
  inner_arcs_physical = [gmsh.model.addPhysicalGroup(1, arcs_chambers, name=f"circle_in")]

  for i in range(N_chambers):
    chambers_physical = [gmsh.model.addPhysicalGroup(1, arcs_chambers[2*i:2*i+2], name=f"chamber_{i}")]

  gmsh.model.geo.synchronize()
  gmsh.model.mesh.generate()
  gmsh.write(msh_file_name)

  if show_gmsh_window:
    import sys
    if 'close' not in sys.argv:
        gmsh.fltk.run()

  gmsh.finalize()

  return xy

def extract_points(mesh, tag:str, reverse=False, make_plot=False) -> tuple:
  ''' Extracts points from a physical group in a mesh object generated by gmsh.

  Input:
    mesh: mesh object read from meshio.read
    tag (str): name of the physical surface to extract points from; this name
      is provided when generating the mesh using gmsh.model.addPhysicalGroup.
    reverse (optional): whether to reverse the list of edges. Useful for
      boundaries in the interior of the domain (as opposed to boundaries on
      the outside).
    make_plot(bool): If True, plots the extracted points to current plt.axes.

  Returns tuple (points, edges).
  
  '''
  # Read list of arrays containing edges [node_idx_i, node_idx_j] for lines or
  # cell (hyperedge) [node_idx_i, node_idx_j, node_idx_k] for tris
  edges = [mesh.cells[i].data[idx_set] for i, idx_set in enumerate(mesh.cell_sets[tag])]
  # Flatten collection of edges into array with shape (ne, 2) belonging to boundary
  edges_flat = np.concatenate([part for part in edges if part.size > 0])
  if reverse:
    edges_flat = np.flip(edges_flat, axis=1)
  # Compute list of coordinates belonging to each edge (n_edges, 2 (vertices on edge), 3 (dimension))
  points = mesh.points[edges_flat]

  if make_plot:
    # Plot the identified points and print shape
    plt.scatter(points[...,0].ravel(), points[...,1].ravel())
    print(f"Points array shape: {points.shape}")
  return points, edges_flat

def geotherm(z, gradient=25):
  return -gradient * z + 500

def G(r):
  ''' Laplace fundamental solution '''
  return -1 / (4.0 * np.pi) * np.log(np.einsum("...i, ...i -> ...", r, r))

def G_vec(r):
  ''' Vectorized Laplace fundamental solution '''
  r2 = np.einsum("...i, ...i -> ...", r, r)
  np.log(r2, out=r2, where=r2>0)
  return r2 / (-4.0 * np.pi)

def dGdn(r, nhat):
  ''' d/dn of Laplace fundamental solution '''
  rnorm = np.linalg.norm(r,axis=-1)
  drdn = np.einsum("...i, ...i -> ...", -nhat, r/rnorm)
  return -1 / (2.0 * np.pi * rnorm) * drdn

def dGdn_vec(r, nhat):
  ''' Vectorized d/dn of Laplace fundamental solution '''
  r2 = np.einsum("...i, ...i -> ...", r, r)
  np.divide(np.einsum("...i, ...i -> ...", -nhat, r),
            -2.0 * np.pi * r2, out=r2, where=r2>0)
  return r2

def rotate_vec(v):
  ''' Rotate vector counterclockwise 90 degrees along last axis '''
  w = v.copy()
  w[...,[1,0]] = w[...,[0,1]]
  w[...,0] *= -1
  return w

def int_r_G(R):
  ''' r * G partially integrated in r from 0 to r = R '''
  return (-1 / (2.0 * np.pi)) * R * R * (2 * np.log(R) - 1.0) / 4.0

def integrateG_ws(x_vertices, N_per_side=3, debug=False, method=2):
  ''' Integrates G in triangle including weak singularity.
  Polar coordinate integration dr dtheta. Analytic integration in
  dr direction, numerical integration in theta(r).
  
  Due to curvature approximation of the dtheta element, max expected order is 2.
  '''
  theta_tot = 0
  I = 0
  # Compute vertex coordinates relative to centroid
  x_shift = x_vertices - x_vertices.mean(axis=0)
  for _ii in range(3):
    # Compute vector representing edge of triangle
    _dx = x_shift[_ii] - x_shift[_ii-1]
    for _jj in range(N_per_side):
      # Trap integration along this edge
      interval_start  = x_shift[_ii-1] + (_jj) * _dx / N_per_side
      interval_center = x_shift[_ii-1] + (_jj + 0.5) * _dx / N_per_side
      interval_end    = x_shift[_ii-1] + (_jj + 1) * _dx / N_per_side
      r_start  = np.linalg.norm(interval_start)
      r_center = np.linalg.norm(interval_center)
      r_end    = np.linalg.norm(interval_end)
      theta_start = np.arctan2(interval_start[1], interval_start[0])
      theta_end   = np.arctan2(interval_end[1], interval_end[0])
      dtheta = np.mod(theta_end - theta_start, 2*np.pi)
      if method==3:
        dI = (int_r_G(r_start) + 4 * int_r_G(r_center) + int_r_G(r_end)) * dtheta / 6.0
      elif method==2:
        dI = (0.5 * int_r_G(r_start) + 0.5 * int_r_G(r_end)) * dtheta
      else:
        raise ValueError("Input param `method` options: {2, 3}.")
      I += dI
      if debug:
        print(dtheta, dI)
        theta_tot += dtheta
        plt.plot(interval_center[0], interval_center[1], '.')

  if debug:
    plt.plot(0, 0, 'ko')
    print(f"Total angle: {theta_tot/np.pi} * pi")
  return I


@dataclass
class MeshGeom:
  ''' Data struct for mesh geometry.
  
  Contains the information needed to manipulate the mesh geometry in the
  context of the boundary element method.
  '''

  # Indices for tri vertices (ntris, 3,)
  tri_indices: np.array
  # 3D coordinates for tri vertices (ntris, 3, ndim=3,)
  tri_points: np.array
  # 3D coordinates for tri centroids (ntris, ndim=3,)
  tri_centroids: np.array
  # Area of tris (ntris,)
  tri_areas: np.array

  # Indices for inner boundary segments (n_in, 2,)
  circ_in_indices: np.array
  # 3D coordinates for inner boundary segment endpoints (n_in, 2, 3)
  circ_in_points: np.array
  # Indices for outer boundary segments (n_out, 2,)
  circ_out_indices: np.array
  # 3D coordinates for outer boundary segment endpoints (n_out, 2, 3)
  circ_out_points: np.array
  # Indices for all boundary segments (n_in + n_out, 2, 3)
  bdry_points: np.array
  # 3D coordinates for all boundary center points (n_in + n_out, 3)
  bdry_edge_centers: np.array

  # Boolean checking if tri is on inner boundary (ntris,)
  is_cell_on_bdry_circ_in: np.array
  # Boolean checking if tri is on outer boundary (ntris,)
  is_cell_on_bdry_circ_out: np.array
  
  node_id_circ_in: np.array
  node_id_circ_out: np.array

  # Counts
  n_circ_in_elts: np.array
  n_circ_out_elts: np.array
  n_nodes: np.array
  n_cells: np.array
  
  # Original mesh object
  mesh: object


class MeshOperators():
  ''' Operators L, f for a given mesh, setting up the equation
    L @ u = f,
  where L is a linear operator, u is the solution vector, and f is the data
  vector. This equation is the discretization of the heat equation using the
  theta-method time discretization (theta = 1: implicit Euler, theta = 0: 
  explicit Euler).

    geom: MeshGeom object containing cell and boundary information.
     '''
  def __init__(self, geom:MeshGeom, N_per_side=10, plot_normal_vecs=False,):
    # Geometry processing
    self.n_cells = geom.tri_centroids.shape[0]
    self.n_bdry_node = geom.n_circ_in_elts + geom.n_circ_out_elts 
    self.N = geom.n_cells + self.n_bdry_node
    self.geom = geom
    # Assemble global x points as list (N, 2)
    self.x_nodes = np.concatenate((geom.bdry_edge_centers[:,0:2],
                            geom.tri_centroids[:,0:2],), axis=0)

    # Set integration opption
    self.N_per_side = 10

    # Directed boundary element dg (dgamma) with shape (n_bdry_node, 2,)
    self.dg = (geom.bdry_points[:,1,0:2] - geom.bdry_points[:,0,0:2])
    # Measure of each boundary element with shape (n_bdry_node,) 
    self.dg_size = np.linalg.norm(self.dg, axis=-1)
    # Compute outward normal unit vectors with shape (n_bdry_node, 2,)
    self.dn = rotate_vec(self.dg)
    np.divide(self.dn, np.linalg.norm(self.dn, axis=-1, keepdims=True),
              out=self.dn) 
    
    # Compute singular integral
    self.di_singular = self.compute_di_singular()

    # Option to plot normal vectors
    if plot_normal_vecs:
      for j in range(self.n_bdry_node):
        plt.arrow(geom.bdry_edge_centers[j,0], geom.bdry_edge_centers[j,1],
                  self.dn[j,0], self.dn[j,1], head_width=0.06)
    
  def compute_di_singular(self):
    ''' Compute singular part of domain integral '''
    # Shift vertex coordinates relative to centroid (n_cells, 3, 2,)
    x_shift = (self.geom.tri_points[:,:,0:2]
               - self.geom.tri_points[:,:,0:2].mean(axis=1, keepdims=True))
    # Compute vectors representing triangle edges
    _dx = x_shift[:,[1,2,0],:] - x_shift
    # Set vector of variables in [0, 1] parametrizing distance along edge
    xi = np.expand_dims(
      np.linspace(0, 1, self.N_per_side, endpoint=False) / self.N_per_side,
      axis=(1,2,3))
    # Compute interval start points with shape (N_per_side, n_cells, 3, 2)
    interval_start = x_shift + xi * _dx
    # Compute interval end points with shape (N_per_side, n_cells, 3, 2)
    interval_end = x_shift + (xi + 1/self.N_per_side) * _dx
    # Convert interval endpoints to (r, theta)
    r_start = np.linalg.norm(interval_start, axis=-1)
    r_end = np.linalg.norm(interval_end, axis=-1)
    theta_start = np.arctan2(interval_start[...,1], interval_start[...,0])
    theta_end = np.arctan2(interval_end[...,1], interval_end[...,0])
    # Compute dtheta, wrapping to [0, 2*pi]
    dtheta = np.mod(theta_end - theta_start, 2*np.pi)
    # Compute second-order quadrature of weakly singular part as vector of
    # size (n_cells,) into bottom part of M (tall matrix; bottom part will become
    # block diagonal of larger square matrix)
    return np.einsum(
      "i...k, i...k -> ...",
      0.5 * int_r_G(r_start) + 0.5 * int_r_G(r_end), dtheta)

  def L(self, u, dt, theta=1.0):
    ''' Compute LHS matrix as linear transformation '''

    beta_d = 1.0

    u = u.squeeze()

    # Allocate output and temp vectors
    out = np.zeros_like(u)
    temp = u.copy()
    # Apply diagonal mass part, signed for the boundary term
    temp *= np.concatenate((-self.dg_size, self.geom.tri_areas/dt))

    # Multiply boundary part of u
    for i in range(self.N):
      out[i] = np.dot(
        G_vec(self.x_nodes[i,:] - self.geom.bdry_edge_centers[:,0:2]),
        temp[:self.n_bdry_node])
      
      out[i] += np.dot(
        G_vec(self.x_nodes[i,:] - self.geom.tri_centroids[:,0:2]),
        temp[self.n_bdry_node:])

    def G_singular_1D(r):
      ''' Singular part integrated in 1D '''
      return 2 * (-1 / (2.0 * np.pi)) * (r * np.log(r) - r)

    # Add singular part elementwise multiplying u
    out += np.concatenate((-G_singular_1D(self.dg_size/2),
                           self.di_singular/dt + beta_d * theta)) * u
    
    return out

  def L_matrix(self, dt, theta=1.0):
    ''' Returns dense LHS matrix L. '''

    # Geometric constant
    BETA_D = 1.0

    def G_singular_1D(r):
        ''' Singular part integrated in 1D '''
        return 2 * (-1 / (2.0 * np.pi)) * (r * np.log(r) - r)

    # Compute G(x_i - x_j), with zeros on diagonal
    G = G_vec(self.x_nodes[:,np.newaxis,:] - self.x_nodes)
    # Compute G(x _i - x_j) weighted against appropriate measure
    out = np.einsum("ij, j -> ij", G,
      np.concatenate((-self.dg_size, self.geom.tri_areas/dt)))
    # Add diagonal singular part
    out[np.arange(self.N), np.arange(self.N)] += np.concatenate((
      -G_singular_1D(self.dg_size/2),
      self.di_singular/dt + BETA_D * theta))
    
    return out

  def f(self, u, u_bdry, dt, theta=1.0):
    ''' Compute RHS '''
    beta_b = 0.5
    beta_d = 1.0
    
    u = u.squeeze()
    u_bdry = u_bdry.squeeze()

    out = np.zeros_like(u)

    # Boundary data dependent part
    for i in range(self.N):
      out[i] = np.dot(
        -dGdn_vec(self.x_nodes[i,:] - self.geom.bdry_edge_centers[:,0:2], self.dn),
        u_bdry * self.dg_size) 
    out[:self.n_bdry_node] -= beta_b * u_bdry

    boundary_part = out.copy()
    
    # Apply diagonal mass part, signed for the boundary term
    temp = u.copy()
    temp *= np.concatenate((0.0 * self.dg_size, self.geom.tri_areas/dt))

    # Domain dependent part
    for i in range(self.N):
      out[i] += np.dot(
        G_vec(self.x_nodes[i,:] - self.geom.tri_centroids[:,0:2]),
        temp[self.n_bdry_node:])

    out[self.n_bdry_node:] += (self.di_singular/dt
                               + beta_d * (theta - 1.0)) * u[self.n_bdry_node:]
      
    interior_part = out - boundary_part

    return boundary_part, interior_part, out
    
def read_netlike_mesh(msh_file_name) -> MeshGeom:
  ''' Reads mesh for annulus or network geometry and returns MeshGeom object.
  
  This function is specific to these geometries because of the naming of
  physical groups in the .msh file. The boundary consists of a circle_in and
  circle_out. The circle_in boundary corresponds in the case of the annulus
  to the inner radius, and corresponds in the case of the chamber network to
  the union of all chamber boundaries. The circle_out boundary corresponds to 
  the outer circular boundary.
  To adapt the reading to an arbitrary geometry, change out the physical
  curves where the points are extracted using extract_points.

  '''

  # Import .msh file
  mesh = meshio.read(msh_file_name)

  n_nodes = mesh.points.shape[0]
  n_cells = np.concatenate(mesh.cell_sets["surface"]).size

  # Extract points in gmsh physical groups
  tri_points, tri_indices = extract_points(
    mesh, "surface", reverse=False)
  circ_in_points, circ_in_indices = extract_points(
    mesh, "circle_in", reverse=False)
  circ_out_points, circ_out_indices = extract_points(
    mesh, "circle_out", reverse=True)
  # Compute centroids by cell (ncells)
  tri_centroids = tri_points.mean(axis=1, keepdims=False)
  # Merge arrays of boundary cell-centers
  n_circ_in_elts = circ_in_points.shape[0]
  n_circ_out_elts = circ_out_points.shape[0]
  bdry_points = np.concatenate([circ_in_points, circ_out_points], axis=0)
  # Compute array of centers for edges on the boundary of the domain
  bdry_edge_centers = bdry_points.mean(axis=1, keepdims=False)

  ''' Compute boundary membership of cells '''
  # IDs of nodes that lie on the circular boundary
  node_id_circ_in = np.unique(circ_in_indices.ravel())
  node_id_circ_out = np.unique(circ_out_indices.ravel())

  # Compute boolean array showing which cell ID is on the boundaries
  is_cell_on_bdry_circ_in = np.empty((tri_indices.shape[0]), dtype=bool)
  is_cell_on_bdry_circ_out = np.empty((tri_indices.shape[0]), dtype=bool)
  for i in range(tri_indices.shape[0]):
    # Check membership of any bdry node along axis 1
    if len(np.intersect1d(tri_indices[i,:], node_id_circ_in)) == 0:
      is_cell_on_bdry_circ_in[i] = False
    else:
      is_cell_on_bdry_circ_in[i] = True
    # Check membership of any bdry node along axis 1
    if len(np.intersect1d(tri_indices[i,:], node_id_circ_out)) == 0:
      is_cell_on_bdry_circ_out[i] = False
    else:
      is_cell_on_bdry_circ_out[i] = True

  ''' Compute tri-element areas '''
  tri_areas = 0.5 * np.abs(np.cross(tri_points[:,2,:] - tri_points[:,0,:],
          tri_points[:,1,:] - tri_points[:,0,:])[:,-1])
  tri_areas.sum(), np.pi*(1.5**2 - 0.25**2)

  return MeshGeom(
    tri_indices,
    tri_points,
    tri_centroids,
    tri_areas,
    circ_in_indices,
    circ_in_points,
    circ_out_indices,
    circ_out_points,
    bdry_points,
    bdry_edge_centers,
    is_cell_on_bdry_circ_in,
    is_cell_on_bdry_circ_out,
    node_id_circ_in,
    node_id_circ_out,
    n_circ_in_elts,
    n_circ_out_elts,
    n_nodes,
    n_cells,
    mesh)

def assemble(geom:MeshGeom, plot_normal_vecs=False, N_per_side=10):
  ''' Assemble matrices K, C, and M. These matrices appear in the DBEM equation

  M @ du/dt == K @ du/dn + dK/dn @ u
  
  Inputs:
    geom: MeshGeom object
  
    plot_normal_vecs (bool, optional): sets whether a plot is created, showing the outward
    normal direction.
    N_per_side (default=10): number of quadrature points per boundary element.

  '''

  bdry_edge_centers = geom.bdry_edge_centers
  bdry_points = geom.bdry_points
  tri_centroids = geom.tri_centroids
  tri_points = geom.tri_points
  tri_areas = geom.tri_areas

  n_bdry_node = bdry_edge_centers.shape[0]
  n_cells = tri_centroids.shape[0]

  N = n_cells + n_bdry_node

  K = np.zeros((N, n_bdry_node)) # Single-layer potential coefficient matrix (K @ du/dn)
  C = np.zeros((N, n_bdry_node)) # Double-layer potential coefficient matrix (dK/dn @ u)
  M = np.zeros((N, n_cells))     # Volume integral (dK/dt @ u)

  # Assemble global x points as list (bdry_centers, tri_centers)
  x_nodes = np.concatenate((bdry_edge_centers[:,0:2],
                            tri_centroids[:,0:2],), axis=0)
  
  ''' Compute boundary contributions

  Compute regular parts of operators K, C that multiply dudn and u respectively.
  Then compute singular part of K using explicit integration. Singular part
  of C is zero since dr/dn is zero on the boundary.

  To access the jth boundary element, use bdry_points[j,:,0:2] for the vertices
  and bdry_edge_centers[j,0:2] for the boundary center.
  '''
  # Matrix of all r-vectors with shape (n_bdry_nodes + n_cells, n_bdry_nodes, 2)
  r_all = (x_nodes[:,np.newaxis,:] - bdry_edge_centers[np.newaxis,:,0:2])
  # Directed boundary element dg (dgamma) with shape (n_bdry_node, 2,)
  dg = (bdry_points[:,1,0:2] - bdry_points[:,0,0:2])
  # Measure of each boundary element with shape (n_bdry_node,) 
  dg_size_all = np.linalg.norm(dg, axis=-1)
  # Compute outward normal unit vectors with shape (n_bdry_node, 2,)
  dn_all = rotate_vec(dg)
  dn_all /= np.linalg.norm(dn_all, axis=-1)[:,np.newaxis]

  # Vectorized computation of K = G(r) * dgamma_j, operator for @ dudn
  K = G_vec(r_all) * dg_size_all
  # Vectorized computation of C = dG/dn(r) * dgamma_j, operator for @ u
  C = dGdn_vec(r_all, dn_all) * dg_size_all
  # Compute singular boundary integrals using exact integration around pole
  for i in range(n_bdry_node):
    r_eff = (dg_size_all[i]/2)
    K[i,i] = 2 * (-1 / (2.0 * np.pi)) * (r_eff * np.log(r_eff) - r_eff)

  # Option to plot normal vectors
  if plot_normal_vecs:
    for j in range(n_bdry_node):
      plt.arrow(bdry_edge_centers[j,0], bdry_edge_centers[j,1],
                dn_all[j,0], dn_all[j,1], head_width=0.06)

  ''' Compute volume contributions

  Compute regular parts of operator M multiplies u_interior.
  Then compute singular part of M using explicit integration in r, numerically
  integrating in angle theta. 

  To access the jth cell, use x_vertices[j,:,0:2] for the vertices
  and tri_centroids[j,0:2] for the boundary center.
  '''

  # Matrix of all r-vectors with shape (n_bdry_nodes + n_cells, n_cells, 2)
  r_all = (x_nodes[:,np.newaxis,:] - tri_centroids[np.newaxis,:,0:2])
  # Vectorized computation of M = G(r) * dOmega_j, operator for @ u_interior
  M = G_vec(r_all) * tri_areas

  # Shift vertex coordinates relative to centroid (n_cells, 3, 2,)
  x_shift = (tri_points[:,:,0:2] - tri_points[:,:,0:2].mean(axis=1, keepdims=True))
  # Compute vectors representing triangle edges
  _dx = x_shift[:,[1,2,0],:] - x_shift
  # Set vector of variables in [0, 1] parametrizing distance along edge
  xi = np.expand_dims(
    np.linspace(0, 1, N_per_side, endpoint=False) / N_per_side, axis=(1,2,3))
  # Compute interval start points with shape (N_per_side, n_cells, 3, 2)
  interval_start = x_shift + xi * _dx
  # Compute interval end points with shape (N_per_side, n_cells, 3, 2)
  interval_end = x_shift + (xi + 1/N_per_side) * _dx
  # Convert interval endpoints to (r, theta)
  r_start = np.linalg.norm(interval_start, axis=-1)
  r_end = np.linalg.norm(interval_end, axis=-1)
  theta_start = np.arctan2(interval_start[...,1], interval_start[...,0])
  theta_end = np.arctan2(interval_end[...,1], interval_end[...,0])
  # Compute dtheta, wrapping to [0, 2*pi]
  dtheta = np.mod(theta_end - theta_start, 2*np.pi)
  # Compute second-order quadrature of weakly singular part as vector of
  # size (n_cells,) into bottom part of M (tall matrix; bottom part will become
  # block diagonal of larger square matrix)
  M[np.arange(0,n_cells) + n_bdry_node, np.arange(0,n_cells)] = np.einsum(
    "i...k, i...k -> ...",
    0.5 * int_r_G(r_start) + 0.5 * int_r_G(r_end), dtheta)

  return K, C, M

def det(m, a, b):
  ''' Determinant of matrix M in
    [0, 0]^T = M(m) @ [c_J, c_Y]^T,
  which is the eigenfunction condition for a Dirichlet problem in a 2D annulus.
  The general solution is a linear combination
    c_J * J0(r) + c_Y * Y0(r),
  and the linear system above admits non-trivial solutions when det M == 0.
  '''
  return (scipy.special.j0(m*a) * scipy.special.y0(m*b)
          - scipy.special.j0(m*b) * scipy.special.y0(m*a))

def det_dm(m, a, b):
  ''' Derivative of function `det` with respect to argument m. '''
  return -(
    a * scipy.special.j1(m*a) * scipy.special.y0(m*b)
    + b * scipy.special.j0(m*a) * scipy.special.y1(m*b)
    - b * scipy.special.j1(m*b) * scipy.special.y0(m*a)
    - a * scipy.special.j0(m*b) * scipy.special.y1(m*a)
  )

def precompute_eigenfunction_roots(a, b, N_roots=15):
  ''' Returns array of eigenvalues corresponding to the heat equation
  eigenfunction on a 2-D annulus. '''
  # Approximate with cos(0.5 * m * (b-a)) and estimate roots of det with pi / (b-a)
  m_guesses = np.arange(1,N_roots+1) * np.pi / (b - a)
  # Allocate vector of roots
  roots = np.zeros_like(m_guesses)
  # Precompute roots
  for i, n in enumerate(m_guesses):
    roots[i] = scipy.optimize.fsolve(
      lambda m: det(m, a, b),
      m_guesses[i],
      fprime=lambda m: det_dm(m, a, b))
  return roots

def annulus_steady_term(r, t, a, b, dim=2):
  ''' Steady term in series solution for annulus heat conduction (dim==2),
  or spherical shell (dim==3).
   
  Returns steady term in series expansion for T(r,t) for a <= r <= b.
  '''
  if dim == 2:
    return np.log(r / b) / np.log(a / b)
  elif dim == 3:
    return a / (a - b) * (1 - b / r)

def annulus_series_term(r, t, a, b, n, dim=2, alpha=1):
  ''' Time-dependent term in series solution for annulus heat conduction (2D,
  default), or spherical shell (3D).
  
  Returns n-th transient term (n >= 1) in series expansion for
  T(r,t) for a <= r <= b. Requires array of `roots` of the Sturm-Liouville
  problem (c_J * J0(r) + c_Y * Y0(r) == 0) in the 2D case, defaulting to
  precomputed root set.
  
  '''

  if n < 1:
    raise ValueError("Series terms are defined for integer n >= 1.")

  if dim == 2:
    # If root precomputed:
    # if n < 1 or n > roots.size:
      # raise ValueError(f"n = {n} not valid. Need n = 1 to {roots.size} (or compute more roots).")
    
    # Compute eigenvalue numerically using Newton iteration
    eig = float(scipy.optimize.fsolve(
      lambda m: det(m, a, b),
      n * np.pi / (b - a),
      fprime=lambda m: det_dm(m, a, b)))
    # Compute coefficients using r = a boundary condition
    cj = 1.0
    cy = - cj * scipy.special.j0 (eig * a) / scipy.special.y0(eig * a)
    # Time-dependent factor, with diffusivity TODO: for other functions
    t_factor = np.exp(- alpha * eig * eig * t)
    # Space-dependent factor
    r_factor = cj * scipy.special.j0(eig * r) + cy * scipy.special.y0(eig * r)
    return t_factor * r_factor
  elif dim == 3: 
    # Array of eigenvalues
    eig = n * np.pi / (b - a)
    # Return eigenfunction term with nondimensionalizing scaling (b - a) / 2
    return (b - a) / 2 * np.exp(-eig*eig * t) * np.sin(eig*(r - a)) / r

def annulus_steady_ddr(r, t, a, b, dim=2):
  ''' Spatial derivative (d/dr) of annulus_steady_term.
   
  Returns steady term in series expansion for dT/dr(r,t) for a <= r <= b.
  '''
  if dim == 2:
    return 1 / r / np.log(a / b)
  elif dim == 3:
    return  a * b / (a - b) / (r * r)

def annulus_series_ddr(r, t, a, b, n, dim=2, alpha=1.0):
  ''' Spatial derivative (d/dr) of annulus_series_term.
  
  Returns the n-th transient term (n >= 1) in series expansion for
  dT/dr(r,t) for a <= r <= b. Requires array of `roots` of the Sturm-Liouville
  problem (c_J * J0(r) + c_Y * Y0(r) == 0) in the 2D case, defaulting to
  precomputed root set.
  
  '''

  if n < 1:
    raise ValueError("Series terms are defined for integer n >= 1.")

  if dim == 2:
    # Compute eigenvalue numerically using Newton iteration
    eig = float(scipy.optimize.fsolve(
      lambda m: det(m, a, b),
      n * np.pi / (b - a),
      fprime=lambda m: det_dm(m, a, b)))
    # Compute coefficients using r = a boundary condition
    cj = 1.0
    cy = - cj * scipy.special.j0 (eig * a) / scipy.special.y0(eig * a)
    # Time-dependent factor, with diffusivity TODO: for other functions
    t_factor = np.exp(- alpha * eig * eig * t)
    # Space-dependent factor
    r_factor = -eig * (cj * scipy.special.j1(eig * r) + cy * scipy.special.y1(eig * r))
    return t_factor * r_factor
  elif dim == 3:
    # Array of eigenvalues
    eig = n * np.pi / (b - a)
    # Return eigenfunction term with nondimensionalizing scaling (b - a) / 2
    return (b - a) / 2 * np.exp(-eig*eig * t) * (
        eig * np.cos(eig*(r - a)) / r
        - np.sin(eig*(r - a)) / (r*r)
      )

def generate_annulus_IC(ops:MeshOperators, a, b):
  ''' Sets initial condition equal to the steady-state temperature plus
  the first eigenfunction for an annulus with inner radius a and outer
  radius b. '''

  geom = ops.geom
  n_bdry_node = ops.n_bdry_node
  n_circ_in_elts = geom.n_circ_in_elts
  n_circ_out_elts = geom.n_circ_out_elts

  # Set initial conditions
  r_eval = np.linalg.norm(geom.tri_centroids, axis=1)
  # Evaluate data in cell
  IC_cell_data = (annulus_steady_term(r_eval, 0.0, a, b)
                  + annulus_series_term(r_eval, 0.0, a, b, 1))
  # Evaluate Neumann data on boundary
  IC_bdry_circ_in_data = -(
      annulus_steady_ddr(a, 0.0, a, b) + annulus_series_ddr(a, 0.0, a, b, 1)
    ) * np.ones((geom.node_id_circ_in.shape))
  IC_bdry_circ_out_data = (
      annulus_steady_ddr(b, 0.0, a, b) + annulus_series_ddr(b, 0.0, a, b, 1)
    ) * np.ones((geom.node_id_circ_out.shape))

  # Fill data into mixed vector
  u0 = np.zeros((ops.N, ))
  # First, add data for boundary of inner circle
  u0[0:n_circ_in_elts] = IC_bdry_circ_in_data
  # Second, add data for boundary of outer circle
  u0[n_circ_in_elts:n_circ_in_elts+n_circ_out_elts] = IC_bdry_circ_out_data
  # Finally, add data for interior cells
  u0[n_bdry_node:] = IC_cell_data
  # Reshape initial condition
  u0 = u0[:,np.newaxis]

  return u0

def generate_network_IC(ops:MeshOperators, dx, gradient=25):
  ''' Sets initial condition equal to the ambient temperature gradient.
   '''

  geom = ops.geom
  n_bdry_node = ops.n_bdry_node
  n_circ_in_elts = geom.n_circ_in_elts
  n_circ_out_elts = geom.n_circ_out_elts
  bdry_points = geom.bdry_points

  # Matrix of all r-vectors with shape (n_bdry_nodes + n_cells, n_bdry_nodes, 2)
  r_all = (ops.x_nodes[:,np.newaxis,:]
           - geom.bdry_edge_centers[np.newaxis,:,0:2])
  # Directed boundary element dg (dgamma) with shape (n_bdry_node, 2,)
  dg = (bdry_points[:,1,0:2] - bdry_points[:,0,0:2])
  # Measure of each boundary element with shape (n_bdry_node,) 
  dg_size_all = np.linalg.norm(dg, axis=-1)
  # Compute outward normal unit vectors with shape (n_bdry_node, 2,)
  dn_all = rotate_vec(dg)
  dn_all /= np.linalg.norm(dn_all, axis=-1)[:,np.newaxis]

  # Set initial conditions
  r_eval = np.linalg.norm(geom.tri_centroids, axis=1)
  # Evaluate data in cell
  IC_cell_data = geotherm(ops.x_nodes[n_bdry_node:, 1])
  # Evaluate Neumann data on inner boundary (approx)
  IC_bdry_circ_in_neumann = 1.0/dx
  # Evaluate Neumann data on boundary (geothermal gradient)
  IC_bdry_circ_out_neumann = (-gradient * dn_all[n_circ_in_elts:, 1])

  # Fill data into mixed vector
  u0 = np.zeros((ops.N, ))
  # First, add data for boundary of inner circle
  u0[0:n_circ_in_elts] = IC_bdry_circ_in_neumann
  # Second, add data for boundary of outer circle
  u0[n_circ_in_elts:n_circ_in_elts+n_circ_out_elts] = IC_bdry_circ_out_neumann
  # Finally, add data for interior cells
  u0[n_bdry_node:] = IC_cell_data
  # Reshape initial condition
  u0 = u0[:,np.newaxis]

  return u0

def generate_dirichlet_BC(ops:MeshOperators, val=1.0):
  ''' Set constant Dirichlet data (value = val) on the inner boundary and 0
  zero at all other points of the boundary. '''
  # Set 0 as default boundary value
  u_bdry = np.zeros((ops.n_bdry_node, 1))
  # Set value val on inner boundary
  u_bdry[0:ops.geom.n_circ_in_elts] = val
  return u_bdry

def generate_network_BC(ops:MeshOperators):
  ''' Set boundary conditions equal to the geotherm '''
  n_bdry_node = ops.n_bdry_node
  n_circ_in_elts = ops.geom.n_circ_in_elts
  n_circ_out_elts = ops.geom.n_circ_out_elts
  
  u_bdry = np.zeros((n_bdry_node,1))
  # Set Dirichlet data on the inner boundary (access using concatenation order)
  u_bdry[0:ops.geom.node_id_circ_in.size] = 1000 # 1.0
  # Set Dirichlet data on the outer boundary
  circ_out_slice = slice(n_circ_in_elts, n_circ_in_elts+n_circ_out_elts)
  u_bdry[circ_out_slice] = geotherm(ops.x_nodes[circ_out_slice, 1:2])

  return u_bdry

def build_explicit_matrices(ops, dt=1e-2, theta=1.0, show_matrices=True):
  ''' Constructs and shows sparsity pattern of the left hand side and right
  hand side matrices of the discrete DBEM problem. The discrete problem,
  obtained after approximating in space and time, solves for the vector
    U_k = [du/dn, u] at timestep k
  from the equation
    L @ U_k == A @ (du/dn)_bdry + D @ U_{k-1},
  where L, A, and D are matrices, (du/dn) is the vector of boundary gradient
  data, and U_{k-1} is the vector at the previous timestep.

  The matrices L, A a

  This function is for the user to learn the structure of the problem, not for
  direct computation.
  
  '''

  K, C, M = assemble(ops.geom)

  # Geometric constants
  BETA_D = 1.0
  BETA_B = 0.5

  # Get parameters
  N = ops.N
  n_bdry_node = ops.n_bdry_node
  n_cells = ops.geom.n_cells

  # Compute left hand side matrix L for inversion
  L = np.concatenate((-K, M/dt), axis=1)
  L[n_bdry_node:,n_bdry_node:] += BETA_D * theta * np.eye(n_cells)

  # Construct full-sized matrices for right hand side
  B = -C.copy()
  B[:n_bdry_node,:] -= BETA_B * np.eye(n_bdry_node)
  F = (M/dt).copy()
  F[n_bdry_node:,:] += BETA_D * (theta - 1.0) * np.eye(n_cells)
  F = np.concatenate((np.zeros((N, n_bdry_node)), F), axis=1)

  if show_matrices:
    fig, ax = plt.subplots(1, 4, figsize=(14,6))
    ax[0].matshow(L)
    ax[0].set_title("LHS matrix")
    ax[1].matshow(L[0:n_bdry_node, 0:n_bdry_node])
    ax[1].set_title("LHS matrix, top-left corner")
    ax[2].matshow(B)
    ax[2].set_title("Residual mapping from boundary data")
    ax[3].matshow(F)
    ax[3].set_title("Residual mapping from $u^{k-1}$")

    print("At each timestep, the algebraic equation being solved is ")
    print("  LU == f")
    print("where U is the vector containing du/dn and u at the timestep k,")
    print("and f == B(du/dn) + FU_{k-1}.")
    print("Left two plots: sparsity pattern of the matrix L")
    print("Third plot: sparsity pattern of B, mapping boundary data to RHS.")
    print("Fourth plot: sparsity pattern of F, mapping U_{k-1} data to RHS.")
    print("Note the LHS matrix and mappings from previous timestep are dense")

  return L, B, F

def get_chamber_global_indices(geom, N_chambers):
  # Build map from pair of node indices to index
  edge_to_index = {tuple(v): i for i, v
                   in enumerate(geom.circ_in_indices)}
  chamber_global_indices = [None for i in range(N_chambers)]
  for i in range(N_chambers):
    # Get boundary edges as pairs of node indices
    _, bdry_edge_indices = extract_points(geom.mesh, f"chamber_{i}",
                                          reverse=False, make_plot=False)
    # Map pairs to indices for boundary edge degrees of freedom
    chamber_global_indices[i] = np.array([edge_to_index[tuple(edge)]
                                          for edge in bdry_edge_indices])
  return chamber_global_indices

def time_dependent_solve(ops:MeshOperators, u0, u_bdry, dt, N_t, theta=1.0):
  ''' Time-dependent simulation of the heat equation using DBEM method.
    #   L @ q^{k} == CG @ u_Gamma + DG @ q^{k-1}
  Requires setup of the operators on the mesh (ops).

  Returns the time series of du/dn on the boundary degrees of freedom and the
  time series of u for the interior degrees of freedom. 

    du/dn has shape (N_t, ops.geom.n_circ_in_elts + ops.geom.n_circ_out_elts, 1)
    u_interior has shape (N_t, ops.geom.n_cells, 1)

  '''

  u = u0.copy()
  # Allocate storage for u at each time step
  u_hist = np.zeros((N_t, *u0.shape,))

  geom = ops.geom
  N = ops.N
  n_bdry_node = ops.n_bdry_node
  n_circ_in_elts = geom.n_circ_in_elts

  ''' Dense L solve '''

  # Fill L matrix from operator form by multiplying by each unit vector
  L = ops.L_matrix(dt, theta)
  # LU factorization
  lu, piv = scipy.linalg.lu_factor(L)

  for i in range(N_t):
    # Solve equation
    # u = np.linalg.solve(L, C_op @ u_bdry + D_op @ u)
    # using prefactorized form
    u = scipy.linalg.lu_solve((lu, piv), ops.f(u, u_bdry, dt, theta)[2])
    u_hist[i,...] = u[:,np.newaxis]

  # Extract du/dn on boundary and u_interior (u for interior points)
  dudn = u_hist[:, :n_bdry_node]
  u_interior = u_hist[:, n_bdry_node:]

  return dudn, u_interior
