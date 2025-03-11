import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial
import networkx as nx
import matplotlib.pyplot as plt
import multiprocessing as mp

from cnetwork import MagmaChamber

def prop_factory(t_b=1e11, t_d=5e10, K_crust=10e9, G_crust=10e9,
                 K_f=10e9, rho0=2500, mu0=1e6, r_hydr=5):
  return dict(
    t_b = t_b,     # Set Maxwell times
    t_d = t_d,     # Set Maxwell times
    K_crust = K_crust,
    G_crust = G_crust,
    K_f = K_f,
    rho0 = rho0,
    mu0 = mu0,      # Constant viscosity assumption
    r_hydr = r_hydr,     # Effective hydraulic radius
  )

def compute_conductivity(list_nodes, max_edge_dist) -> scipy.sparse.coo_matrix:
  ''' Construct conductivity matrix with units 1/length
  Linearized flow admittance ( (m/s) / Pa ) due to laminar pipe flow is given by
    r_hydr * r_hydr / 16.0 / mu0 * Y[i,j]
  ''' 

  # Legacy algorithm
  # x = np.array([node.x for node in list_nodes])
  # y = np.array([node.y for node in list_nodes])
  # z = np.array([node.z for node in list_nodes])
  # # Compute distance into temp
  # temp = np.sqrt((x - x[:,np.newaxis]) ** 2
  #          + (y - y[:,np.newaxis]) ** 2
  #          + (z - z[:,np.newaxis]) ** 2)

  coords = np.array([[node.x, node.y, node.z] for node in list_nodes])
  # Compute pairwise distances into Y
  Y = scipy.spatial.distance.cdist(coords, coords)
  # Mask out edges based on threshold
  np.multiply(0.0, Y, where=Y > max_edge_dist, out=Y)
  # Compute 1/distance into Y where nonzero
  np.divide(1.0, Y, where=Y>0, out=Y)
  return scipy.sparse.coo_matrix(Y)

def compute_conductivity_ij(list_nodes, max_edge_dist, i, j):
  ''' 1/(edge length) between node i and node j
  '''
  if i == j:
    return 0.0

  node_i = list_nodes[i]
  node_j = list_nodes[j]
  # Compute distance
  dist = np.sqrt((node_i.x - node_j.x) ** 2
                  + (node_i.y - node_j.y) ** 2
                  + (node_i.z - node_j.z) ** 2)
  if np.isclose(dist, 0.0):
    raise ValueError("Distance between chamber {i} and {j} is zero.")
  elif dist > max_edge_dist:
    return 0.0
  else:
    return 1.0 / dist

class RandomGraph():
  def __init__(self, node_list, max_edge_dist,
               add_virtual_source_sink=False,
               source_sink_conductivity=1e2):
    self.node_list = node_list
    self.Y = compute_conductivity(node_list, max_edge_dist)
    if add_virtual_source_sink:
      # Add edge with infinite conductance for extra (source/sink) nodes
      self.Y[0,:] = 0.0
      self.Y[:,0] = 0.0
      self.Y[-1,:] = 0.0
      self.Y[:,-1] = 0.0
      # Source connectivity ~infty
      self.Y[0,1] = source_sink_conductivity
      self.Y[1,0] = source_sink_conductivity
      self.Y[-1,-2] = source_sink_conductivity
      self.Y[-2,-1] = source_sink_conductivity

    self.V = len(node_list)
    self.G = nx.Graph(self.Y)
    self.L = nx.laplacian_matrix(self.G)

  def draw_network(self, length_scale=1.0, node_size=200):
     nx.draw_networkx(self.G, pos = {i:(n.x / length_scale, n.z / length_scale,)
                for (i, n) in enumerate(self.node_list)}, node_size=node_size)
  
  def edge_conductivity_scaling(self, props):
    ''' Scaling factor to convert 1/length to hydraulic conductivity
    units ( (Pa) / (m/s))'''
    r_hydr = props["r_hydr"]
    mu0 = props["mu0"]
    return r_hydr * r_hydr / 16.0 / mu0

  def edge_conductivity_list(self, get_ij=False) -> tuple:
    ''' Return array of non-zero conductivity of each edge from the
    lower triangular part of Y.
    If get_ij is True, returns array of tuples (edge_cond, i, j) instead. '''
    if get_ij:
      return (self.Y.data, self.Y.row, self.Y.col)
    else:
      return self.Y.data
  
    # Legacy dense matrix algorithm
    # edge_conductance = []
    # for i in range(self.Y.shape[0]):
    #   for j in range(i):
    #     edge_conductance.append(self.Y[i,j])
    # return np.array(edge_conductance)

  def graph_conductivity(self, use_dense_matrix=False):
    if not nx.has_path(self.G, 0, self.V-1):
      return 0.0
    # Construct s-t test vector
    chi = np.zeros((self.L.shape[0],))
    chi[0] = 1.0
    chi[-1] = -1.0

    if use_dense_matrix:
      # Compute dense pseudo inverse
      iL = scipy.linalg.pinv(self.L.todense())
      # Compute conductivity (reciprocal of s-t resistance)
      st_resistance = np.einsum("i, ij, j ->", chi, iL, chi)
    else:
      v, exit_code = scipy.sparse.linalg.cg(self.L, chi)
      if exit_code != 0:
        raise ValueError(f"scipy.sparse.linalg.cg terminated with unsuccessful exit code {exit_code}.")
      st_resistance = np.dot(chi, v)

    return 1.0 / st_resistance
  
  def compute_st_potentials(self, s:int=0, t:int=-1):
    if not nx.has_path(self.G, 0, self.V-1):
      return np.full((self.V,), np.nan)
    # Construct s-t test vector
    chi = np.zeros((self.V,))
    chi[s] = 1.0
    chi[t] = -1.0
    v, exit_code = scipy.sparse.linalg.cg(self.L, chi)
    if exit_code != 0:
      raise ValueError(f"scipy.sparse.linalg.cg terminated with unsuccessful exit code {exit_code}.")
    return v

  def draw_matrix_spy(self):
    fig, ax = plt.subplots(1 ,3, figsize=(11,7))
    ax[0].matshow(self.Y)
    ax[0].set_title("Y")
    ax[1].matshow(self.L.todense())
    ax[1].set_title("L")
    ax[2].matshow(scipy.linalg.pinv(self.L.todense()))
    ax[2].set_title(r"$L^{\dagger}$")
    return fig, ax

# Network parameters
def make_1D_uniform_network(N_chamber, z_scale=40e3, R=500,) -> list:
  # Uniformly randomly distribution volume
  V = (4/3)*np.pi*R**3
  # List generation of chamber characteristics
  x_nodes = np.zeros((N_chamber,))
  y_nodes = np.zeros((N_chamber,))
  z_nodes = np.linspace(0.0, z_scale, N_chamber)
  V_nodes = V * np.ones((N_chamber,))

  return [MagmaChamber(x=x, y=y, z=z,
                       p_setting=None, T_setting=1200, V_setting=V)
    for (x,y,z,V) in zip(x_nodes, y_nodes, z_nodes, V_nodes)]

def make_2D_grid_network(N_x, N_z, x_scale=10e3, z_scale=40e3,
                         R=500,
                         add_source_sink=False) -> list:
  # Set planar geometry (y-scale = 0)
  y_scale = 0.0
  # Compute totla number of chambers in grid
  N_chamber = N_x * N_z
  # Generate chamber coordinates
  z_axis = np.linspace(0, z_scale, N_z)
  # Shuffle index so center node starts
  _x1 = np.arange(N_x//2-1, -1, -1)
  _x2 = np.arange(N_x//2, N_x, )
  _ind = np.empty((N_x,), dtype=int)
  _ind[::2] = _x2
  _ind[1::2] = _x1
  x_axis = np.linspace(0, x_scale, N_x)[_ind]

  mg_x, mg_z = np.meshgrid(x_axis, z_axis)
  x_nodes = mg_x.ravel()
  y_nodes = y_scale * np.zeros((N_chamber,))
  z_nodes = mg_z.ravel()

  # Swap -1 index with the center node at the top layer
  x_nodes[-1], x_nodes[-N_x] = x_nodes[-N_x], x_nodes[-1]
  z_nodes[-1], z_nodes[-N_x] = z_nodes[-N_x], z_nodes[-1]

  V_nodes = (4/3)*np.pi*R**3 * np.ones((N_chamber,))

  list_nodes = [MagmaChamber(x=x, y=y, z=z,
                       p_setting=None, T_setting=1200, V_setting=V)
    for (x,y,z,V) in zip(x_nodes, y_nodes, z_nodes, V_nodes)]
  
  if add_source_sink:
    # Append virtual source and sink to list_nodes
    z_margin = 2000
    source = MagmaChamber(x=0.0, y=0.0, z=-z_margin,
      p_setting=None, T_setting=1200, V_setting=1e9)
    sink = MagmaChamber(x=0.0, y=0.0, z=z_scale + z_margin,
      p_setting=None, T_setting=1200, V_setting=1e9)
    list_nodes = [source, *list_nodes, sink]

  return list_nodes

def make_2D_random_network(N_chamber, x_scale=10e3, z_scale=40e3,
                           R_min=500, R_max=500,
                           random_state=None,
                           add_source_sink=False) -> list:
  # Set planar geometry (y-scale = 0)
  y_scale = 0.0
  # Uniformly randomly distribution volume
  V_min = (4/3)*np.pi*R_min**3
  V_max = (4/3)*np.pi*R_max**3
  # List generation of chamber characteristics
  if random_state is None:
    x_nodes = x_scale * np.random.rand(N_chamber)
    y_nodes = y_scale * np.random.rand(N_chamber)
    z_nodes = z_scale * np.random.rand(N_chamber)
    V_nodes = V_min + (V_max - V_min) * np.random.rand(N_chamber)
  else:
    x_nodes = x_scale * random_state.rand(N_chamber)
    y_nodes = y_scale * random_state.rand(N_chamber)
    z_nodes = z_scale * random_state.rand(N_chamber)
    V_nodes = V_min + (V_max - V_min) * random_state.rand(N_chamber)

  # Force one node at bottom, one at top
  z_nodes[0] = 0.0
  z_nodes[-1] = z_scale
  # Center pinning
  x_nodes[0] = x_scale * 0.5
  x_nodes[-1] = x_scale * 0.5

  list_nodes = [MagmaChamber(x=x, y=y, z=z,
                       p_setting=None, T_setting=1200, V_setting=V)
    for (x,y,z,V) in zip(x_nodes, y_nodes, z_nodes, V_nodes)]
  
  if add_source_sink:
    # Append virtual source and sink to list_nodes
    z_margin = 2000
    source = MagmaChamber(x=0.0, y=0.0, z=-z_margin,
      p_setting=None, T_setting=1200, V_setting=1e9)
    sink = MagmaChamber(x=0.0, y=0.0, z=z_scale + z_margin,
      p_setting=None, T_setting=1200, V_setting=1e9)
    list_nodes = [source, *list_nodes, sink]

  return list_nodes

def compute_graph_conductivity(N_nodes, x_scale, z_scale, max_edge_dist, R_min, R_max):
  ''' Compute one realization of random network's graph conductivity '''
  node_list = make_2D_random_network(
    N_nodes, x_scale=x_scale, z_scale=z_scale, R_min=R_min, R_max=R_max,)
  RG = RandomGraph(node_list, max_edge_dist)
  return RG.graph_conductivity()

def initializer():
  ''' RNG initializer for process pool. '''
  global rng
  rng = np.random.default_rng()

def run_2D_ensemble(ensemble_size, N_nodes, x_scale, z_scale,
                    max_edge_dist, R_min, R_max, seed=None, processes:int=0):
  ''' Run 2D random graph ensemble, returning vector of effective graph
  conductivity values with unit 1/length. '''
  if processes == 0 or processes == 1:
    if seed is not None:
      np.random.seed(seed)    
    out_vec = np.zeros((ensemble_size))
    for i in range(ensemble_size):
      out_vec[i] = compute_graph_conductivity(N_nodes, x_scale, z_scale, max_edge_dist, R_min, R_max)
  else:
    if seed is not None:
      raise NotImplementedError("No implementation for seeded multiprocess. Use single process for seeded.")

    # Set inputs
    inputs = [(ensemble_size, x_scale, z_scale, max_edge_dist, R_min, R_max,) for _ in range(ensemble_size)]
    # Run on pool
    with mp.Pool(processes=processes, initializer=initializer) as pool:
      out_vec = pool.starmap(compute_graph_conductivity, inputs)  
  
  return out_vec

if __name__ == "__main__":
  pass