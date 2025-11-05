import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial
import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import networkx as nx
g = 10

class Node():
  ''' Base graph node class '''
  def __init__(self):
    pass

def p_lithostatic(z, p_surf=1e5, z_surf=0, rho_crust=2.5e3, g=10):
  ''' Lithostatic pressure as function of depth z. '''
  return p_surf - rho_crust * g * (z - z_surf)

def p_hydrostatic(z, p_surf=1e5, z_surf=0, rho_magma=2.5e3, g=10):
  ''' Hydrostatic pressure as function of depth z. '''
  return p_surf - rho_magma * g * (z - z_surf)

def T_geothermal(z, T_surf=273.15, z_surf=0, grad=-10/1e3):
  ''' Crust geothermal temperature as function of depth z.
  Default gradient is (10 K / km). '''
  return T_surf + grad * (z - z_surf)

def zero_aligned_cmap(clim):
  ''' Return a Colormap object where the neutral color is aligned with 
  zero value in the given clim interval.
  Input:
    clim: tuple (min, max) representing the limits of the color data '''
  # Compute clipping interval for the colorbar
  if clim[0] >= 0:
    cinterval = (0.5, 1.0) # Use full range but the correct color tone
  elif clim[1] <= 0:
    cinterval = (0.0, 0.5) # Use full range but the correct color tone
  elif 0.5 * (clim[0] + clim[1]) > 0:
    cinterval = (0.5 * (1 + clim[0] / clim[1]), 1)
  else:
    cinterval = (0.0, 0.5 * (1 - clim[1] / clim[0]))
  # Shifted diverging cmap
  return matplotlib.colors.LinearSegmentedColormap.from_list(
      f'trunc(bwr,{cinterval[0]},{cinterval[1]})',
      matplotlib.cm.bwr(np.linspace(cinterval[0], cinterval[1], 1000)))

def smoother(x, scale):
  ''' Returns one-sided compact smoothed step, such that
    1. u(x < -scale) = 0
    2. u(x >= 0) = 1.
    3. u smoothly interpolates from 0 to 1 in between.
  '''
  # Shift, scale, and clip to [-1, 0] to prevent exp overflow
  if scale != 0:
    _x = np.clip(x / scale + 1, 0, 1)
  else:
    _x = np.where(x >= 0, 1, 0)
  f0 = np.exp(-1/np.where(_x == 0, 1, _x))
  f1 = np.exp(-1/np.where(_x == 1, 1, 1-_x))
  # Return piecewise evaluation
  return np.where(_x >= 1, 1,
         np.where(_x <= 0, 0, 
         f0 / (f0 + f1)))

def op_D(h, Nr):
    ''' Central first-derivative operator '''
    upper = 0.5/h*np.ones(Nr-1)
    upper[0] *= 2.0
    lower = -0.5/h*np.ones(Nr-1)
    lower[-1] *= 2.0
    diag = np.zeros(Nr)
    diag[0] = -1.0/h
    diag[-1] = 1.0/h
    D = scipy.sparse.diags([upper, diag, lower], [1, 0, -1])
    return D

def op_D2( h, Nr):
    ''' Central second-derivative operator. Nothing is done at the boundary. '''
    # Define left-biased derivative operator for u
    DL = scipy.sparse.lil_matrix(
        scipy.sparse.diags([1.0/h*np.ones(Nr), -1.0/h*np.ones(Nr-1)], [0, -1]))
    DL[0,:] = DL[1,:]
    # Define right-biased derivative operator for stress
    DR = scipy.sparse.lil_matrix(
        scipy.sparse.diags([-1.0/h*np.ones(Nr), 1.0/h*np.ones(Nr-1)], [0, 1]))
    DR[-1,:] = DR[-2,:]
    return DL @ DR

def op_E_drr(h, Nr, r_mesh):
  ''' Linear mapping from radial displacement to spherically symmetric deviatoric rr-strain'''
  # Diagonal matrix containing values of 1/r
  diag_inv_r = scipy.sparse.diags([1.0/r_mesh], [0])
  E_drr = (2.0/3.0) * (op_D(h, Nr) - diag_inv_r)
  return E_drr

def op_E_kk(h, Nr, r_mesh):
  ''' Linear mapping from radial displacement to spherically symmetric kk-strain'''
  # Diagonal matrix containing values of 1/r
  diag_inv_r = scipy.sparse.diags([1.0/r_mesh], [0])
  E_kk = op_D(h, Nr) + 2.0*diag_inv_r
  return E_kk

def op_A(h, Nr, r_mesh):
  ''' Elasticity differential operator valid in the interior nodes:
        d^2/dr^2 + 2/r * d/dr - 2/r^2
    '''
  diag_inv_r = scipy.sparse.diags([1.0/r_mesh], [0])
  A = (op_D2(h, Nr)
        + 2.0 * diag_inv_r @ op_D(h, Nr)
        - 2.0 * diag_inv_r * diag_inv_r)
  return A

def default_props(r_hydr_default=5, mu_default=1e6,
                  t_b_default=1e11, t_d_default=5e10,
                  K_crust_default=10e9, G_crust_default=10e9,
                  K_f_default=10e9,):
  ''' Package and return default material properties as a dictionary.
  Keyword arguments provided override the specified property in the returned
  dictionary.
  '''

  return dict(
    r_hydr_default = r_hydr_default,     # Effective hydraulic radius
    mu_default = mu_default,             # Constant viscosity assumption
    t_b_default = t_b_default,           # Set Maxwell times
    t_d_default = t_d_default,           # Set Maxwell times
    K_crust_default = K_crust_default,
    G_crust_default = G_crust_default,
    K_f_default = K_f_default,
  )

def default_numerics(Nr=50, R_outer_ratio=20):
  ''' Package and return default numerical settings as a dictionary.
  Keyword arguments provided override the specified property in the returned
  dictionary.
  '''

  return dict(
    Nr = Nr,                              # Number of cells in r-coordinate
    R_outer_ratio = R_outer_ratio,        # R_max / R_min
  )

class MagmaChamber(Node):
  def __init__(self,
               x:float=np.nan, y:float=0.0, z:float=np.nan,
               p_setting:object=None, T_setting:object=None,
               V_setting:object=None,
               c_v=1e3, K=10e9, vref=1/2.5e3, pref=25e6, g=10):
    ''' Initializes a magma chamber from coordinates, pressure, temperature,
    and volume.

    Args:
      x: horizontal coordinate
      y: (optional) in-plane direction (default 0.0)
      z: depth
      p_setting: pressure setting, with the following options:
        1) float: pressure in Pa
        2) None: lithostatic pressure
        3) object with field `mode` and field `value`
      T_setting: temperature setting, with the following options:
        1) float: temperature in K
        2) None: in equilibrium with geothermal gradient
        3) object with field `mode` and field `value`
      V: float volume
        1) float: volume in m^3
        2) object with field `mode` and field `value`
    '''

    # Magma parameters
    self.c_v = c_v # Specific heat capacity (J/kg K)
    self.K = K
    self.vref = vref
    self.rhoref = 1.0 / vref
    self.pref = pref
    self.g = g

    # Spatial settings
    self.x = x
    self.y = y
    self.z = z

    # Parse setting objects: pressure
    self.p_init = np.nan
    if p_setting is None:
      # Set initial pressure to lithostatic
      self.p_init = p_lithostatic(z)
    elif isinstance(p_setting, (int, float)):
      # Set initial pressure to a specified value
      self.p_init = p_setting
    else:
      try:
        if p_setting.mode.casefold() == "overpressure":
          self.p_init = p_setting.value
        else:
          raise ValueError(f"Unknown p_setting mode {p_setting.mode}")
      except AttributeError as e:
        print("setting object must contain field `mode` and `value`.")
        raise ValueError from e
    
    # Parse setting objects: temperature
    self.T_init = np.nan
    if T_setting is None:
      # Set initial temperature to geothermal gradient (this should be solid/cold)
      self.T_init = T_geothermal(z)
    elif isinstance(T_setting, (int, float),):
      # Set initial temperature to a specified value
      self.T_init = T_setting
    else:
      try:
        raise ValueError(f"Unknown T_setting mode {T_setting.mode}")
      except AttributeError as e:
        print("setting object must contain field `mode` and `value`.")
        raise ValueError from e
    # Parse setting objects: volume
    self.V = np.nan
    if V_setting is None:
      raise ValueError("Volume arg cannot be None (default is None).")
    elif isinstance(V_setting, (int, float),):
      self.V = V_setting
    else:
      try:
        raise ValueError(f"Unknown V_setting mode {V_setting.mode}")
      except AttributeError as e:
        print("setting object must contain field `mode` and `value`.")
        raise ValueError from e

    ''' Compute E, m '''
    # Compute mass from volume and specific volume v(p) via EOS
    self.m = self.V / MagmaChamber.v_p(self.p_init, self.K, self.vref, self.pref)
    # Compute energy via volume and volumetric energy (caloric equation)
    self.E = self.V * MagmaChamber.volenergy_pT(
        self.p_init, self.T_init, self.K, self.vref, self.pref, self.c_v)

  ''' Static equation of state (EOS) and caloric implementations

     (p - pref) / K = - (v - vref) / vref

  where K is the magma bulk modulus and (pref, vref) is a linearization point.
  Energy is defined as

     E = m * c_v * T + 0.5 * (p - pref)^2 / K,

  where c_v is a constant-volume heat capacity.

  '''

  @staticmethod
  def v_p(p, K, vref, pref):
    ''' Specific volume as function of pressure from EOS
    '''
    return vref * (1 - (p - pref) / K)

  @staticmethod
  def p_v(v, K, vref, pref):
    ''' Pressure as function of specific volume from EOS
    '''
    return pref - K * (v - vref) / vref

  @staticmethod
  def strain_volenergy_p(p, K, vref, pref):
    ''' Strain energy in magma as function of pressure from EOS
    '''
    return 0.5 * (p - pref)*(p - pref) / K

  @staticmethod
  def strain_volenergy_v(v, K, vref, pref):
    ''' Strain energy in magma as function of spec. vol. from EOS
    '''
    return 0.5 * (1 - v/vref)*(1 - v/vref) * K

  @staticmethod
  def volenergy_pT(p, T, K, vref, pref, c_v):
    ''' Volumetric energy from p, T '''
    e_mech = 0.5 * (p - pref)*(p - pref) / K
    e_int  = c_v * T / MagmaChamber.v_p(p, K, vref, pref)
    return e_mech + e_int

  ''' Dependent quantities as object properties '''

  @property
  def _U(self) -> np.array:
    ''' Vector of dependent variables U = [m, E, V].
    Private attribute (unused at the moment).
    '''
    return np.array([self.m, self.E, self.V])

  @property
  def v(self):
    ''' Specific volume. '''
    return self.V / self.m

  @property
  def rho(self):
    ''' Density. '''
    return self.m / self.V

  @property
  def p(self):
    ''' Returns pressure through EOS and parameters in self. '''
    return MagmaChamber.p_v(self.v, self.K, self.vref, self.pref)

  @property
  def strain_volenergy(self):
    ''' Strain energy per volume and parameters in self. '''
    return MagmaChamber.strain_volenergy_v(self.V/self.m, self.K, self.vref, self.pref)

  @property
  def internal_volenergy(self):
    ''' Internal energy per volume. '''
    return self.E/self.V - self.strain_volenergy

  @property
  def T(self):
    return (self.v * self.internal_volenergy) / self.c_v

  @property
  def e(self):
    ''' Specific energy (per mass) '''
    return self.E / self.m

  @property
  def h(self):
    ''' Specific enthalpy (per mass) '''
    return self.E / self.m + self.p * self.v

  @property
  def xyz(self):
    return np.array([self.x, self.y, self.z])

  def __repr__(self):
    ''' Pretty table print of this magma chamber '''
    output_dict = {
      "Chamber at      ": f"({self.x}, {self.y}, {self.z})",
      "Mass (kg)       ": f"{self.m:.5e}",
      "Energy (MJ)     ": f"{(self.E/1e6):.5e}",
      "Volume (m^3)    ": f"{self.V:.7f}",
      "Temperature (K) ": f"{self.T:.7f}",
      "Pressure (MPa)  ": f"{(self.p/1e6):.8f}",
      "Density (kg)    ": f"{self.rho:7f}",
      "Depth (km)      ": f"{self.z/1e3:2f}",
    }
    return "\n".join([k + v for k, v in output_dict.items()])


class GlobalSystemThreshold():
  ''' Global coupled system of chambers with methods for manipulating the network.

  Heterogeneous properties of the chamber network are accepted.
  '''

  # Define schema for data shape
  @property
  def data_slice(self):
    ''' Schema for organizing data within vector for a block.
    Defines a dict that maps keys to non-overlapping, contiguous slices. First
    slice must start at index 0. '''
    Nr = self.Nr
    schema = dict(
        gamma_drr=slice(0, Nr),
        gamma_kk=slice(Nr, 2*Nr),
        mass=slice(2*Nr, 2*Nr+1),
        energy=slice(2*Nr+1, 2*Nr+2),
        massCO2=slice(2*Nr+2, 2*Nr+3),
        massH2O=slice(2*Nr+3, 2*Nr+4),
    )
    # Return schema
    return schema

  def data_slice_global(self, i, qty_name):
    ''' Map (chamber_idx, qty_name) to data slice in global vector '''
    try:
      local_slice=self.data_slice[qty_name]
    except KeyError as e:
      raise ValueError(f"Quantity name '{qty_name}' was not found in schema;"
                       + f" here is a list of valid quantity names: "
                       + str(self.data_slice.keys())) from e
    return slice(i*self.block_size+local_slice.start,
                 i*self.block_size+local_slice.stop)

  @property
  def block_size(self):
    ''' Size of a single block, corresponding to one chamber. '''
    return max([s.stop for s in self.data_slice.values()])

  def check_schema_validity(self) -> None:
    ''' Check validity of schema (basic checks only). Checks that the
    implementation of GlobalSystem.data_slice is a valid mapping to slices of a
    vector of size `block_size`. '''
    schema = self.data_slice
    _validation = dict()
    for k, v in schema.items():
      _validation[v.start] = _validation.get(v.start, 0) + 1
      _validation[v.stop]  = _validation.get(v.stop, 0) + 1
    _range_endpoints = list(schema.keys())
    _occur_count = list(schema.values())
    _occur_count_sorted = [count for _, count
                           in sorted(zip(_range_endpoints, _occur_count))]
    if (sorted(_range_endpoints)[0] == 0 # Range starts 0
        and _occur_count_sorted[-1] == 1 # Last index is unique
        and _occur_count_sorted[0] == 1  # First index is unique
        and all([val == 2 for val in _occur_count_sorted[1:-1]])): # Data is contiguous
      return
    else:
      return _range_endpoints, _occur_count
      raise ValueError("Data schema seems invalid. The location of data in the "
                      + "state vector for a single chamber may be invalid.")

  def __init__(self,
              nodes:list,
              r_hydr_default=5,
              mu_default=1e6,
              t_b_default=1e11,
              t_d_default=5e10,
              K_crust_default=10e9,
              G_crust_default=10e9,
              K_f_default=10e9,
              Nr=50,
              R_outer_ratio=20,
              p_crit=1e3, p_threshold_scale=1e2,
              dpdx_crit=1e3, dpdx_threshold_scale=1e2,
              max_edge_dist=np.inf, remote_sigma_xx=0.0):
    self.nodes:list = nodes
    self.K_f_default = K_f_default
    self.t_b_default = t_b_default
    self.t_d_default = t_d_default
    self.K_crust_default = K_crust_default
    self.G_crust_default = G_crust_default
    self.r_hydr_default = r_hydr_default
    self.mu_default = mu_default

    self.Nr = Nr
    self.p_crit = p_crit
    self.p_threshold_scale = p_threshold_scale
    self.dpdx_crit = dpdx_crit
    self.dpdx_threshold_scale = dpdx_threshold_scale
    self.R_outer_ratio = R_outer_ratio

    self.max_edge_dist = max_edge_dist
    self.remote_sigma_xx = remote_sigma_xx

    self.M_crust_default = K_crust_default + 4.0*G_crust_default/3.0
    self.num_blocks = len(nodes)
    self.num_dof = self.num_blocks * self.block_size

    # Check implemented data schema for organizing field variables
    self.check_schema_validity()
    # Initialize nodes with linearization point, operators
    [self._init_node(node) for node in self.nodes]

    self.mat_props = dict(
      t_b = t_b_default,
      t_d = t_d_default,
      K_crust = K_crust_default,
      G_crust = G_crust_default,
      K_f = K_f_default,
      mu0 = mu_default,
      r_hydr = r_hydr_default,
    )

    # Set initial condition with initial (absolute) mass
    self.q0 = np.zeros((self.num_dof, 1))
    for i, node in enumerate(self.nodes):
      self.q0[self.data_slice_global(i, "mass")] = node.m0

    # Dictionary mapping ordered chamber index tuple (i,j), i < j to matrix
    # representing flow sparsity pattern
    self.M_stencils = dict()
    # Allocate mapping for the row-vector representation of M_stencils
    self.M_vecs = dict()
    num_blocks, block_size = self.num_blocks, self.block_size
    for i in range(num_blocks):
      node_i = self.nodes[i]
      for j in range(i+1, num_blocks):
        node_j = self.nodes[j]
        # Set up dimensionless flow matrix for the first time
        M_loc = scipy.sparse.lil_matrix((num_blocks * block_size, num_blocks * block_size))

        _M_vec = np.zeros((num_blocks * block_size,))
        _M_vec[i*block_size:(i+1)*block_size] = (-3.0 * node_i.H[0,:] / node_i.R0).toarray().ravel()
        _M_vec[j*block_size:(j+1)*block_size] = (3.0 * node_j.H[0,:] / node_j.R0).toarray().ravel()
        _M_vec[i*block_size + 2*Nr] += 1.0 / node_i.m0
        _M_vec[j*block_size + 2*Nr] += -1.0 / node_j.m0

        M_loc[i*block_size + 2*Nr, :] = _M_vec
        M_loc[j*block_size + 2*Nr, :] = -_M_vec

        # # Compute dependence of mass rate on ith viscoelastic field as u(r=R) / R0 through H_i
        # M_loc[i*block_size + 2*Nr, i*block_size:(i+1)*block_size] -= 3.0 * node_i.H[0,:] / node_i.R0 # R0i
        # M_loc[j*block_size + 2*Nr, i*block_size:(i+1)*block_size] += 3.0 * node_i.H[0,:] / node_i.R0 # R0i
        # # Compute dependence of mass rate on jth viscoelastic field as u(r=R) / R0 through H_j
        # M_loc[i*block_size + 2*Nr, j*block_size:(j+1)*block_size] += 3.0 * node_j.H[0,:] / node_j.R0 # H_j, R0j
        # M_loc[j*block_size + 2*Nr, j*block_size:(j+1)*block_size] -= 3.0 * node_j.H[0,:] / node_j.R0 # H_j, R0j
        # # Compute dependence of mass rate on ith chamber mass
        # M_loc[i*block_size + 2*Nr, i*block_size + 2*Nr] += 1.0 / node_i.m0 # m0i -- note this diagonal term should be +
        # M_loc[j*block_size + 2*Nr, i*block_size + 2*Nr] -= 1.0 / node_i.m0 # m0i
        # # Compute dependence of mass rate on jth chamber mass
        # M_loc[i*block_size + 2*Nr, j*block_size + 2*Nr] -= 1.0 / node_j.m0 # m0j
        # M_loc[j*block_size + 2*Nr, j*block_size + 2*Nr] += 1.0 / node_j.m0 # m0j

        # Register flow matrix to the pair (i,j), i < j
        self.M_stencils[(i,j,)] = M_loc.tocsr()
        self.M_vecs[(i,j,)] = _M_vec

        # Distance validity check
        dist = float(np.sqrt((node_i.x - node_j.x) ** 2
                      + (node_i.y - node_j.y) ** 2
                      + (node_i.z - node_j.z) ** 2))
        if np.isclose(dist, 0.0):
          raise ValueError(f"Distance between chamber {i} and {j} is " \
                           f"close to zero ({dist:.2e}).")
    
    # Array of indices corresponding to dm_i/dt
    self.mass_indices = np.concatenate([
      np.arange(0, self.num_dof)[self.data_slice_global(i, "mass")]
      for i in range(self.num_blocks)])
    
    # Compute pairwise distances
    coords = np.array([[node.x, node.y, node.z] for node in self.nodes])
    self.dists = scipy.spatial.distance.cdist(coords, coords)
    # Check pairwise distances
    if np.any((self.dists == 0) & (1 - np.eye(self.num_blocks)).astype(bool)):
      illegal_nodes = scipy.sparse.coo((self.dists == 0) & (1 - np.eye(self.num_blocks)))
      raise ValueError(f"Distance between nodes was zero for (i,j) = : ({illegal_nodes.row}, {illegal_nodes.col})")

    # Cache global H0, k0 by concatenating H0, k0 for each node
    self.H0_global = scipy.sparse.lil_matrix((num_blocks, self.num_dof))
    for i in range(num_blocks):
      self.H0_global[i, i*block_size:(i+1)*block_size] = self.nodes[i].H0
    self.H0_global = scipy.sparse.csr_matrix(self.H0_global)
    self.k0_global = np.array([self.nodes[i].k0 for i in range(num_blocks)])

    ''' Cache mechanical equilibrium operators for all nodes
    Mechanical equilibrium is achieved by venting mass down to elastic
    equilibrium, without change to the viscous displacement for the given node.
    The equilibrium displacement field is given by
      u_eq = H_mod @ q_loc + k_mod.squeeze()
    '''
    for i, node in enumerate(self.nodes):
      # Outer product matrix representing effect of mass on bdry displacement
      outer = scipy.sparse.lil_matrix((Nr, Nr,))
      outer[:,0] = 3 * node.m0 / node.R0 * node.H[:,self.data_slice["mass"]]
      # Compute operator q -> m_equilibrium, with the form m_eq = H_mod @ q + k_mod
      H_mod = scipy.sparse.linalg.spsolve_triangular(
        scipy.sparse.eye(Nr, Nr) - outer, node.H.todense(), lower=True)
      # Compute modified k
      k_mod =  scipy.sparse.linalg.spsolve_triangular(
        scipy.sparse.eye(Nr, Nr) - outer, node.k, lower=True)
      # Move dependence on mass in input q to dependence on m0
      k_mod += H_mod[:, self.data_slice["mass"]] * node.m0
      H_mod[:, self.data_slice["mass"]] = 0
      # Attach H_mod, k_mod to node
      node.H_mod = scipy.sparse.csr_matrix(H_mod)
      node.k_mod = k_mod

  def _init_node(self, node, K_crust=None, G_crust=None, K_f=None,
                 t_d=None, t_b=None, r_hydr=None, mu0=None) -> None:
    ''' Initializes node by allocating the linear elasticity affine mapping
    and recording the current m, R as the linearization point.
    
    Construcs a matrix and vector representing the mapping from time-dependent
    variables to radial displacement u, i.e., for a time-dependent vector q,
      u = Hq + k
    for the node passed as an input argument.
    Inverts sparsely, but returns a possibly dense matrix H.
    '''

    ''' Assemble local matrix for a single chamber
      This is L + G @ H in
        dq/dt + (L + G @ H) @ q == - G @ k,
      accounting for the effect of static displacement.

      Returns tuple (L, f, H, k) with respective sizes
        (block_size, block_size,)
      and
        (block_size, 1,)
      and
        (block_size, block_size,)
      and
        (block_size, 1,)
      respectively. Here H, k are passed through to reduce redundant computation.
    '''

    # Add variables to scope
    block_size = self.block_size
    R_outer_ratio = self.R_outer_ratio

    # Read properties from args, else use global default
    node.Nr      = self.Nr
    node.K_crust = self.K_crust_default if K_crust is None else K_crust
    node.G_crust = self.G_crust_default if G_crust is None else G_crust
    node.K_f     = self.K_f_default     if K_f is None else K_f
    node.t_d     = self.t_d_default     if t_d is None else t_d
    node.t_b     = self.t_b_default     if t_b is None else t_b 
    # Read edge properties TODO: how to determine edge property from node?
    node.r_hydr  = self.r_hydr_default  if r_hydr is None else r_hydr 
    node.mu0     = self.mu_default      if mu0 is None else mu0 
    # Compute dependent quantities
    node.M_crust = node.K_crust + (4.0 / 3.0) * node.G_crust
    
    # Unpack self properties
    Nr, K_crust, G_crust, M_crust, K_f, t_d, t_b = \
      node.Nr, node.K_crust, node.G_crust, node.M_crust, node.K_f, node.t_d, node.t_b

    m0 = node.m
    R0 = (node.V / (4*np.pi/3))**(1.0/3.0)
    # Set up mesh
    # Set maximum r-coordinate to approximate BC at infinity
    R_inf = R_outer_ratio * R0
    # Compute dx
    dx = (R_inf - R0) / (Nr-1)
    # Set mesh points in r-coordinate
    r_mesh = np.linspace(R0, R_inf, Nr)
    # Define diagonal matrix of values 1/r
    r_mesh_inv = scipy.sparse.diags([1.0 / r_mesh], [0])

    ''' Local differentiation construction '''
    # Construct second-order derivative operator (units 1/length^2)
    A = op_A(dx, Nr, r_mesh)
    # Construct first-order central derivative operator (units 1/length)
    D = op_D(dx, Nr)
    # Construct diagonal matrix 1/r
    diag_inv_r = scipy.sparse.diags([1.0/r_mesh], [0])

    ''' Compute mapping L_u from viscous strains to displacements '''
    # Assemble rectangular system for static equilibrium
    L_u = scipy.sparse.lil_matrix((Nr, Nr + block_size))
    # Construct elastic portion of static equilibrium equation
    L_u[:, 0:Nr] = A
    # Construct mapping of γ_drr to term in static equilibrium equation
    L_u[:, Nr:2*Nr] = 2 * (G_crust/M_crust) * D + 6 * (G_crust/M_crust) * diag_inv_r
    # Construct mapping of γ_kk to term in static equilibrium equation
    L_u[:, 2*Nr:3*Nr] = (K_crust/M_crust) * D

    ''' Set traction boundary condition at r = R0
      \sigma_{rr} = -(p - p_0)
    where \sigma_{rr} is the normal stress (in excess of "crustal prestress")
    and p_0 is the pressure linearization point
    '''
    # Replace first row with boundary traction (normalized by M_crust) Dirichlet
    # lift operator at r = R0 (linearized boundary treatment)
    L_u[0, :] = 0.0
    L_u[0, 0] += -1.0 / dx
    L_u[0, 1] += 1.0 / dx
    L_u[0, 0] += (2*K_crust - 4*G_crust/3) / M_crust/ R0
    # Add r = R boundary dependence on γ_drr
    L_u[0, Nr] = -2 * G_crust / M_crust
    # Add r = R boundary dependence on γ_kk
    L_u[0, 2*Nr] = -K_crust / M_crust
    # Add r = R boundary dependence on boundary pressure, linearly dependent on u, m
    L_u[0, 0] += - 3 * K_f / M_crust / R0
    L_u[0, 3*Nr] += K_f / m0 / M_crust
    # Add RHS loading due to traction boundary condition
    f_u = np.zeros((Nr, 1))
    f_u[0] += K_f / M_crust
    # Save RHS as sparse vector
    f_u = scipy.sparse.csc_matrix(f_u)

    ''' Set boundary condition at r = r_inf '''
    # Replace last row with boundary displacement Dirichlet lift operator
    L_u[Nr-1, :] = 0
    L_u[Nr-1, Nr-1] = 1
    # Finalize matrix format
    L_u = L_u.tocsc()

    ''' Define mapping from time-dependent variables to u '''
    # Compute affine map q -> Hq + k from time-dependent variables (viscous strains, mass, energy...) to u
    node.H = scipy.sparse.linalg.spsolve(L_u[0:Nr, 0:Nr], -L_u[0:Nr, Nr:])
    node.k = scipy.sparse.linalg.spsolve(L_u[0:Nr, 0:Nr], f_u)[:,np.newaxis]
    # Compute shorthand for first row of H as dense np.array
    node.H0 = node.H[0,:].toarray().ravel()
    # Compute shorthand for first entry of k as float
    node.k0 = node.k.ravel()[0]
    ''' Save other states to node '''
    node.dx = dx
    node.r_mesh = r_mesh
    node.inv_r = r_mesh_inv
    node.m0 = node.m
    node.R0 = R0
    node.p_init = node.p

    # Assemble dependence of viscous strain evolution on displacement u (through elastic strain)
    G = scipy.sparse.lil_matrix((block_size, Nr))
    G[0:Nr, 0:Nr] = -1.0 / node.t_d * op_E_drr(dx, Nr, r_mesh)
    G[Nr:2*Nr, 0:Nr] = -1.0 / node.t_b * op_E_kk(dx, Nr, r_mesh)
    # Compute matrix L
    node.L = scipy.sparse.lil_matrix((block_size, block_size))
    node.L[np.arange(0,Nr), np.arange(0,Nr)] = (1 / node.t_d)
    node.L[np.arange(Nr,2*Nr), np.arange(Nr,2*Nr)] = (1 / node.t_b)
    # Add dependence on u through Schur complement term
    node.L += G @ node.H

    ''' Assemble local RHS vector for a single chamber
      This is f - G @ K,
    where f contains any external source terms for the time-dependent variables.
    '''
    # Assemble right hand side for local problem
    node.f = scipy.sparse.lil_matrix((block_size, 1))
    # Put dependence on spherical boundary condition
    node.f -= G @ node.k

    # Save reference to system matrix
    # node.L_u = L_u

  def get_connectivity(self, q, threshold="gradient"):
    ''' 
    Positive connectivity matrix with units of admittance ( (m/s) / Pa )
    Nonzero values appear on either the upper diagonal (i -> j) or lower
    diagonal (j -> i).

    Legacy wrapper for self.mass_rates(q, return_format="Y").
    '''

    return self.mass_rates(q, return_format="Y", threshold=threshold)
  
    # Get system size information
    Nr, num_blocks, block_size = self.Nr, self.num_blocks, self.block_size
    # Compute pressures
    p_node = self.pressure(q)

    Y = np.zeros((num_blocks, num_blocks))

    for i in range(self.num_blocks):
      node_i = self.nodes[i]
      for j in range(self.num_blocks):
        if i == j:
          continue
        node_j = self.nodes[j]
        dist = self.dists[i,j]
        # Check distance threshold
        if dist > self.max_edge_dist:
          continue
        # Compute average pressure gradient
        dpdx = (p_node[i] - p_node[j]) / dist
        # Resolve remote tensile stress in x-direction
        opening_stress = self.remote_sigma_xx * np.abs(node_i.z - node_j.z)/ dist
        # Effective pressure gradient
        dpdx_eff = dpdx
        # Effective critical pressure gradient for opening
        dpdx_crit_eff = self.dpdx_crit - opening_stress

        # Factor between (0, 1) that modulates flow between the two chambers
        if dpdx_crit_eff != 0:
          threshold_factor = smoother(np.abs(dpdx_eff) - dpdx_crit_eff,
                                      self.dpdx_threshold_scale) * float(dpdx_eff > 0)
          if threshold_factor > 1 or threshold_factor < 0:
            raise ValueError
        else:
          threshold_factor = 1.0

        # Compute flow admittance ( (m/s) / Pa )
        Y[i,j] = threshold_factor * self.r_hydr_defualt * self.r_hydr_defualt / 16.0 / self.mu0 / dist

    return Y

  def assemble_global_Lf(self, q):
    ''' Assemble global matrix, coupling all chambers. The ODE system is
        (dq/dt) + L @ q + M(q) @ q = f,
        where L captures the viscoelastic effect and M captures mass transfer.
    '''
    # Abbreviate system size information
    num_blocks, block_size = self.num_blocks, self.block_size
    # Allocate global L, f matrices
    L = scipy.sparse.lil_matrix((num_blocks * block_size, num_blocks * block_size))
    f = scipy.sparse.lil_matrix((num_blocks * block_size, 1))
    for i, node in enumerate(self.nodes):
      L[i*block_size:(i+1)*block_size, i*block_size:(i+1)*block_size] = node.L
      f[i*block_size:(i+1)*block_size,0] = node.f
    return L, f

  def mass_rates(self, q, return_format=None, threshold="gradient"):
    ''' Compute the strictly lower triangular outgoing mass transfer matrix M,
    where M[i,j] represents the mass rate from j to i through edge (i,j),
    where the mass rate is nonnegative.

    Parameter return_format sets the output format:
      * None: default return (mass_rates matrix)
      * "M": global mass transfer matrix, such that M@q is the mass_rates matrix
      * "tups": mass transfer rates in tuple format (i, j, mass rate)
    
    Parameter threshold sets the physical threshold:
      * "gradient" (default): Threshold is set based on gradient |p_i - p_j| / dz
      * "absdiff": Threshold is set based on absolute pressure difference |p_i - p_j|
      * "fracture": Threshold is set based on local overpressure p_i - p_lithostatic

    Properties of output matrix if return_format is None:
      * Entries of M are >= 0
      * Column sums of M give the rate of mass leaving node j
      * Row sums of M give the rate of mass entering node i
      * M - M.T is the full antisymmetric mass transfer matrix, such that
          (M - M.T).sum(axis=1) is the vector (dm/dt) for each row i.
          
    Legacy docs:

      Pressure differences between chambers are
      p_i - p_j = -(K_fi - K_fj) - (3 K_fi u_ri / R_i - 3 K_fj u_rj / R_j) + K_f * (m_i/m_0i - m_j/m_0j)
      and mass rate ~ rho_upstream * hyd_cond * (p_i - p_j).

      Here we estimate
      p_i - p_j = - 3 * K_f (u_ri / R_i - u_rj / R_j) + K_f * (m_i/m_0i - m_j/m_0j)
      and thus
      \dot{m}_{ij} = Adj_{ij} * hydr_cond * rhoref * K_f * (
        - 3 * (u_ri / R_i - u_rj / R_j) + (m_i/m_0i - m_j/m_0j)
      )
      where Adj is the adjacency matrix. Here the hydraulic conductivity has units of
      mass flux per pressure; that is, (m^3/s)/Pa in SI units.

      '''
    # Get system size information
    num_blocks, block_size = self.num_blocks, self.block_size
    
    # Allocate variable for the specified return format
    if return_format == "tups":
      # Allocate output tuples
      tups = []
    elif return_format == "M":
      # Allocate global M matrix
      M = scipy.sparse.csr_matrix((num_blocks * block_size, num_blocks * block_size))
    elif return_format == "Y":
      # Allocate dense conductivity ((m/s)/Pa) matrix, upper triangular
      Y_matrix = np.zeros((num_blocks, num_blocks))
    elif return_format is None:
      # Allocate mass rates from j to i
      mass_rates = scipy.sparse.lil_matrix((self.num_blocks, self.num_blocks))
    else:
      raise ValueError(f"Invalid return_format passed to mass_rates. ('tups'|'M'|'Y'|None)")
    
    # Compute pressures
    p_node = self.pressure(q)
    
    for i in range(num_blocks):
      for j in range(i+1, num_blocks):
        if self.dists[i,j] > self.max_edge_dist:
          continue
        # Compute average pressure gradient between self.nodes[i] and self.nodes[j]
        dpdx = (p_node[i] - p_node[j]) / self.dists[i,j]
        # Hydrostatic pressure difference divided by distance
        dpdx_hydro = 0.5 * (self.nodes[i].rhoref + self.nodes[j].rhoref) * g * (
          (self.nodes[i].z - self.nodes[j].z) / self.dists[i,j])
        # Account for hydrostatic pressure gradient
        dpdx += dpdx_hydro

        # Resolve remote tensile stress in x-direction
        opening_stress = self.remote_sigma_xx \
          * np.abs(self.nodes[i].z - self.nodes[j].z) / self.dists[i,j]
        # Effective critical pressure gradient for opening
        dpdx_crit_eff = self.dpdx_crit - opening_stress

        ''' Compute threshold factor between [0, 1] '''
        if threshold == "gradient":
          if dpdx_crit_eff <= 0:
            threshold_factor = 1.0
          else:
            # Factor between (0, 1) that modulates flow between the two chambers
            threshold_factor = float(smoother(np.abs(dpdx) - dpdx_crit_eff,
                                        self.dpdx_threshold_scale))
            if threshold_factor > 1 or threshold_factor < 0:
              raise ValueError
        else:
          raise ValueError(
            f"Unknown threshold '{threshold}'. Use ('gradient'|'absdiff'|'fracture')")        

        if threshold_factor > 1e-15:
          # Set upstream properties for flow
          if p_node[i] > p_node[j]:
            rho = self.nodes[i].rhoref
            K_f = self.nodes[i].K_f
            r_hydr = self.nodes[i].r_hydr
            mu = self.nodes[i].mu0
          else:
            rho = self.nodes[j].rhoref
            K_f = self.nodes[j].K_f
            r_hydr = self.nodes[j].r_hydr
            mu = self.nodes[j].mu0
          # Compute flow admittance ( (m/s) / Pa )
          #   sign is determined automatically by multiplication with state vector q
          Y = threshold_factor * r_hydr * r_hydr / 16.0 / mu / self.dists[i,j]
          # Multiply upstream density and bulk modulus to units (mass areal flux)
          mflux = Y * rho * K_f
          if return_format == "tups":
            tups.append((i, j, mflux,))
          elif return_format == "M":
            # Multiply mass rate coefficient (kg / s) by dimensionless flow matrix M
            M += mflux * self.M_stencils[(i,j,)]
          elif return_format == "Y":
            if Y > 0:
              Y_matrix[i,j] = Y
            elif Y < 0:
              Y_matrix[j,i] = -Y
          else:
            # Multiply mass rate coefficient (kg / s) by dimensionless flow matrix M
            mass_rates[:,i] += mflux * (
              self.M_stencils[(i,j,)] @ q)[self.mass_indices].squeeze()
      
      # if not (return_format == "tups" or return_format == "M"):
        # mass_rates[:,i] = -(M @ q)[self.mass_indices].squeeze()

    if return_format == "tups":
      return tups
    elif return_format == "M":
      return M
    elif return_format == "Y":
      return Y_matrix
    else:
      # Remove diagonal for mass transfer
      mass_rates[np.arange(0, num_blocks), np.arange(0, num_blocks)] = 0
      return mass_rates

  def pressure(self, q):
    ''' Compute vector of pressures, indexed by chamber number. '''
    p = np.zeros((self.num_blocks, 1))
    # Extract masses, vectorized
    m = q[self.mass_indices]
    # Compute chamber wall displacements, vectorized
    wall_displacement = (self.H0_global @ q).ravel() + self.k0_global
    for i, node in enumerate(self.nodes):
      # Compute boundary displacement
      # u_R0 = np.dot(node.H0, q[i*self.block_size:(i+1)*self.block_size]) + node.k0
      # u_R0 = (node.H @ q[i*self.block_size:(i+1)*self.block_size] + node.k)[0]
      dp_u = -3 * node.K_f * wall_displacement[i] / node.R0
      # Pressure increase contribution due to added mass (m0 may be node-dependent)
      dp_m = node.K_f * ((m[i] - node.m0) / node.m0)
      p[i] = node.p_init + dp_u + dp_m
    return p

  def u(self, q):
    ''' Compute vector of displacements, indexed by chamber number.
    Use this only for matrix valued q.
    '''
    raise DeprecationWarning("GlobalSystemThreshold.u is deprecated. Use GlobalSystemThreshold.compute_m_p_u instead.")
    u = np.zeros((self.num_blocks, self.Nr))
    for i, node in enumerate(self.nodes):
      u[i,:] = (node.H @ q[i*self.block_size:(i+1)*self.block_size] + node.k).squeeze()
    return u

  def sigma_rr(self, q):
    # Extract q blockwise, for each chamber
    sigma_rr = np.zeros((self.num_blocks, self.Nr))
    for i, node in enumerate(self.nodes):
      q_loc = q[i*self.block_size:(i+1)*self.block_size].squeeze()
      # Compute boundary displacement
      u_loc = (node.H @ q_loc + node.k.squeeze())
      # Radial component of strain
      radial = (op_D(node.dx, node.Nr) @ u_loc)
      # Angular components (phi + theta) of stress div. by M_crust
      angular = u_loc / node.r_mesh
      # Elastic strain
      eps_drr = (2.0/3.0) * (radial - angular)
      eps_kk = radial + 2.0 * angular
      # Viscous strain γ_drr
      gamma_drr = q_loc[0:self.Nr]
      # Viscous strain γ_drr
      gamma_kk = q_loc[self.Nr:2*self.Nr]
      # Compute stress from elastic strain
      sigma_drr = 2 * node.G_crust * (eps_drr - gamma_drr)
      sigma_kk = 3 * node.K_crust * (eps_kk - gamma_kk)
      sigma_rr[i,:] = sigma_drr + (1.0/3.0) * sigma_kk
    return sigma_rr

  @staticmethod
  def compute_eigen_system(L) -> dict:
    ''' Post-process system '''

    # Compute eigensystem of L
    eig_result = np.linalg.eig((L).todense())
    # Filter out imaginary noise
    try:
      L_eigval = eig_result.eigenvalues # np.real_if_close(eig_result.eigenvalues)
      L_eigvec = eig_result.eigenvectors # np.real_if_close(eig_result.eigenvectors)
    except AttributeError: # Backward compatible eig syntax
      L_eigval = eig_result[0]
      L_eigvec = eig_result[1]

    # Compute 1/eig where nonzero
    Linv_eigval = np.full(L_eigval.shape, np.inf, dtype=np.complex128)
    np.divide(1.0, L_eigval, out=Linv_eigval, where=L_eigval!=0)

    # Sort finite 1/eig
    sort_index = np.argsort(L_eigval.real)[::-1]
    Linv_eigval_sorted = Linv_eigval[sort_index]
    Linv_eigval_finite = Linv_eigval_sorted[np.where(Linv_eigval_sorted != 0)]
    eigs = dict(
        eigval=L_eigval[sort_index],
        eigvec=L_eigvec[sort_index],
        Linv_eigval=Linv_eigval_sorted,
        Linv_eigval_finite =Linv_eigval_finite,
    )

    return eigs

  @staticmethod
  def matshow(L):
    ''' Wrapper for matshow '''
    return plt.matshow(np.log10(np.abs(L).todense()), cmap=plt.cm.Blues)

  @staticmethod
  def eigshow(eig, t_d, t_b):
    ''' Eigenvalue plot on complex plane '''
    plt.subplot(1,2,1)
    plt.plot(1/t_d, 0, '^r')
    plt.plot(1/t_b, 0, '*r')
    plt.scatter(eig["eigval"].real, eig["eigval"].imag, c='k')
    plt.xlabel("Re$(\lambda)$, 1/s")
    plt.ylabel("Im$(\lambda)$, 1/s")
    plt.gca().set_xscale("log")
    plt.grid("on")
    plt.title("Eigenvalues of $L$ in system $\dot{\mathbf{q}} + L\mathbf{q} = \mathbf{f}$")

    plt.subplot(1,2,2)
    plt.scatter(-eig["eigval"].real, -eig["eigval"].imag, c='k')
    plt.xlabel("Re$(-\lambda)$, 1/s")
    plt.ylabel("Im$(-\lambda)$, 1/s")
    plt.gca().set_xscale("log")
    plt.grid("on")
    plt.title("Negative eigenvalues if any")
    plt.tight_layout()

  def show_network(self, q, node_scale=1000,
                   add_ax_labels=True, ax=None, *args, **kwargs):
    ''' Plots nodes and edges that are "on" given the state vector q.
    Requires python module networkx '''
    try:
      import networkx as nx
    except ModuleNotFoundError as e:
      raise ModuleNotFoundError("This method needs package networkx. Aborting and dumping the error message.") from e
   
    if ax is None:
      ax = plt.gca()

    # Set color palette
    cmap = matplotlib.cm.hsv
    colors = cmap(np.linspace(0,1,self.num_blocks,endpoint=False))
    # Construct a directed graph using connectivity matrix computed from state vector q
    Y = self.get_connectivity(q)
    G = nx.DiGraph(Y)
    # Node index-position mapping (2D projection)
    length_scale = 1e3
    pos = {i:(n.x / length_scale, n.z / length_scale,)
           for (i, n) in enumerate(self.nodes)}
    V_nodes = np.array([node.V for node in self.nodes])
    node_size = node_scale*V_nodes/V_nodes.max()
    nx.draw_networkx(G, pos, node_size=node_size, node_color=colors,
                            edge_cmap=(0, 1), ax=ax, *args, **kwargs)
    
    if add_ax_labels:
      ax.tick_params(labelleft=True, labelbottom=True)
      ax.set_xlabel("$x$ (km)")
      ax.set_ylabel("$z$ (km)")

  def post_process(self, t, q):
    ''' Post-process array t and nd-array q into dependent quantities.
    Input:
      t: array of time points with size n_t
      q: nd-array of state vectors at each time; has shape (n_t, n_states)
    Output:
      masses: array of masses with shape (n_t, n_chambers)
      pressures: array of masses with shape (n_t, n_chambers)
      sigma_rr: array of radial stresses with shape (n_t, n_chambers, N_r)
      displacements: array of radial displacements with shape (n_t, n_chambers, N_r)
    '''
    masses = np.zeros((t.size, self.num_blocks, ))
    pressures = np.zeros((t.size, self.num_blocks, ))
    sigma_rr = np.zeros((t.size, self.num_blocks, self.Nr,))
    displacements = np.zeros((t.size, self.num_blocks, self.Nr,))

    for i in range(q.shape[0]):
      # State vector q at time t
      q_t = q[i,...]
      masses[i,...] = np.array([q_t[self.data_slice_global(i, "mass")]
                          for i in range(self.num_blocks)]).squeeze()
      pressures[i,...] = np.array(self.pressure(q_t)).squeeze()
      sigma_rr[i,...] = np.array(self.sigma_rr(q_t))
      displacements[i,...] = np.array(self.u(q_t))
    return masses, pressures, sigma_rr, displacements

  def compute_m_p_u(self, q):
    ''' Faster post-process returning masses, pressures, displacements of
    chamber wall. Assumes q is an array of shape (num_time_steps, num_states).

    Return m, p, and u, each with shape (num_time_steps, num_chambers).
    '''

    # Allocate
    u = np.zeros((q.shape[0], self.num_blocks,))
    p = np.zeros_like(u)
    # Extract masses, vectorized
    m = q[:,self.mass_indices]
    # Compute wall displacements time-by-time
    for i in range(q.shape[0]):
      u[i,:] = (self.H0_global @ q[i,:]).ravel() + self.k0_global
    # Compute pressure node-by-node
    for j, node in enumerate(self.nodes):
      dp_u = -3 * node.K_f * u[:,j] / node.R0
      dp_m = node.K_f * ((m[:,j] - node.m0) / node.m0)
      p[:,j] = node.p_init + dp_u + dp_m

    return m, p, u

  def create_single_mass_injection_source(self, mdot_inj):
    def f(t, q):
      f_inj = np.zeros((self.num_dof))
      f_inj[self.data_slice_global(0, "mass")] = mdot_inj
      return f_inj
    return f
  
  def create_single_mass_injection_custom(self, fn):
    ''' Wrapper for custom function of mass rate mdot(t, q). '''
    def f(t, q):
      f_inj = np.zeros((self.num_dof))
      f_inj[self.data_slice_global(0, "mass")] = fn(t, q)
      return f_inj
    return f
  
  def create_mass_injection_layer(self, fn_z=None, z_max=-30e3, mdot_inj=1.0):
    ''' Adds mass injection to all chambers according to given function
      fn_z(t, q, z),
    where z is the depth of the chamber. If fn_z not provided, uses default
    uniform distribution below threshold depth z_max at rate mdot_inj. '''

    z_nodes = [node.z for node in self.nodes]
    if fn_z is None:

      i_z_nodes = [(i, z) for i, z in enumerate(z_nodes) if z <= z_max]

      def f_uniform_depth(t, q):
        f_inj = np.zeros((self.num_dof))
        for i, z in enumerate(i_z_nodes):
          f_inj[self.data_slice_global(i, "mass")] = mdot_inj
        return f_inj
      return f_uniform_depth

    def f(t, q):
      f_inj = np.zeros((self.num_dof))
      for i, z in enumerate(z_nodes):
        f_inj[self.data_slice_global(i, "mass")] = fn_z(t, q, z)
      return f_inj
    return f

  def create_row_mass_injection_source(self, mdot_inj, nodes_per_layer:list):
    ''' Create row of injection sources splitting mdot_inj between them. '''
    def f(t, q):
      f_inj = np.zeros((self.num_dof))
      # Count nodes in the bottom layer for an N-way split
      N_split = nodes_per_layer[0]
      for i in range(N_split):
        f_inj[self.data_slice_global(i, "mass")] = mdot_inj / N_split
      return f_inj
    return f

  def create_single_pressure_injection_source(self, feed_overpressure):
    ''' Returns callable function representing mass injection at node 0
    due to a constant pressure source with overpressure relative to the
    initial state of node 0.
    '''
    def f(t, q):
      f_inj = np.zeros((self.num_dof))
      mu = self.nodes[0].mu0
      p_node = self.pressure(q)
      # Compute actual overpressure for node 0
      deltap = self.nodes[0].p_init + feed_overpressure - p_node[0]
      # Compute injection rate by flow rule
      injection_rate = self.nodes[0].rhoref * (deltap / (16.0 * mu)) / 10 # * r_hydr * r_hydr * r_hydr # TODO: extract parameter
      f_inj[self.data_slice_global(0, "mass")] = injection_rate
      return f_inj
    return f

  def create_eruption_source(self, p_erupt = 5e6, mu_erupt= 1e5, r_conduit = 25): # Eruption parameters # TODO: melt mu(T(z))?
    def f(t, q):
      ''' Compute eruption rate at index -1 '''
      p_node = self.pressure(q)
      f_erupt = np.zeros((self.num_dof))
      # Compute pressure in excess of critical eruption overpressure
      deltap = (p_node[-1] - self.nodes[-1].p_init) - p_erupt
      if deltap > 0:
        eruption_rate = self.nodes[-1].rhoref * (deltap / (16.0 * mu_erupt)) * r_conduit * r_conduit * r_conduit
        # Set eruption rate in mass conservation equation
        f_erupt[self.data_slice_global(-1, "mass")] = -eruption_rate
      return f_erupt
    
    # Set indices of eruptibe nodes
    return f

  def create_eruptible_layer(self,
                             p_erupt_min=5e6,
                             p_erupt_max=10e6,
                             z_min=-5e6,
                             z_max=0.0,
                             mu_erupt= 1e5,
                             r_conduit=25, distr_method="linear"):
    
    ''' Return a function that computes the eruption rate for an eruptible layer.
    The eruptible layer assigns an eruption pressure to a layer bounded by
      (z_min, z_max)
    with critical eruption overpressure of
      (p_erupt_max, p_erupt_min)
    corresponding to the bottom and top of the layer, respectively.
    '''

    if distr_method == "linear":
      # Array of index-depth-pressure tuples for each node in eruptible layer 
      i_z_p_nodes = [(i,
                      node.z,
                      p_erupt_max + (node.z - z_min) / (z_max - z_min)
                        * (p_erupt_min - p_erupt_max))
                    for i, node in enumerate(self.nodes)
                    if node.z >= z_min and node.z <= z_max]
    else:
      raise ValueError(
        f"Eruption-pressure distribution method {distr_method} not recognized.")

    def f(t, q):
      ''' Compute eruption rate of all chambers '''
      p_node = self.pressure(q)
      # Allocate
      f_erupt = np.zeros((self.num_dof))
      # Compute pressure in excess of critical eruption overpressure
      for i, (node_idx, z, p_erupt) in enumerate(i_z_p_nodes):
        deltap = (p_node[node_idx] - self.nodes[node_idx].p_init) - p_erupt
        if deltap > 0:
          # Compute eruption rate based on erupting node parameters and 
          # specified r_conduit length scale
          eruption_rate = self.nodes[node_idx].rhoref * (
            deltap / (16.0 * mu_erupt)) * r_conduit * r_conduit * r_conduit
          # Set eruption rate in mass conservation equation
          f_erupt[self.data_slice_global(node_idx, "mass")] = -eruption_rate
      return f_erupt
    
    f.i_z_p_nodes = i_z_p_nodes    
    return f

  def simulation(self, q0, t_vec, f_inject:callable, f_erupt:callable, method_order=1,
                 solve_full_matrix = False, check_residual=False,
                 limit_eruption_rate_by_p=True):
    ''' Timestepping using a partially implicit scheme. Opening of network edges
    are done explicitly, with a "limiter" for eruption. '''

    # Start q with initial condition
    q = q0.copy()
    # Allocate full output storage
    q_out = np.zeros((t_vec.size, *q.shape))
    # Compute constant timestep
    dt = t_vec[1] - t_vec[0]

    # Legacy: fancy timestepping
    dt_last = np.nan
    dt_last_last = np.nan
    # Max order possible at each timestep
    max_order = np.ones(t_vec.size, dtype=int)
    self._step_strategy = np.zeros(t_vec.size, dtype=float)

    # Assemble L, M(t = 0), and f
    L, f = self.assemble_global_Lf(q)
    M = self.mass_rates(q, return_format="M")
    f = f.toarray()

    m_erupted = 0.0
    m_erupted_out = np.zeros((t_vec.size,))

    # Inverse desired: inv(scipy.sparse.eye(global_sys.num_dof) + dt * L + dt * M)
    dt = t_vec[1] - t_vec[0]
    # Additive separation of LHS matrix
    static_factor = scipy.sparse.eye(self.num_dof) + dt * L
    # Compute sparse inverse of static part
    static_inv = scipy.sparse.linalg.inv(scipy.sparse.csc_matrix(static_factor))

    def low_rank_update(f, Ainv, U, V):
      """ Compute inv(A + UV) @ f, given vector f and matrices inv(A), U, V. 
      """
      N_edges = np.min(U.shape)
      if N_edges == 0:
        return (Ainv @ f).ravel()
      t1 = Ainv @ f
      t2 = V @ t1
      t2 = scipy.linalg.solve(np.eye(N_edges) + V @ (Ainv @ U) , t2)
      return (t1 - Ainv @ (U @ t2)).ravel()
    
    # Construct scaling for state vector
    _scaling_vec = 0.01 * np.ones((self.num_dof))
    for i, node in enumerate(self.nodes):
      _scaling_vec[self.mass_indices[i]] = node.m0

    # Construct diagonal scaling matrix
    _scaling_mat = scipy.sparse.spdiags(_scaling_vec, 0, self.num_dof, self.num_dof)
    _scaling_mat_inv = scipy.sparse.spdiags(1/_scaling_vec, 0, self.num_dof, self.num_dof)

    if check_residual:
      self.residuals = np.zeros((t_vec.size,))

    for time_index, t in enumerate(t_vec):
      if time_index > 0:

        # Strang split
        # q = scipy.sparse.linalg.spsolve(scipy.sparse.eye(global_sys.num_dof) + dt * L, q + f * dt)

        # Evaluate initial eruption rate
        vec_f_erupt = f_erupt(t, q)

        # Eruption rate limiter for first-order explicit Euler
        # Compute elastic equilibrium for top node
        q_loc = q[(self.num_blocks-1) * self.block_size:
                  self.num_blocks * self.block_size]
        u_eq = (self.nodes[-1].H_mod @ q_loc + self.nodes[-1].k_mod).squeeze()
        m_eq = self.nodes[-1].m0 * (1 + 3 * u_eq[0] / self.nodes[-1].R0)

        if limit_eruption_rate_by_p:
          # Max eruption rate down to p0
          max_eruption_rate = np.abs(float(q[self.data_slice_global(-1, "mass")] - m_eq) / dt)
        else:
          # Max eruption rate down to m0
          max_eruption_rate = np.abs(float(q[self.data_slice_global(-1, "mass")] - self.nodes[-1].m0) / dt)

        vec_f_erupt = np.clip(vec_f_erupt, -max_eruption_rate, max_eruption_rate)
        # Integrate erupted mass
        m_erupted += -float(vec_f_erupt[self.data_slice_global(-1, "mass")]) * dt
        # Vector shape cleanup
        f_tot = np.reshape(
          f.ravel() + f_inject(t, q).ravel() + vec_f_erupt.ravel(),
          (self.num_dof, 1))

        RHS = q + dt * f_tot
        if solve_full_matrix:
          # Construct LHS matrix for backward Euler
          M = self.mass_rates(q, return_format="M")
          LHS_BE = scipy.sparse.eye(self.num_dof) + dt * (L + M)
          # Construct RHS data
          # Quasi-implicit one-step solve (strictly M(q^n) is used instead of M(q^n+1))
          q = scipy.sparse.linalg.spsolve(LHS_BE, RHS)
          # Reshape q
          q = np.reshape(q, (self.num_dof, 1))
        else:
          # Compute tuples (i, j, mass_rate_scaling)
          mass_transfer_tuples = self.mass_rates(q, return_format="tups")
          U = np.zeros((self.num_dof, len(mass_transfer_tuples)))
          V = np.zeros((len(mass_transfer_tuples), self.num_dof))
          for edge_idx, (i, j, mdot_scaling) in enumerate(mass_transfer_tuples):
            U[self.data_slice_global(i, "mass"), edge_idx] = 1.0
            U[self.data_slice_global(j, "mass"), edge_idx] = -1.0
            V[edge_idx, :] = dt * mdot_scaling * self.M_vecs[(i,j)]
          q_lr = low_rank_update(RHS, static_inv, U, V)
          # Reshape q
          q_lr = np.reshape(q_lr, (self.num_dof, 1))

          if check_residual:
            # Construct LHS matrix for backward Euler
            M = self.mass_rates(q, return_format="M")
            LHS_BE = scipy.sparse.eye(self.num_dof) + dt * (L + M)          
            res = np.linalg.norm((_scaling_mat_inv @ (LHS_BE @ q_lr - RHS)).squeeze())
            self.residuals[time_index] = res

          q = q_lr

      dt_last_last = dt_last
      dt_last = dt
      q = np.reshape(q, (q.size, 1))
      # Save result
      q_out[time_index,...] = q
      m_erupted_out[time_index] = m_erupted

    return q_out.squeeze(), m_erupted_out

  def sample_mass_rate_z(self, q_out, z_samples):
    ''' Return vertical mass rate as an array, where
      m_dot[i,j] = (t[i], z[j]),
    t is the vector of times specified for the simulation, and
    z spans z_min, z_max with a number of nodes equal to `z_samples`. '''
  
    # Compute sampling mesh
    z = np.array([node.z for node in self.nodes])
    z_scale = z.max() - z.min()
    z_sample_mesh = np.linspace(0, z_scale, z_samples)
    # Allocate
    mdot_grid = np.zeros((q_out.shape[0], z_sample_mesh.size))

    for time_idx in range(q_out.shape[0]):
      # Extract global state vector
      q = q_out[time_idx,:]
      # Compute strictly lower triangular mass transfer matrix
      M = self.mass_rates(q).tocoo()

      # For each positive mass rate along edge
      for i, j, mdot in zip(M.row, M.col, M.data):
        # Generate mask 
        z_i, z_j = self.nodes[i].z, self.nodes[j].z
        z_min = 0.5 * (z_i + z_j) - 0.5 * np.abs(z_j - z_i)
        z_max = 0.5 * (z_i + z_j) + 0.5 * np.abs(z_j - z_i)
        # Add mdot resolved onto z-axis for z-samples within z-bounds of edge 
        mdot_grid[time_idx,:] += np.where((z_sample_mesh >= z_min) & (z_sample_mesh < z_max),
                 np.sign(z_i - z_j) * mdot, 0.0)

    return z_sample_mesh, mdot_grid

  def make_composite_plot(self, t_vec, q_out, m_out,
                          z_samples=400,
                          include_m_out=True, node_scale= 10):
    ''' Generates useful plots and returns (fig, ax). '''

    # Set axis plotting scales (with manual axis labeling)
    t_plot_scale = 1e9
    m_plot_scale = 1e9

    # Plot specifications
    fig = plt.figure(figsize=(11,8), dpi=100)
    gridspec = fig.add_gridspec(4, 3, width_ratios=[1, 5, 1],
        height_ratios=[1, 2, 1, 2])
    ax = [fig.add_subplot(gridspec[0,1]),
          fig.add_subplot(gridspec[1,1]),
          fig.add_subplot(gridspec[2,1]),
          fig.add_subplot(gridspec[3,0]),
          fig.add_subplot(gridspec[3,1]),
          fig.add_subplot(gridspec[3,2]),]
    cmap = matplotlib.cm.hsv
    # Assign color to each node in cmap
    colors = cmap(np.linspace(0, 1, self.num_blocks,endpoint=False))

    ''' Create stacked area plot '''
    # Compute mass of each chamber
    m, p, u = self.compute_m_p_u(q_out)
    # Create mass color bar plot
    if include_m_out:
      # Augment data with erupted mass time series
      m_bars_data = np.concatenate((m.T, m_out[np.newaxis,:]), axis=0)
      colors_aug = np.concatenate((colors, [[0,0,0,1]]), axis=0)
    else:
      m_bars_data = m.T
      colors_aug = colors
    polys = ax[1].stackplot(t_vec/t_plot_scale,
                            (m_bars_data - m_bars_data.min(axis=1, keepdims=True))/m_plot_scale,
                            colors=colors_aug)
    ax[1].set_ylabel("$\Delta m$ ($10^9$ kg)", fontsize=12)
    ax[1].set_xlim(t_vec[0]/1e9, t_vec[-1]/1e9)

    ''' Create extrusive ratio plot '''
    # Compute time series of total mass in
    m_total = np.concatenate((m.T, m_out[np.newaxis,:]), axis=0).sum(axis=0)
    m_in_total = m_total - m_total[0]
    # Compute extrusive / total
    extrusive_ratio = np.zeros_like(m_in_total)
    np.divide(m_out, m_in_total, where=m_in_total!=0, out=extrusive_ratio)
    ax[0].plot(t_vec/t_plot_scale, extrusive_ratio, 'r')
    ax[0].set_ylabel(r"$\dot{m}_{out} / \dot{m}_{in}$", fontsize=12)
    ax[0].set_xlim(t_vec[0]/1e9, t_vec[-1]/1e9)
    ax[0].set_ylim(0, 1)

    ''' Create total mass erupted time series '''
    ax[2].plot(t_vec/t_plot_scale, m_out/m_plot_scale, 'k')
    ax[2].set_ylabel("Erupted ($10^9$ kg)", fontsize=12)
    ax[2].set_xlim(t_vec[0]/1e9, t_vec[-1]/1e9)

    ''' Create mdot(z) color plot '''
    # Sample mass rates at fixed z_sample
    z_sample, mdot_grid = self.sample_mass_rate_z(q_out, z_samples)
    # Create color map with range of mdot_grid
    clim = (mdot_grid.min(), mdot_grid.max(),)
    shift_div_cmap = zero_aligned_cmap(clim)
    if np.all(mdot_grid == 0):
      print("Data values are all zero. Skipping mass rate spatial plot.")
    else:
      plt.sca(ax[4])
      mg_t, mg_z = np.meshgrid(t_vec, z_sample)
      plt.contourf(mg_t/1e9, mg_z/1e3, mdot_grid.T, cmap=shift_div_cmap, levels=np.linspace(clim[0], clim[1], 100))
      cb = plt.colorbar(ax=ax[5], location='left')#, ax=ax[:])
      ax[5].set_visible(False)
      cb.set_label("Upward mass rate (kg/s)", fontsize=12)
      plt.xlabel("Time ($10^9$ s)", fontsize=12)
      plt.ylabel("Depth (km)", fontsize=12)

    fig.tight_layout()

    # Plot graph, with no border
    self.show_network(q_out[0,:], ax=ax[3], node_scale=node_scale, add_ax_labels=False, font_size=0, clip_on=False)
    [spine.set_visible(False) for spine in ax[3].spines.values()]

    ax[3].set_ylim(0, (z_sample[-1] - z_sample[0])/1e3)

    fig.set_tight_layout(True)

    return fig, ax

  def residence_time_sim(self, t_vec, q_out, N_particles=10000,
                         create_plot=True, add_legend=True):
    ''' Computes residence time statistics as a post-process.
    A set of particles are placed at node 0, and with Poisson probability
    proportional to mass rate divided by current node mass, the particles walk
    randomly on the graph.

    Returns (node_location, node_z), where
      node_location is the history of node indices for a particle ensemble,
        with size (t_vec.size, N_particles)
      node_z is the history of z-position for a particle ensemble, with
        size (t_vec.size, N_particles)
    If create_plot is True, a plot is created showing the mean and quartile
    depths of the particle ensemble. Flag add_legend specifies whether to
    include length in the plot.

    Note: dqdt = (f + f_inj + f_erupt) - (L + M) @ q, and we can extract
    dmdt from the appropriate indices. At the same time, for
      F_diag = scipy.sparse.diags(
        (f + f_inj + f_erupt)[self.mass_indices].toarray().squeeze())
    At the same time,
      dmdt = (self.mass_rates(q) + F_diag).todense().sum(axis=1).T
    '''

    # Initialize set of particles at node 0
    curr_node = np.full((N_particles,), 0, dtype=int)
    # Allocate node location history for each particle
    node_location = np.full((t_vec.size, N_particles), -1, dtype=int) 
    # Fill initial
    node_location[0,:] = 0

    for i, t in enumerate(t_vec[1:]):
      # Load q, pressure from ODE solution
      q = q_out[i,...]

      p_node = self.pressure(q)

      # Compute turnover rates (mdot / m)
      turnover_rates = scipy.sparse.diags(1/q[self.mass_indices].squeeze()) @ self.mass_rates(q)
      # Compute instantaneous Poisson rates from total out rate from node j
      poisson_lambda = np.array(turnover_rates.sum(axis=0)).squeeze()
      # Probability of escaping from node j to node i given that particle escapes
      path_probability = np.full(turnover_rates.shape, 1 / self.num_blocks)
      np.divide(turnover_rates.toarray(), poisson_lambda, where=poisson_lambda!=0, out=path_probability)
      # Probability of exiting in current timestep
      prob = poisson_lambda[curr_node] * (t_vec[i] - t_vec[i-1])

      # End walk at last node
      prob[-1] = 0

      # Determine whether each particle exits node
      exit_roll = np.random.rand(N_particles) < prob
      # Choose exit path based on probability
      new_node = (np.random.rand(N_particles)[np.newaxis,:]
                  < np.cumsum(path_probability[:,curr_node], axis=0)).argmax(axis=0)
      # Move all particles
      curr_node = np.where(exit_roll, new_node, curr_node)
      # Save particle location
      node_location[i+1,:] = curr_node

    if create_plot:
      t_plot_scale = 1e9
      z_plot_scale = 1e3
      # Convert residence node to depth
      node_z = np.array([node.z for node in self.nodes])[node_location]
      # Ensemble statistics at current timestep
      loc_quants = np.quantile(node_z, [0.25, 0.5, 0.75], axis=1)
      loc_mean = node_z.mean(axis=1)
      loc_std = node_z.std(axis=1)
      plt.plot(t_vec/t_plot_scale, loc_quants[1,:]/z_plot_scale, 'k-', label="median")
      plt.plot(t_vec/t_plot_scale, loc_mean/z_plot_scale, 'k--', label="mean")
      # Shade between 1- and 3-quartiles
      plt.fill_between(t_vec/t_plot_scale,
                       loc_quants[0,:]/z_plot_scale,
                       loc_quants[2,:]/z_plot_scale, color=[0.1, 0, 1.0, 0.5])
      plt.xlabel("Time ($10^9$ s)", fontsize=12)
      plt.ylabel("Depth (km)", fontsize=12)
      # plt.title("Tracer location quartiles")
      if add_legend:
        plt.legend()

    return node_location, node_z

  def compute_effective_connectivity(self, t_vec, q_out, window_nodes_vec=None):
    ''' Compute effective connectivity at several averaging timescales
    Returns list of tuples [(n, dt_window, t_window_center, effective_conductivity)]
    If windows_nodes_vec is not provided, selects some windows of size ~4^n.
    '''

    if window_nodes_vec is None:
      # Select window nodes with size ~4^n
      window_nodes_vec = 2**np.arange(1, int(np.log2(t_vec.size)), 2) - 1

    # Compute Y(t)
    Y_list = [None for _ in range(q_out.shape[0])]
    for i in range(q_out.shape[0]):
      # Get instantaneous, asymmetric connectivity matrix of graph
      Y_native = self.get_connectivity(q_out[i,:])
      # Symmetrized connectivity
      Y_list[i] = scipy.sparse.csr_matrix(np.maximum(Y_native, Y_native.T))

    all_effective_conductivities = []
    for n in window_nodes_vec:
      # From t_vec compute array of window-center times
      if n == 1:
        t_avg_range = t_vec
      else:
        t_avg_range = 0.5 * (t_vec[:-n+1] + t_vec[n-1:]) 
      # Compute window size
      dt_window = n * (t_vec[1] - t_vec[0])

      # Allocate output for current window size
      effective_cond = np.zeros_like(t_avg_range, dtype=float)

      for i in range(t_vec.size - n + 1):
        # Time-averaged connectivity
        Y_avg = np.sum(np.abs(Y_list[i:i+n])) / n
        if Y_avg.nnz == 0:
          # No edges, zero connectivity 
          continue
        # Check node 0 is connected to node -1
        G = nx.Graph(Y_avg)
        if not nx.has_path(G, 0, self.num_blocks-1):
          continue
        # Construct s-t test vector
        chi = np.zeros(Y_avg.shape[0],)
        chi[[0, -1]] = [1.0, -1.0]
        # Compute effective conductivity using conjugate gradient for PSD matrix
        v, exit_code = scipy.sparse.linalg.cg(nx.laplacian_matrix(G), chi)
        if exit_code != 0:
          raise ValueError(f"scipy.sparse.linalg.cg terminated with unsuccessful exit code {exit_code}.")
        st_resistance = np.dot(chi, v)
        # Save effective conductivity at time t to output vector
        effective_cond[i] = 1.0 / st_resistance

      all_effective_conductivities.append((n, dt_window, t_avg_range, effective_cond))

    return all_effective_conductivities


if __name__ == "__main__":

  print("cnetwork.py is a tool file; import GlobalSystemThreshold to use.")