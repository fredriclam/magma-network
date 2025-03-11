import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt

g = 10

class Node():
  ''' Base graph node class '''
  def __init__(self):
    pass

def p_lithostatic(z, p_surf=1e5, z_surf=0, rho_crust=2.8e3, g=10):
  ''' Lithostatic pressure as function of depth z. '''
  return p_surf - rho_crust * g * (z - z_surf)

def T_geothermal(z, T_surf=273.15, z_surf=0, grad=-25/1e3):
  ''' Crust geothermal temperature as function of depth z.
  Default gradient is (25K/km). '''
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


class MagmaChamber(Node):
  def __init__(self,
               x:float=np.nan, y:float=0.0, z:float=np.nan,
               p_setting:object=None, T_setting:object=None,
               V_setting:object=None,
               c_v=1e3, K=1e9, v0=1/2.5e3, p0=25e6, g=10):
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
    self.v0 = v0
    self.rho0 = 1.0/v0
    self.p0 = p0
    self.g = g

    # Spatial settings
    self.x = x
    self.y = y
    self.z = z

    # Parse setting objects: pressure
    self.p_init = np.nan
    if p_setting is None:
      self.p_init = p_lithostatic(z)
    elif isinstance(p_setting, (int, float)):
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
      self.T_init = T_geothermal(z)
    elif isinstance(T_setting, (int, float),):
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
    self.m = self.V / MagmaChamber.v_p(self.p_init, self.K, self.v0, self.p0)
    # Compute energy via volume and volumetric energy (caloric equation)
    self.E = self.V * MagmaChamber.volenergy_pT(
        self.p_init, self.T_init, self.K, self.v0, self.p0, self.c_v)

  ''' Static equation of state (EOS) and caloric implementations

     (p - p0) / K = - (v - v0) / v0

  where K is the magma bulk modulus and (p0, v0) is a linearization point.
  Energy is defined as

     E = m * c_v * T + 0.5 * (p - p0)^2 / K,

  where c_v is a constant-volume heat capacity.

  '''

  @staticmethod
  def v_p(p, K, v0, p0):
    ''' Specific volume as function of pressure from EOS
    '''
    return v0 * (1 - (p - p0) / K)

  @staticmethod
  def p_v(v, K, v0, p0):
    ''' Pressure as function of specific volume from EOS
    '''
    return p0 - K * (v - v0) / v0

  @staticmethod
  def strain_volenergy_p(p, K, v0, p0):
    ''' Strain energy in magma as function of pressure from EOS
    '''
    return 0.5 * (p - p0)*(p - p0) / K

  @staticmethod
  def strain_volenergy_v(v, K, v0, p0):
    ''' Strain energy in magma as function of spec. vol. from EOS
    '''
    return 0.5 * (1 - v/v0)*(1 - v/v0) * K

  @staticmethod
  def volenergy_pT(p, T, K, v0, p0, c_v):
    ''' Volumetric energy from p, T '''
    e_mech = 0.5 * (p - p0)*(p - p0) / K
    e_int  = c_v * T / MagmaChamber.v_p(p, K, v0, p0)
    return e_mech + e_int

  ''' Dependent quantities as object properties '''

  @property
  def U(self) -> np.array:
    ''' Vector of dependent variables U = [m, E, V] '''
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
    return MagmaChamber.p_v(self.v, self.K, self.v0, self.p0)

  @property
  def strain_volenergy(self):
    ''' Strain energy per volume and parameters in self. '''
    return MagmaChamber.strain_volenergy_v(self.V/self.m, self.K, self.v0, self.p0)

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

class GlobalSystem():
  ''' Global coupled system of chambers with methods for manipulating the network.
  Assumes a fixed admittance matrix Y. Uses unoptimized numerical scheme.

  This is an approximation to the continuum percolation formulation, where the
  percolation condition is strictly based on a distance cutoff.
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

  def __init__(self, Y:np.array, t_b, t_d, K_crust, G_crust,
               rho0=2500, R0=100, p0=10e6, K_f=10e9, Nr=50):
    # Save parameters TODO: generalize to chamber-by-chamber input
    self.Y = Y

    self.R0 = R0
    self.rho0 = rho0
    self.p0 = p0
    self.K_f = K_f
    self.t_b = t_b
    self.t_d = t_d
    self.K_crust = K_crust
    self.G_crust = G_crust
    self.M_crust = K_crust + 4.0*G_crust/3.0

    self.Nr = Nr
    self.num_blocks = Y.shape[0]
    self.num_dof = self.num_blocks * self.block_size

    # Check implemented data schema
    self.check_schema_validity()

    # Compute reference mass
    self.m0 = rho0 * 4.0 / 3.0 * np.pi * R0 ** 3
    # Compute mesh info
    self.num_inf = 20*R0
    self.num_min = R0
    self.h = (self.num_inf - self.num_min) / (self.Nr-1)

    ''' Numerical differential operators '''
    # Define vector values 1/r
    self.r_mesh = np.linspace(self.num_min, self.num_inf, self.Nr)
    # Define diagonal matrix of values 1/r
    self.inv_r = scipy.sparse.diags([1.0/self.r_mesh], [0])

    # Initialize matrix H, vector k
    self.H = None
    self.k = None

  def op_D(self, h, Nr):
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

  def op_D2(self, h, Nr):
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

  def op_E_drr(self, h, Nr, r_mesh):
    ''' Linear mapping from radial displacement to spherically symmetric deviatoric rr-strain'''
    # Diagonal matrix containing values of 1/r
    diag_inv_r = scipy.sparse.diags([1.0/r_mesh], [0])
    E_drr = (2.0/3.0) * (self.op_D(h, Nr) - diag_inv_r)
    return E_drr

  def op_E_kk(self, h, Nr, r_mesh):
    ''' Linear mapping from radial displacement to spherically symmetric kk-strain'''
    # Diagonal matrix containing values of 1/r
    diag_inv_r = scipy.sparse.diags([1.0/r_mesh], [0])
    E_kk = self.op_D(h, Nr) + 2.0*diag_inv_r
    return E_kk

  def op_A(self, h, Nr, r_mesh):
    ''' Elasticity differential operator valid in the interior nodes:
          d^2/dr^2 + 2/r * d/dr - 2/r^2
     '''
    diag_inv_r = scipy.sparse.diags([1.0/r_mesh], [0])
    A = (self.op_D2(h, Nr)
         + 2.0 * diag_inv_r @ self.op_D(h, Nr)
         - 2.0 * diag_inv_r * diag_inv_r)
    return A

  def local_construct_affine_u_map(self) -> tuple:
    ''' Construct matrix and vector representing the mapping from time-dependent
    variables to radial displacement u, i.e., for a time-dependent vector q,
      u = Hq + k.
    Returns tuple (H, k). Inverts sparsely, but returns a possibly dense matrix H.
    '''

    # Add variables to scope
    Nr, h, block_size, r_mesh, R0, p0, m0, M_crust, K_crust, G_crust = (self.Nr, self.h,
      self.block_size, self.r_mesh, self.R0, self.p0, self.m0, self.M_crust, self.K_crust, self.G_crust)

    # Construct differential operators explicitly
    A = self.op_A(h, Nr, r_mesh)
    D = self.op_D(h, Nr)
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
    # Replace first row with boundary traction (normalized by M_crust) Dirichlet lift operator at r = R0 (linearized boundary treatment)
    L_u[0, :] = 0.0
    L_u[0, 0] += -1.0 / h
    L_u[0, 1] += 1.0 / h
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
    H = scipy.sparse.linalg.spsolve(L_u[0:Nr, 0:Nr], -L_u[0:Nr, Nr:])
    k = scipy.sparse.linalg.spsolve(L_u[0:Nr, 0:Nr], f_u)[:,np.newaxis]

    return H, k

  @property
  def Hk(self):
    ''' Wrapper for caching H, k matrices mapping time-dependent states q to u
    via
      u = H @ q + k.
    See also local_construct_affine_u_map '''
    if self.H is None or self.k is None:
      self.H, self.k = self.local_construct_affine_u_map()
    return self.H, self.k

  def local_construct_Lf(self) -> tuple:
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
    Nr, block_size, h, r_mesh = self.Nr, self.block_size, self.h, self.r_mesh
    t_d, t_b = self.t_d, self.t_b

    # Get affine map from state vector q to instantaneous displacement u
    H_block, k_block = self.local_construct_affine_u_map()

    # Assemble dependence of viscous strain evolution on displacement u (through elastic strain)
    G = scipy.sparse.lil_matrix((block_size, Nr))
    G[0:Nr, 0:Nr] = -1.0 / t_d * self.op_E_drr(h, Nr, r_mesh)
    G[Nr:2*Nr, 0:Nr] = -1.0 / t_b * self.op_E_kk(h, Nr, r_mesh)
    # Compute matrix L
    L_block = scipy.sparse.lil_matrix((block_size, block_size))
    L_block[np.arange(0,Nr), np.arange(0,Nr)] = (1 / t_d)
    L_block[np.arange(Nr,2*Nr), np.arange(Nr,2*Nr)] = (1 / t_b)
    # Add dependence on u through Schur complement term
    L_block += G @ H_block

    ''' Assemble local RHS vector for a single chamber
      This is f - G @ K,
    where f contains any external source terms for the time-dependent variables.
    '''
    # Assemble right hand side for local problem
    f_block = scipy.sparse.lil_matrix((block_size, 1))
    # Put dependence on spherical boundary condition
    f_block -= G @ k_block

    return L_block, f_block, H_block, k_block

  def assemble_global_Lf(self):
    ''' Assemble global matrix, coupling all chambers '''
    Nr, num_blocks, block_size = self.Nr, self.num_blocks, self.block_size

    L = scipy.sparse.lil_matrix((num_blocks * block_size, num_blocks * block_size))
    f = scipy.sparse.lil_matrix((num_blocks * block_size, 1))
    # Construct block representing independent viscoelastic evolution // TODO: have each depend on its own parameters
    L_block, f_block, H_block, k_block = self.local_construct_Lf()
    for i in range(num_blocks):
      L[i*block_size:(i+1)*block_size, i*block_size:(i+1)*block_size] = L_block
      f[i*block_size:(i+1)*block_size,0] = f_block

    ''' Add mass transfer terms

    Pressure differences between chambers are
    p_i - p_j = -(K_fi - K_fj) - (3 K_fi u_ri / R_i - 3 K_fj u_rj / R_j) + K_f * (m_i/m_0i - m_j/m_0j)
    and mass rate ~ rho_upstream * hyd_cond * (p_i - p_j).

    Here we estimate
    p_i - p_j = - 3 * K_f (u_ri / R_i - u_rj / R_j) + K_f * (m_i/m_0i - m_j/m_0j)
    and thus
    \dot{m}_{ij} = Adj_{ij} * hydr_cond * rho0 * K_f * (
      - 3 * (u_ri / R_i - u_rj / R_j) + (m_i/m_0i - m_j/m_0j)
    )
    where Adj is the adjacency matrix. Here the hydraulic conductivity has units of
    mass flux per pressure; that is, (m^3/s)/Pa in SI units.

    '''

    # Save the closed-system linear dynamics matrix
    self.L_closed_system = L.copy()

    for i in range(self.Y.shape[0]):
      for j in range(i+1, self.Y.shape[1]):
        # Compute mass rate coefficient (kg / s)
        _coeff = self.Y[i,j] * self.rho0 * self.K_f
        # Compute dependence of mass rate on ith viscoelastic field as u(r=R) / R0 through H_i
        L[i*block_size + 2*Nr, i*block_size:(i+1)*block_size] -= 3.0 * _coeff * H_block[0,:] / self.R0 # R0i
        L[j*block_size + 2*Nr, i*block_size:(i+1)*block_size] += 3.0 * _coeff * H_block[0,:] / self.R0 # R0i
        # Compute dependence of mass rate on jth viscoelastic field as u(r=R) / R0 through H_j
        L[i*block_size + 2*Nr, j*block_size:(j+1)*block_size] += 3.0 * _coeff * H_block[0,:] / self.R0 # H_j, R0j
        L[j*block_size + 2*Nr, j*block_size:(j+1)*block_size] -= 3.0 * _coeff * H_block[0,:] / self.R0 # H_j, R0j
        # Compute dependence of mass rate on ith chamber mass
        L[i*block_size + 2*Nr, i*block_size + 2*Nr] += _coeff / self.m0 # m0i -- note this diagonal term should be +
        L[j*block_size + 2*Nr, i*block_size + 2*Nr] -= _coeff / self.m0 # m0i
        # Compute dependence of mass rate on jth chamber mass
        L[i*block_size + 2*Nr, j*block_size + 2*Nr] -= _coeff / self.m0 # m0j
        L[j*block_size + 2*Nr, j*block_size + 2*Nr] += _coeff / self.m0 # m0j

    return L, f

  def pressure(self, q):
    ''' Compute vector of pressures, indexed by chamber number '''
    H_block, k_block = self.Hk
    p = np.zeros((self.num_blocks, 1))
    for i in range(self.num_blocks):
      # Compute boundary displacement
      u_R0 = (H_block @ q[i*self.block_size:(i+1)*self.block_size] + k_block)[0]
      dp_u = -3 * self.K_f * u_R0 / self.R0
      # Mass added pressure increase
      dp_m = self.K_f * (q[self.data_slice_global(i,"mass")] - self.m0) / self.m0
      p[i] = self.p0 + dp_u + dp_m
    return p

  def u(self, q):
    ''' Compute vector of displacements, indexed by chamber number '''
    H_block, k_block = self.Hk
    u = np.zeros((self.num_blocks, self.Nr))
    for i in range(self.num_blocks):
      u[i,:] = (H_block @ q[i*self.block_size:(i+1)*self.block_size] + k_block).squeeze()
    return u

  def sigma_rr(self, q):
    # Extract q blockwise, for each chamber
    H_block, k_block = self.Hk
    sigma_rr = np.zeros((self.num_blocks, self.Nr))

    for i in range(self.num_blocks):
      q_loc = q[i*self.block_size:(i+1)*self.block_size].squeeze()
      # Compute boundary displacement
      u_loc = (H_block @ q_loc + k_block.squeeze())
      # Radial component of strain
      radial = (self.op_D(self.h, self.Nr) @ u_loc)
      # Angular components (phi + theta) of stress div. by M_crust
      angular = u_loc / self.r_mesh
      # Elastic strain
      eps_drr = (2.0/3.0) * (radial - angular)
      eps_kk = radial + 2.0 * angular
      # Viscous strain γ_drr
      gamma_drr = q_loc[0:self.Nr]
      # Viscous strain γ_drr
      gamma_kk = q_loc[self.Nr:2*self.Nr]
      # Compute stress from elastic strain
      sigma_drr = 2 * self.G_crust * (eps_drr - gamma_drr)
      sigma_kk = 3 * self.K_crust * (eps_kk - gamma_kk)
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
  def eigshow(eig):
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

  def __init__(self, nodes:list, t_b, t_d, K_crust, G_crust, r_hydr, mu0,
               rho0=2500, K_f=10e9, Nr=50,
               dpdx_crit=1e3, dpdx_threshold_scale=1e2, R_outer_ratio=20,
               max_edge_dist=np.inf):
    self.nodes:list = nodes
    self.rho0 = rho0
    self.K_f = K_f
    self.t_b = t_b
    self.t_d = t_d
    self.K_crust = K_crust
    self.G_crust = G_crust
    self.r_hydr = r_hydr
    self.mu0 = mu0
    self.M_crust = K_crust + 4.0*G_crust/3.0
    self.Nr = Nr
    self.dpdx_crit = dpdx_crit
    self.dpdx_threshold_scale = dpdx_threshold_scale
    self.R_outer_ratio = R_outer_ratio
    self.max_edge_dist = max_edge_dist
    self.num_blocks = len(nodes)
    self.num_dof = self.num_blocks * self.block_size
    # Check implemented data schema for organizing field variables
    self.check_schema_validity()
    # Initialize nodes with linearization point, operators
    [self._init_node(node) for node in self.nodes]

    self.mat_props = dict(
      t_b = t_b,
      t_d = t_d,
      K_crust = K_crust,
      G_crust = G_crust,
      K_f = K_f,
      rho0 = rho0,
      mu0 = mu0,
      r_hydr = r_hydr,
    )

    # Set initial condition with initial (absolute) mass
    self.q0 = np.zeros((self.num_dof, 1))
    for i, node in enumerate(self.nodes):
      self.q0[self.data_slice_global(i, "mass")] = node.m0

    # Dictionary mapping ordered chamber index tuple (i,j), i < j to matrix
    # representing flow sparsity pattern
    self.M_stencils = dict()
    num_blocks, block_size = self.num_blocks, self.block_size
    for i in range(num_blocks):
      node_i = self.nodes[i]
      for j in range(i+1, num_blocks):
        node_j = self.nodes[j]
        # Set up dimensionless flow matrix for the first time
        M_loc = scipy.sparse.lil_matrix((num_blocks * block_size, num_blocks * block_size))
        # Compute dependence of mass rate on ith viscoelastic field as u(r=R) / R0 through H_i
        M_loc[i*block_size + 2*Nr, i*block_size:(i+1)*block_size] -= 3.0 * node_i.H[0,:] / node_i.R0 # R0i
        M_loc[j*block_size + 2*Nr, i*block_size:(i+1)*block_size] += 3.0 * node_i.H[0,:] / node_i.R0 # R0i
        # Compute dependence of mass rate on jth viscoelastic field as u(r=R) / R0 through H_j
        M_loc[i*block_size + 2*Nr, j*block_size:(j+1)*block_size] += 3.0 * node_j.H[0,:] / node_j.R0 # H_j, R0j
        M_loc[j*block_size + 2*Nr, j*block_size:(j+1)*block_size] -= 3.0 * node_j.H[0,:] / node_j.R0 # H_j, R0j
        # Compute dependence of mass rate on ith chamber mass
        M_loc[i*block_size + 2*Nr, i*block_size + 2*Nr] += 1.0 / node_i.m0 # m0i -- note this diagonal term should be +
        M_loc[j*block_size + 2*Nr, i*block_size + 2*Nr] -= 1.0 / node_i.m0 # m0i
        # Compute dependence of mass rate on jth chamber mass
        M_loc[i*block_size + 2*Nr, j*block_size + 2*Nr] -= 1.0 / node_j.m0 # m0j
        M_loc[j*block_size + 2*Nr, j*block_size + 2*Nr] += 1.0 / node_j.m0 # m0j
        # Register flow matrix to the pair (i,j), i < j
        self.M_stencils[(i,j,)] = M_loc.tocsr()

        # Distance validity check
        dist = float(np.sqrt((node_i.x - node_j.x) ** 2
                      + (node_i.y - node_j.y) ** 2
                      + (node_i.z - node_j.z) ** 2))
        if np.isclose(dist, 0.0):
          raise ValueError(f"Distance between chamber {i} and {j} is " \
                           f"close to zero ({dist:.2e}).")

  def _init_node(self, node) -> None:
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
    Nr = self.Nr
    block_size = self.block_size
    R_outer_ratio = self.R_outer_ratio
    K_f = self.K_f
    M_crust, K_crust, G_crust, = (self.M_crust, self.K_crust, self.G_crust,)

    # Set up mesh
    m0 = node.m
    R0 = (node.V / (4*np.pi/3))**(1.0/3.0)
    R_inf = R_outer_ratio * R0
    dx = (R_inf - R0) / (Nr-1)
    r_mesh = np.linspace(R0, R_inf, Nr)
    # Define diagonal matrix of values 1/r
    r_mesh_inv = scipy.sparse.diags([1.0 / r_mesh], [0])

    ''' Local differentiation construction '''
    # Matrix construction
    A = op_A(dx, Nr, r_mesh)
    D = op_D(dx, Nr)
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

    ''' Save other states to node '''
    node.dx = dx
    node.r_mesh = r_mesh
    node.inv_r = r_mesh_inv
    node.m0 = node.m
    node.R0 = R0
    node.p_init = node.p

    # Assemble dependence of viscous strain evolution on displacement u (through elastic strain)
    G = scipy.sparse.lil_matrix((block_size, Nr))
    G[0:Nr, 0:Nr] = -1.0 / self.t_d * op_E_drr(dx, Nr, r_mesh)
    G[Nr:2*Nr, 0:Nr] = -1.0 / self.t_b * op_E_kk(dx, Nr, r_mesh)
    # Compute matrix L
    node.L = scipy.sparse.lil_matrix((block_size, block_size))
    node.L[np.arange(0,Nr), np.arange(0,Nr)] = (1 / self.t_d)
    node.L[np.arange(Nr,2*Nr), np.arange(Nr,2*Nr)] = (1 / self.t_b)
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

  def get_connectivity(self, q):
    ''' Signed connectivity matrix with units of admittance '''
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
        # Compute distance
        dist = np.sqrt((node_i.x - node_j.x) ** 2
                       + (node_i.y - node_j.y) ** 2
                       + (node_i.z - node_j.z) ** 2)
        if np.isclose(dist, 0.0):
          raise ValueError("Distance between chamber {i} and {j} is zero.")
        elif dist > self.max_edge_dist:
          continue
        # Compute average pressure gradient
        dpdx = (p_node[i] - p_node[j]) / dist
        # Factor between (0, 1) that modulates flow between the two chambers
        if self.dpdx_crit != 0:
          threshold_factor = smoother(np.abs(dpdx) - self.dpdx_crit, self.dpdx_threshold_scale) * float(dpdx > 0)
          if threshold_factor > 1 or threshold_factor < 0:
            raise ValueError
        else:
          threshold_factor = 1.0

        # Compute flow admittance ( (m/s) / Pa )
        Y[i,j] = threshold_factor * self.r_hydr * self.r_hydr / 16.0 / self.mu0 / dist

    return Y

  def assemble_global_LMf(self, q, skip_Lf=False):
    ''' Assemble global matrix, coupling all chambers. The ODE system is
        (dq/dt) + L @ q + M(q) @ q = f,
        where L captures the viscoelastic effect and M captures mass transfer.
    '''

    # Get system size information
    num_blocks, block_size = self.num_blocks, self.block_size

    # Allocate global L, f matrices
    if not skip_Lf:
      L = scipy.sparse.lil_matrix((num_blocks * block_size, num_blocks * block_size))
      f = scipy.sparse.lil_matrix((num_blocks * block_size, 1))
      for i, node in enumerate(self.nodes):
        L[i*block_size:(i+1)*block_size, i*block_size:(i+1)*block_size] = node.L
        f[i*block_size:(i+1)*block_size,0] = node.f

    ''' Add mass transfer terms

    Pressure differences between chambers are
    p_i - p_j = -(K_fi - K_fj) - (3 K_fi u_ri / R_i - 3 K_fj u_rj / R_j) + K_f * (m_i/m_0i - m_j/m_0j)
    and mass rate ~ rho_upstream * hyd_cond * (p_i - p_j).

    Here we estimate
    p_i - p_j = - 3 * K_f (u_ri / R_i - u_rj / R_j) + K_f * (m_i/m_0i - m_j/m_0j)
    and thus
    \dot{m}_{ij} = Adj_{ij} * hydr_cond * rho0 * K_f * (
      - 3 * (u_ri / R_i - u_rj / R_j) + (m_i/m_0i - m_j/m_0j)
    )
    where Adj is the adjacency matrix. Here the hydraulic conductivity has units of
    mass flux per pressure; that is, (m^3/s)/Pa in SI units.

    '''

    # Compute pressures
    p_node = self.pressure(q)

    # Allocate global M matrix
    M = scipy.sparse.csr_matrix((num_blocks * block_size, num_blocks * block_size))

    # For each edge (i,j) there is a block matrix representing the connectivity
    # between the pair

    for i in range(num_blocks):
      node_i = self.nodes[i]
      for j in range(i+1, num_blocks):
        node_j = self.nodes[j]

        # Compute distance
        dist = float(np.sqrt((node_i.x - node_j.x) ** 2
                      + (node_i.y - node_j.y) ** 2
                      + (node_i.z - node_j.z) ** 2))
        # Compute average pressure gradient
        dpdx = (p_node[i] - p_node[j]) / dist
        # Factor between (0, 1) that modulates flow between the two chambers
        threshold_factor = float(smoother(np.abs(dpdx) - self.dpdx_crit,
                                    self.dpdx_threshold_scale))
        if threshold_factor > 1 or threshold_factor < 0:
          raise ValueError
        if threshold_factor > 1e-15:
          # Compute flow admittance ( (m/s) / Pa ) -- sign is determined automatically by multiplication with state vector q
          Y = threshold_factor * self.r_hydr * self.r_hydr / 16.0 / self.mu0 / dist
          # Multiply mass rate coefficient (kg / s) by dimensionless flow matrix M_loc
          M += (Y * self.rho0 * self.K_f) * self.M_stencils[(i,j,)]

    if skip_Lf:
      return M
    else:
      return L, M, f

  def pressure(self, q):
    ''' Compute vector of pressures, indexed by chamber number '''
    p = np.zeros((self.num_blocks, 1))
    for i, node in enumerate(self.nodes):
      # Compute boundary displacement
      u_R0 = (node.H @ q[i*self.block_size:(i+1)*self.block_size] + node.k)[0]
      dp_u = -3 * self.K_f * u_R0 / node.R0
      # Mass added pressure increase
      dp_m = self.K_f * ((q[self.data_slice_global(i,"mass")] - node.m0) / node.m0)
      p[i] = node.p0 + dp_u + dp_m
    return p

  def u(self, q):
    ''' Compute vector of displacements, indexed by chamber number '''
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
      radial = (op_D(node.dx, self.Nr) @ u_loc)
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
      sigma_drr = 2 * self.G_crust * (eps_drr - gamma_drr)
      sigma_kk = 3 * self.K_crust * (eps_kk - gamma_kk)
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
  def eigshow(eig):
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

  def simulation(self, t_vec, q0, method_order=1):
    ''' Timestepping using a partially implicit scheme. Opening of network edges
    are done explicitly, with a "limiter" for eruption. '''

    

    # Start q with initial condition
    q = q0.copy()
    # Allocate full output storage
    q_out = np.zeros((t_vec.size, *q.shape))
    # Save last dt for cache check
    dt = np.nan
    dt_last = np.nan
    dt_last_last = np.nan
    # Max order possible
    max_order = np.ones(t_vec.size, dtype=int)

    self._step_strategy = np.zeros(t_vec.size, dtype=float)
    L, M, f = self.assemble_global_LMf(q)
    # lu_out = scipy.sparse.linalg.splu(scipy.sparse.eye(global_sys.num_dof) + 0.5 * dt * L)

    m_erupted = 0.0
    m_erupted_out = np.zeros((t_vec.size,))

    ''' Set eruption parameters '''
    # Eruption parameters
    r_conduit = 25
    mu_erupt = 1e5
    # Overpressure required for eruption
    p_erupt = 5e6
    # Set total mass rate for injection
    mdot_inj = 3.0
    # Add source term for injection
    f_inj = 0.0 * f
    # Count nodes in the bottom layer for an N-way split
    N_split = nodes_per_layer[0]
    for i in range(N_split):
      f_inj[self.data_slice_global(i, "mass")] = mdot_inj / N_split

    for i, t in enumerate(t_vec):
      if i > 0:
        # Compute timestep
        dt = t_vec[i] - t_vec[i-1]

        M = self.assemble_global_LMf(q, skip_Lf=True)

        # Strang split      
        # q = scipy.sparse.linalg.spsolve(scipy.sparse.eye(global_sys.num_dof) + dt * L, q + f * dt)

        p_node = self.pressure(q)
        
        deltap = (p_node[-1] - self.nodes[-1].p0) - p_erupt
        f_erupt = 0.0 * f_inj
        if deltap > 0:
          eruption_rate = self.rho0 * (deltap / (16.0 * mu_erupt)) * r_conduit * r_conduit * r_conduit
          # Eruption rate limiter for first-order Euler
          max_eruption_rate = (q[self.data_slice_global(-1, "mass")] - self.nodes[-1].m0) / dt
          if eruption_rate > max_eruption_rate:
            eruption_rate = max_eruption_rate
          # Set eruption rate in mass conservation equation
          f_erupt[self.data_slice_global(-1, "mass")] = -eruption_rate
          # Integrate erupted mass
          m_erupted += eruption_rate * dt

        # Quasi-implicit one-step solve (strictly M(q^n) is used instead of M(q^n+1))
        q = scipy.sparse.linalg.spsolve(scipy.sparse.eye(self.num_dof) + dt * (L + M), q + dt * (f + f_inj + f_erupt))[:,np.newaxis]

        if False:
          # (1/3) BDF1, update matrix inv(I + dt*L)
          # lu_out = scipy.sparse.linalg.splu(scipy.sparse.eye(global_sys.num_dof) + 0.5 * dt * L)
          q = lu_out.solve(q + f * 0.5 * dt)
          # (2/3) Semi-BDF1, update matrix (approximate threshold explicitly, not implicitly)
          q = scipy.sparse.linalg.spsolve(scipy.sparse.eye(global_sys.num_dof) + dt * M, q)[:,np.newaxis]
          # (3/3) BDF1, update matrix inv(I + dt*L)
          q = lu_out.solve(q + f * 0.5 * dt)
          global_sys._step_strategy[i] = 1.0

        if False:
          if method == 2 and np.isclose(dt, dt_last) and not np.isclose(dt_last_last, dt_last):
            # BDF2, update matrix
            lu_out = scipy.sparse.linalg.splu(scipy.sparse.eye(global_sys.num_dof) + (2.0/3.0) * dt * L)
            q = lu_out.solve((4.0/3.0) * q + (2.0/3.0) * f * dt - (1.0/3.0) * q_out[i-2,...])
            global_sys._step_strategy[i] = 2.0
          elif method == 2 and np.isclose(dt, dt_last) and np.isclose(dt_last_last, dt_last):
            # BDF2, use cached matrix
            q = lu_out.solve((4.0/3.0) * q + (2.0/3.0) * f * dt - (1.0/3.0) * q_out[i-2,...])
            global_sys._step_strategy[i] = 2.5
          elif not np.isclose(dt, dt_last):
            # BDF1, update matrix inv(I + dt*L)
            lu_out = scipy.sparse.linalg.splu(scipy.sparse.eye(global_sys.num_dof) + dt * L)
            q = lu_out.solve(q + f * dt)
            global_sys._step_strategy[i] = 1.0
          else:
            # BDF1, use cached matrix inv(I + dt*L)
            q = lu_out.solve(q + f * dt)
            global_sys._step_strategy[i] = 1.5

      dt_last_last = dt_last
      dt_last = dt
      q = np.reshape(q, (q.size, 1))
      # Save result
      q_out[i,...] = q
      m_erupted_out[i] = m_erupted

      return q_out, m_erupted_out

def solve_network_N(N_row, total_vol, mass_inj, t_vec=None, N_t:int=100, method:int=1,
                    t_b=1e11, t_d=5e10, K_crust=10e9, G_crust=10e9, K_f=5e9, rho0=2500):
  ''' Solve network problem for an N-by-N grid.
  
  N_row:     Number of rows in grid
  total_vol: Total volume
  mass_inj:  Total mass injected in chamber -1
  method:    Order of BDF method to use [1|2]
  t_vec:     If None, uses N_t to compute a fixed t_vec. Else, uses t_vec directly
  '''

  if method > 2:
    method = 2
    print("Warning: method of order > 2 is not implemented. Using BDF2. ")

  # Ingest material properties
  mat_props = dict(
    t_b = t_b,
    t_d = t_d,
    K_crust = K_crust,
    G_crust = G_crust,
    K_f = K_f,
    rho0 = rho0,
  )

  # Create grid of chambers by coordinates x, z (fixing y = 0)
  N_col = N_row

  x_axis = np.linspace(0,2e3,N_col)
  z_axis = np.linspace(0,-2e3,N_row)

  # Compute 1-D arrays of x-coordinates and z-coordinates
  N_nodes = x_axis.size * z_axis.size
  mg_x, mg_z = np.meshgrid(x_axis, z_axis)
  x_nodes = mg_x.flatten()
  z_nodes = mg_z.flatten()

  # Create list of MagmaChamber objects with corresponding coordinates
  list_nodes = [MagmaChamber(x=x, y=0.0, z=z,
                    p_setting=None,
                    T_setting=None,
                    V_setting=0.1)
              for (x,z) in zip(x_nodes, z_nodes)]

  # Create weighted adjacency matrix by starting with dense graph + pruning
  # Symmetric distance matrix
  d = np.sqrt((x_nodes - x_nodes[:,np.newaxis]) ** 2 + (z_nodes - z_nodes[:,np.newaxis]) ** 2)
  # Constant viscosity assumption
  mu0 = 1e5
  # Effective hydraulic radius
  r_hydr = 1

  # Allocate flow admittance matrix
  Y = np.zeros_like(d)
  # Minimum distance; filter out zeros unless zero is the only entry
  dist_list = np.sort(np.array(np.ravel(d)))
  dist_list = dist_list[dist_list > 0]
  if len(dist_list) == 0:
    dist_list = np.array([0])

  # Adjacency filter: only nodes satisfying this condition are connected
  adj_filter:np.array = (d <= dist_list.min() * (1 + 1e-7))
  # Compute 1/dist into Y
  np.divide(1.0, d, where=(d != 0)&adj_filter, out=Y)
  # Compute flow admittance matrix ( (m/s) / Pa )
  Y *= r_hydr * r_hydr / 16 / mu0

  R0 = ((total_vol/N_nodes) / (4*np.pi/3))**(1.0/3.0)

  global_sys = GlobalSystem(Y, t_b, t_d, K_crust, G_crust,
                    rho0=rho0, R0=R0, p0=10e6, K_f=10e9, Nr=100)
  # Assemble L, f system
  L, f = global_sys.assemble_global_Lf()
  # Get slice for accessing mass from chamber [0]
  global_sys.data_slice_global(0, "mass")

  # Set initial condition
  q0 = np.zeros((global_sys.num_dof, 1))
  # Set absolute mass
  for i in range(global_sys.num_blocks):
    q0[global_sys.data_slice_global(i, "mass")] = global_sys.m0
  # Add mass increment in chamber N-1
  q0[global_sys.data_slice_global(global_sys.num_blocks - 1, "mass")] += mass_inj

  if t_vec is None:
    t1 = 1e9
    t2 = 0.5e12
    # Define vector of t points for both timescales
    t_vec = np.array([*np.linspace(0, t1, N_t+1), *np.linspace(t1, t2, N_t+1)[1:]])
  # Start q with initial condition
  q = q0.copy()
  # Allocate output vectors
  q_out = np.zeros((t_vec.size, *q.shape))
  # Save last dt for cache check
  dt = np.nan
  dt_last = np.nan
  dt_last_last = np.nan
  # Max order possible
  max_order = np.ones(t_vec.size, dtype=int)
  
  global_sys._step_strategy = np.zeros(t_vec.size, dtype=float)

  for i, t in enumerate(t_vec):
    if i > 0:
      # Compute timestep
      dt = t_vec[i] - t_vec[i-1]
        
      # q = scipy.sparse.linalg.spsolve(scipy.sparse.eye(global_sys.num_dof) + dt * L, q + f * dt)

      if method == 2 and np.isclose(dt, dt_last) and not np.isclose(dt_last_last, dt_last):
        # BDF2, update matrix
        lu_out = scipy.sparse.linalg.splu(scipy.sparse.eye(global_sys.num_dof) + (2.0/3.0) * dt * L)
        q = lu_out.solve((4.0/3.0) * q + (2.0/3.0) * f * dt - (1.0/3.0) * q_out[i-2,...])
        global_sys._step_strategy[i] = 2.0
      elif method == 2 and np.isclose(dt, dt_last) and np.isclose(dt_last_last, dt_last):
        # BDF2, use cached matrix
        q = lu_out.solve((4.0/3.0) * q + (2.0/3.0) * f * dt - (1.0/3.0) * q_out[i-2,...])
        global_sys._step_strategy[i] = 2.5
      elif not np.isclose(dt, dt_last):
        # BDF1, update matrix inv(I + dt*L)
        lu_out = scipy.sparse.linalg.splu(scipy.sparse.eye(global_sys.num_dof) + dt * L)
        q = lu_out.solve(q + f * dt)
        global_sys._step_strategy[i] = 1.0
      else:
        # BDF1, use cached matrix inv(I + dt*L)
        q = lu_out.solve(q + f * dt)
        global_sys._step_strategy[i] = 1.5

    dt_last_last = dt_last
    dt_last = dt
    q = np.reshape(q, (q.size, 1))
    # Save result
    q_out[i,...] = q

  # Tack on used material properties
  # TODO: integrate material properties into data object
  global_sys.mat_props = mat_props

  return t_vec, q_out, global_sys



if __name__ == "__main__":

  ''' Generate single magma chamber with (x,y,z),(p_litho, T_geotherm, V=100) '''
  R0_example = 100
  mc1 = MagmaChamber(x=100.0, y=0.0, z=-1000.0,
                    p_setting=10e6,
                    T_setting=None,
                    V_setting=4/3*np.pi*R0_example**3)
  print("Example magma chamber:")
  print(mc1)


  # Set Maxwell times
  t_b = 1e11
  t_d = 5e10
  K_crust = 10e9
  G_crust = 10e9
  K_f = 5e9
  rho0 = 2500

  # Fix total volume
  total_vol = (4/3)*np.pi*1000.0**3
  # Fix mass injection
  mass_inj = total_vol * rho0 * 0.001

  N_range = np.arange(1,7)
  t_outs = [None for _ in N_range]
  q_outs = [None for _ in N_range]
  gs_outs = [None for _ in N_range]

  # Define vector of t points spanning both timescales
  N_t = 100
  t1 = 1e7
  t2 = 1e8
  t_vec = np.array([*np.linspace(0, t1, N_t+1), *np.linspace(t1, t2, N_t+1)[1:]])

  for i, N in enumerate(N_range):
    t_outs[i], q_outs[i], gs_outs[i] = solve_network_N(N, total_vol, mass_inj, t_vec=t_vec, method=2,
                                                      t_b=1e6, t_d=1e6, K_crust=10e9, G_crust=10e9, K_f=5e9, rho0=2500)
    print(f"Solved network N = {N}.")

    ''' Post-process time-dependent simulation '''
  def post(t_vec, q_out, global_sys):
    masses = np.zeros((t_vec.size, global_sys.num_blocks, ))
    pressures = np.zeros((t_vec.size, global_sys.num_blocks, ))
    stresses = np.zeros((t_vec.size, global_sys.num_blocks, global_sys.Nr,))
    displacements = np.zeros((t_vec.size, global_sys.num_blocks, global_sys.Nr,))

    for i in range(q_out.shape[0]):
      # State vector q at time t
      q_t = q_out[i,...]
      masses[i,...] = np.array([q_t[global_sys.data_slice_global(i, "mass")]
                          for i in range(global_sys.Y.shape[0])]).squeeze()
      pressures[i,...] = np.array(global_sys.pressure(q_t)).squeeze()
      stresses[i,...] = np.array(global_sys.sigma_rr(q_t))
      displacements[i,...] = np.array(global_sys.u(q_t))
    return masses, pressures, stresses, displacements

  outputs = [post(*tup) for tup in zip (t_outs, q_outs, gs_outs)]
  m_outs, p_outs, sigma_rr_outs, u_outs = zip(*outputs)

  fig, ax = plt.subplots(1,len(outputs),figsize=(15,4.2))

  for i in range(len(outputs)):
    polys = ax[i].stackplot(t_outs[i][:len(t_outs[i])//2],
                          m_outs[i][:len(t_outs[i])//2].T - gs_outs[i].m0)
  fig.tight_layout()
  print("Excess mass distribution over shorter transfer timescale")

  for i in range(len(p_outs)):
    plt.loglog(t_outs[i], p_outs[i][:,-1] / 1e6, '.-', label=f"${N_range[i]} \\times {N_range[i]}$")
  plt.xlabel("Time (s)")
  plt.ylabel("Pressure (MPa)")
  plt.legend()
  plt.title("Chamber #-1 (injection site)")

  # Define vector of t points spanning both timescales
  N_t = 100
  t1 = 1e7
  t2 = 1e8
  t_vec = np.array([*np.linspace(0, t1, N_t+1), *np.linspace(t1, t2, N_t+1)[1:]])
  t_d_range = np.geomspace(1e5, 1e9, 11)

  t_outs = [None for _ in t_d_range]
  q_outs = [None for _ in t_d_range]
  gs_outs = [None for _ in t_d_range]

  for i, t_d in enumerate(t_d_range):
    t_outs[i], q_outs[i], gs_outs[i] = solve_network_N(4, total_vol, mass_inj, t_vec=t_vec, method=2,
                                                      t_b=t_d, t_d=t_d, K_crust=10e9, G_crust=10e9, K_f=5e9, rho0=2500)
    print(f"Solved network t_d = {t_d}.")

  outputs = [post(*tup) for tup in zip (t_outs, q_outs, gs_outs)]
  m_outs, p_outs, sigma_rr_outs, u_outs = zip(*outputs)

  outputs_short = outputs[::2]
  t_d_range_short = t_d_range[::2]

  fig, ax = plt.subplots(1,len(outputs_short),figsize=(15,4.2))

  for i in range(0, len(outputs_short)):
    polys = ax[i].stackplot(t_outs[i][:len(t_outs[i])],
                          m_outs[i][:len(t_outs[i])].T - gs_outs[i].m0)
    ax[i].set_title(f"$t_d = ${t_d_range_short[i]:.2e}")
  fig.tight_layout()
  print("Excess mass distribution over shorter transfer timescale")