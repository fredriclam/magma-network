import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import cnetwork

def run(mass_in_rate=5.0e2, fig_name="summary_output.png",
        npz_filename="temp_outs.npz", dpdx_crit=5e2, Nr=20,
        time_factor:int=1, t_d_min=1e9, t_d_max=1e15, t_vec=None,
        return_object_only=False):

  if t_vec is None:
    t_vec = np.linspace(0, 2e11*float(time_factor), 4000*int(time_factor))
  else:
    ''' t_vec is taken as an np.array '''
    pass

  ''' Generate geometry '''
  x_scale = 40e3
  z_scale = 40e3

  def ab_transform(x, a, b):
    ''' Maps random [0, 1] to [a, b] '''
    return a + (b - a) * x

  # Fix random seed for reproducible distribution
  np.random.seed(2)

  # Injection layer
  N_inject = 8
  x_inject = np.linspace(-0.5, 0.5, N_inject) * x_scale
  z_inject = np.zeros(N_inject) - z_scale

  # Propagation layer cluster 1
  N_prop1 = 15
  x_prop1 = ab_transform(np.random.rand(N_prop1), -0.5, -0.1) * x_scale
  z_prop1 = np.random.rand(N_prop1) * z_scale - z_scale

  # Propagation layer cluster 2
  N_prop2 = 5
  x_prop2 = ab_transform(np.random.rand(N_prop2), 0.0, 0.2) * x_scale
  z_prop2 = np.random.rand(N_prop2) * z_scale - z_scale

  # Propagation layer cluster 3
  N_prop3 = 3
  x_prop3 = ab_transform(np.random.rand(N_prop3), 0.4, 0.5) * x_scale
  z_prop3 = 0.5 * (0.5 + np.random.rand(N_prop3)) * z_scale - z_scale

  # Eruption layer
  N_erupt = 1
  if N_erupt == 1:
    x_erupt = np.linspace(0.0, 0.0, N_erupt) * x_scale
  else:
    x_erupt = np.linspace(-0.5, 0.5, N_erupt) * x_scale
  z_erupt = np.full(N_erupt, z_scale) - z_scale

  N_chamber = N_inject + N_prop1 + N_prop2 + N_prop3 + N_erupt

  # Set default properties--these are what each magma chamber takes as default if
  # properties are not individually specified for each chamber.

  crust_props = cnetwork.default_props(
    t_b_default=1e14,           # Maxwell / relaxation time (in isotropic strain) (Pa)
    t_d_default = 5e12,         # Maxwell / relaxation time (in deviatoric strain) (Pa)
    K_crust_default = 10e9,     # Bulk modulus of crust (Pa)
    G_crust_default = 10e9,     # Shear modulus of crust (Pa)
    K_f_default = 10e9,         # Bulk modulus of fluid / melt (Pa)
    r_hydr_default = 5,        # Hydraulic radius of channels / dikes (m)
    mu_default = 1e6,           # Melt viscosity in crust, constant (Pa s)
  )

  other_props = dict(
    R_min = 500,                # Minimum radius of magma chamber for random generation (m)
    R_max = 500,                # Maximum radius of magma chamber for random generation (m)
    N_chamber = N_chamber,             # Number of chambers to generate
    x_scale = x_scale,             # Length scale of horizontal (x) direction (m)
    z_scale = z_scale,             # Length scale of horizontal (x) direction (m)
    mass_in_rate = mass_in_rate,         # Input mass rate at bottom node (kg / s)
    p_erupt = 5e6,              # Overpressure required for eruption at top node (Pa)
    dpdx_crit = dpdx_crit,            # Minimum pressure gradient for opening dike (Pa / m)
    dpdx_threshold_scale = 0.0, # Numerical smoothing parameter for dpdx_crit (Pa / m)
    # Eruption parameters (the erupting conduit is treated differently from dikes between chambers)
    mu_erupt = 1e5,             # Melt viscosity in erupting conduit (Pa s)
    r_conduit_erupt = 25,       # Conduit radius for erupting conduit (m)
    rhoref = 2500,              # Reference density of fluid / melt (Pa)
  )

  numerics = cnetwork.default_numerics()
  numerics["Nr"] = Nr

  R_min = other_props["R_min"]
  R_max = other_props["R_max"]
  N_chamber = other_props["N_chamber"]
  x_scale = other_props["x_scale"]
  z_scale = other_props["z_scale"]
  rhoref = other_props["rhoref"]

  # Uniformly randomly distribution volume
  V_min = (4/3)*np.pi*R_min**3
  V_max = (4/3)*np.pi*R_max**3
  # List generation of chamber characteristics
  x_nodes = x_scale * np.random.rand(N_chamber)
  y_nodes = 0.0 * np.random.rand(N_chamber)
  z_nodes = np.linspace(0.0, z_scale, N_chamber)

  x_nodes = np.array([*x_inject, *x_prop1, *x_prop2, *x_prop3, *x_erupt])
  y_nodes = np.zeros_like(x_nodes)
  z_nodes = np.array([*z_inject, *z_prop1, *z_prop2, *z_prop3, *z_erupt])

  # Generate random magma chamber size (if R_max == R_min, no randomness)
  V_nodes = V_min + (V_max - V_min) * np.random.rand(N_chamber)

  # Generate list of MagmaChamber nodes, which store the volume, mass, energy in chamber
  list_nodes = [cnetwork.MagmaChamber(x=x, y=y, z=z,
    p_setting=None, vref=1.0/rhoref,
    T_setting=1000+273.15, V_setting=V)
    for (x,y,z,V) in zip(x_nodes, y_nodes, z_nodes, V_nodes)]

  def t_d_curve(z, mode=1):
    z_center = -20000
    z_scale = 1000
    a_min = t_d_min
    a_max = t_d_max
    if mode == 0:
      # Exponential decay
      val = a_min + (a_max - a_min) * np.exp((z - z_center)/z_scale)
    elif mode == 1:
      # Tanh
      val = a_min + 0.5 * (a_max - a_min) * (1 + np.tanh(0.5 * (z - z_center)/z_scale))
    
    return val

  # Append viscosity (Maxwell time)
  for node in list_nodes:
    node.t_d = t_d_curve(node.z, mode=1)
    node.t_b = node.t_d

  # Set up main system for timestepping later
  global_sys = cnetwork.GlobalSystemThreshold(list_nodes, **crust_props,
    **numerics, dpdx_crit=other_props["dpdx_crit"],
    dpdx_threshold_scale=other_props["dpdx_threshold_scale"],
    max_edge_dist=np.inf)

  if return_object_only:
    return global_sys

  f_erupt = global_sys.create_eruptible_layer(
    z_min=-z_scale*0.001,
    z_max=0.0,
    p_erupt_min=other_props["p_erupt"],
    p_erupt_max=other_props["p_erupt"],
    mu_erupt=other_props["mu_erupt"],
    r_conduit=other_props["r_conduit_erupt"])
  f_inj = global_sys.create_mass_injection_layer(
    fn_z=None,
    mdot_inj = other_props["mass_in_rate"] / N_chamber,
    z_max=-z_scale+.001,)

  # Time-dependent simulation
  q_out, m_erupted_out = global_sys.simulation(global_sys.q0, t_vec, f_inj, f_erupt)
  m_hist, p, u = global_sys.compute_m_p_u(q_out)

  ''' Output data to file '''
  np.savez_compressed(npz_filename, q_out=q_out,
    m_erupted_out=m_erupted_out, m_hist=m_hist, p=p, u=u, t_vec=t_vec)

  plt.figure(figsize=(9,5), dpi=200)

  plt.subplot(1,3,1)
  plt.plot(t_vec, m_erupted_out)
  plt.xlabel("Time (s)")
  plt.ylabel("Erupted mass (kg)")

  plt.subplot(1,3,2)
  # plt.plot(t_vec[:10000], p[:10000,-1] / 1e6)
  plt.plot(t_vec[:], p[:,-1] / 1e6)
  plt.plot(t_vec[:], p[:,0] / 1e6)
  plt.xlabel("Time (s)")
  plt.ylabel("Pressure (MPa) of top chamber")

  plt.subplot(1,3,3)
  # plt.plot(t_vec[:10000], m_hist[:10000,-1] / 1e12)
  plt.plot(t_vec, m_hist / 1e12)
  plt.xlabel("Time (s)")
  plt.ylabel("Mass ($10^{12}$ kg) of top chamber")

  plt.tight_layout()
  plt.draw()
  plt.savefig(fig_name, dpi=200)

  # Return abridged output
  return m_erupted_out, m_hist, p, u, t_vec