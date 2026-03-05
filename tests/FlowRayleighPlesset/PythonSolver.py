"""
Python conversion of C++ Diffuse Interface Compressible Navier-Stokes Solver
Converted from Hydro2.H with HLLC Riemann solver
Uses Spec_Vol = 1 (energy per unit volume formulation)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time

class PhysicalParams:
    def __init__(self):
        # Fluid properties - matching C++ code
        # Fluid 0 (gas/bubble interior)
        self.gamma0 = 1.4           # Gas specific heat ratio
        self.p0_0 = 0.0             # Tammann pressure for gas [Pa]
        self.mu0 = 1.8e-5           # Gas dynamic viscosity [Pa*s]
        self.mu0_b = 0.0            # Gas bulk viscosity [Pa*s]
        self.cp0 = 1005.0           # Gas specific heat at constant pressure [J/(kg*K)]
        self.cv0 = 718.0            # Gas specific heat at constant volume [J/(kg*K)]
        
        # Fluid 1 (liquid)
        self.gamma1 = 1.4#7.15          # Liquid specific heat ratio (stiffened gas)
        self.p0_1 = 0.0 #3.0e8           # Tammann pressure for liquid [Pa]
        self.mu1 = 1e-3             # Liquid dynamic viscosity [Pa*s]
        self.mu1_b = 0.0            # Liquid bulk viscosity [Pa*s]
        self.cp1 = 4186.0           # Liquid specific heat at constant pressure [J/(kg*K)]
        self.cv1 = 4186.0           # Liquid specific heat at constant volume [J/(kg*K)]
        
        # Surface tension
        self.sigma = 0.0728         # Surface tension [N/m]
        
        # Pressure conditions
        self.p_atm = 101325.0       # Atmospheric pressure [Pa]
        self.p_bubble = self.p_atm + 2e5  # Bubble pressure (2 bar above atm) [Pa]
        self.p_inf = self.p_atm    # Far-field pressure [Pa]
        self.pref = 0.0             # Reference pressure for Roe solver
        
        # Initial bubble radius
        self.R0 = 1e-4              # Initial radius [m] = 0.1 mm
        
        # Phase field parameters
        self.epsilon = 0.001        # Interface thickness [m]
        self.cutoff = 1e-8          # Eta cutoff value
        
        # Numerical parameters
        self.small = 1e-10          # Small regularization value
        self.lagrange = 0.0         # Lagrange no-penetration factor
        self.g = 9.81               # Gravitational acceleration [m/s^2]
        
        # Flags
        self.apply_surface_tension = True
        self.apply_weight = False
        self.apply_vaporization = False
        self.static_eta = False
        
        # Curvature method
        self.kappa_method = 2

class NumericalParams:
    def __init__(self, R0):
        # Domain size (100x bubble radius to avoid boundary effects)
        self.L = 100 * R0           # Domain size [m]
        self.nx = 128               # Grid points in x
        self.ny = 128               # Grid points in y
        self.dx = self.L / self.nx  # Grid spacing x
        self.dy = self.L / self.ny  # Grid spacing y
        
        # Time stepping
        self.cfl = 0.3              # CFL number
        self.cfl_v = 0.3            # Viscous CFL number
        self.dt_max = 1e-8          # Maximum time step [s]
        self.t_end = 5e-5           # End time [s]
        self.save_interval = 50     # Save every N steps

class HLLCSolver:
    """HLLC Riemann solver matching C++ implementation"""
    
    @staticmethod
    def solve(rho_L, Mn_L, Mt_L, E_L, gamma_L, p0_L,
              rho_R, Mn_R, Mt_R, E_R, gamma_R, p0_R,
              p_ref, small):
        """
        HLLC Riemann solver with Tammann EOS
        Spec_Vol = 1: energies are per unit volume
        """
        
        # Ensure no negative densities
        rho_L = max(small, rho_L)
        rho_R = max(small, rho_R)
        
        # Fluid primitives (per unit volume)
        ke_L = 0.5 * (Mn_L**2 + Mt_L**2) / (rho_L + small)
        ue_L = E_L - ke_L
        p_L = (gamma_L - 1.0) * ue_L - gamma_L * p0_L + p_ref
        
        ke_R = 0.5 * (Mn_R**2 + Mt_R**2) / (rho_R + small)
        ue_R = E_R - ke_R
        p_R = (gamma_R - 1.0) * ue_R - gamma_R * p0_R + p_ref
        
        # Velocities
        u_L = Mn_L / (rho_L + small)
        u_R = Mn_R / (rho_R + small)
        v_L = Mt_L / (rho_L + small)
        v_R = Mt_R / (rho_R + small)
        
        # Sound speeds
        a_L = np.sqrt(gamma_L * (p_L + p0_L) / (rho_L + small))
        a_R = np.sqrt(gamma_R * (p_R + p0_R) / (rho_R + small))
        
        # Wave speed estimates
        S_L = min(u_L - a_L, u_R - a_R)
        S_R = max(u_L + a_L, u_R + a_R)
        
        # Middle wave speed (contact discontinuity)
        S_star = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)) / \
                 (rho_L * (S_L - u_L) - rho_R * (S_R - u_R) + small)
        
        # Star states
        rho_star_L = rho_L * (S_L - u_L) / (S_L - S_star + small)
        rho_star_R = rho_R * (S_R - u_R) / (S_R - S_star + small)
        
        E_star_L = rho_star_L * (E_L / rho_L + (S_star - u_L) * 
                                 (S_star + p_L / (rho_L * (S_L - u_L) + small)))
        E_star_R = rho_star_R * (E_R / rho_R + (S_star - u_R) * 
                                 (S_star + p_R / (rho_R * (S_R - u_R) + small)))
        
        # Fluxes for left and right states (per unit volume)
        f_L_mass = rho_L * u_L
        f_L_mom_n = rho_L * u_L * u_L + p_L
        f_L_mom_t = rho_L * u_L * v_L
        f_L_energy = u_L * (E_L + p_L)
        
        f_R_mass = rho_R * u_R
        f_R_mom_n = rho_R * u_R * u_R + p_R
        f_R_mom_t = rho_R * u_R * v_R
        f_R_energy = u_R * (E_R + p_R)
        
        # Star fluxes
        f_star_L_mass = f_L_mass + S_L * (rho_star_L - rho_L)
        f_star_L_mom_n = f_L_mom_n + S_L * (rho_star_L * S_star - rho_L * u_L)
        f_star_L_mom_t = f_L_mom_t + S_L * (rho_star_L * v_L - rho_L * v_L)
        f_star_L_energy = f_L_energy + S_L * (E_star_L - E_L)
        
        f_star_R_mass = f_R_mass + S_R * (rho_star_R - rho_R)
        f_star_R_mom_n = f_R_mom_n + S_R * (rho_star_R * S_star - rho_R * u_R)
        f_star_R_mom_t = f_R_mom_t + S_R * (rho_star_R * v_R - rho_R * v_R)
        f_star_R_energy = f_R_energy + S_R * (E_star_R - E_R)
        
        # Compute HLLC flux based on wave positions
        if 0.0 <= S_L:
            flux_mass = f_L_mass
            flux_mom_n = f_L_mom_n
            flux_mom_t = f_L_mom_t
            flux_energy = f_L_energy
        elif S_L <= 0.0 <= S_star:
            flux_mass = f_star_L_mass
            flux_mom_n = f_star_L_mom_n
            flux_mom_t = f_star_L_mom_t
            flux_energy = f_star_L_energy
        elif S_star <= 0.0 <= S_R:
            flux_mass = f_star_R_mass
            flux_mom_n = f_star_R_mom_n
            flux_mom_t = f_star_R_mom_t
            flux_energy = f_star_R_energy
        else:
            flux_mass = f_R_mass
            flux_mom_n = f_R_mom_n
            flux_mom_t = f_R_mom_t
            flux_energy = f_R_energy
        
        return flux_mass, flux_mom_n, flux_mom_t, flux_energy

class DiffuseInterfaceSolver:
    def __init__(self, phys_params, num_params):
        self.pp = phys_params
        self.np = num_params
        
        # Create grid
        self.x = np.linspace(0, self.np.L, self.np.nx)
        self.y = np.linspace(0, self.np.L, self.np.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Initialize fields
        self.initialize_fields()
        
        # Storage for time history
        self.time_history = []
        self.radius_history = []
        
        print(f"Domain: {self.np.L*1e3:.2f} mm x {self.np.L*1e3:.2f} mm")
        print(f"Grid: {self.np.nx} x {self.np.ny}")
        print(f"Grid spacing: {self.np.dx*1e6:.3f} microns")
        print(f"Interface thickness: {self.pp.epsilon*1e6:.3f} microns")
        
    def initialize_fields(self):
        """Initialize phase field, velocity, pressure, and density"""
        # Center of domain
        cx, cy = self.np.L / 2, self.np.L / 2
        
        # Distance from center
        r = np.sqrt((self.X - cx)**2 + (self.Y - cy)**2)
        
        # Phase field: eta = 0 (gas/bubble), eta = 1 (liquid)
        # Using tanh profile
        self.eta = 0.5 * (1.0 + np.tanh((r - self.pp.R0) / (np.sqrt(2) * self.pp.epsilon)))
        
        # Initial densities for each phase
        rho0_init = 1.2  # Gas density [kg/m^3]
        rho1_init = 1000.0  # Liquid density [kg/m^3]
        
        # Mixture density
        self.rho = self.eta * rho0_init + (1.0 - self.eta) * rho1_init
        
        # Velocity field (initially zero)
        self.u = np.zeros_like(self.eta)
        self.v = np.zeros_like(self.eta)
        
        # Momentum
        self.Mx = self.rho * self.u
        self.My = self.rho * self.v
        
        # Initialize pressure field
        self.p = np.where(r < self.pp.R0, self.pp.p_bubble, self.pp.p_inf)
        
        # Compute initial energy (per unit volume)
        A = self.eta / (self.pp.gamma0 - 1.0) + (1.0 - self.eta) / (self.pp.gamma1 - 1.0)
        B = (self.eta * self.pp.gamma0 * self.pp.p0_0) / (self.pp.gamma0 - 1.0) + \
            ((1.0 - self.eta) * self.pp.gamma1 * self.pp.p0_1) / (self.pp.gamma1 - 1.0)
        
        # Internal energy per unit volume
        UE_vol = (self.p + self.pp.pref) * A + B
        
        # Kinetic energy per unit volume
        KE_vol = 0.5 * self.rho * (self.u**2 + self.v**2)
        
        # Total energy per unit volume
        self.E = KE_vol + UE_vol
        
    def gradient(self, field):
        """Compute gradient using central differences"""
        grad_x = np.zeros_like(field)
        grad_y = np.zeros_like(field)
        
        # Interior points
        grad_x[1:-1, 1:-1] = (field[2:, 1:-1] - field[:-2, 1:-1]) / (2 * self.np.dx)
        grad_y[1:-1, 1:-1] = (field[1:-1, 2:] - field[1:-1, :-2]) / (2 * self.np.dy)
        
        # Boundaries (one-sided differences)
        grad_x[0, :] = (field[1, :] - field[0, :]) / self.np.dx
        grad_x[-1, :] = (field[-1, :] - field[-2, :]) / self.np.dx
        grad_y[:, 0] = (field[:, 1] - field[:, 0]) / self.np.dy
        grad_y[:, -1] = (field[:, -1] - field[:, -2]) / self.np.dy
        
        return grad_x, grad_y
    
    def laplacian(self, field):
        """Compute Laplacian using central differences"""
        lap = np.zeros_like(field)
        
        # Interior points
        lap[1:-1, 1:-1] = (
            (field[2:, 1:-1] - 2*field[1:-1, 1:-1] + field[:-2, 1:-1]) / self.np.dx**2 +
            (field[1:-1, 2:] - 2*field[1:-1, 1:-1] + field[1:-1, :-2]) / self.np.dy**2
        )
        
        # Free-slip boundary conditions (zero normal derivative)
        lap[0, :] = lap[1, :]
        lap[-1, :] = lap[-2, :]
        lap[:, 0] = lap[:, 1]
        lap[:, -1] = lap[:, -2]
        
        return lap
    
    def compute_curvature(self, eta):
        """Compute interface curvature using method 2 from C++ code"""
        grad_eta_x, grad_eta_y = self.gradient(eta)
        grad_eta_mag = np.sqrt(grad_eta_x**2 + grad_eta_y**2 + self.pp.small)
        
        # Normal vector
        n_hat_x = grad_eta_x / grad_eta_mag
        n_hat_y = grad_eta_y / grad_eta_mag
        
        # Compute Hessian of eta
        grad_eta_x_x, grad_eta_x_y = self.gradient(grad_eta_x)
        grad_eta_y_x, grad_eta_y_y = self.gradient(grad_eta_y)
        
        # Curvature calculation (method 2)
        kappa = np.zeros_like(eta)
        
        for i in range(1, self.np.ny-1):
            for j in range(1, self.np.nx-1):
                if grad_eta_mag[i, j] > 1e-4:
                    # Hessian matrix
                    H = np.array([[grad_eta_x_x[i, j], grad_eta_x_y[i, j]],
                                  [grad_eta_y_x[i, j], grad_eta_y_y[i, j]]])
                    
                    n = np.array([n_hat_x[i, j], n_hat_y[i, j]])
                    
                    # Orthogonal tangent vector
                    if abs(n[0]) > abs(n[1]):
                        t = np.array([-n[1], n[0]]) / np.sqrt(n[0]**2 + n[1]**2 + self.pp.small)
                    else:
                        t = np.array([n[1], -n[0]]) / np.sqrt(n[0]**2 + n[1]**2 + self.pp.small)
                    
                    # Principal curvatures
                    kappa1 = -n.dot(H @ n)
                    kappa2 = -t.dot(H @ t) * 2.0 * self.pp.epsilon
                    
                    kappa[i, j] = kappa2
        
        return kappa, grad_eta_x, grad_eta_y, grad_eta_mag
    
    def compute_rhs(self, eta, rho, Mx, My, E):
        """Compute right-hand side of conservation equations"""
        
        # Compute primitive variables
        u = Mx / (rho + self.pp.small)
        v = My / (rho + self.pp.small)
        
        # Kinetic energy per unit volume
        KE_vol = 0.5 * rho * (u**2 + v**2)
        
        # Internal energy per unit volume
        UE_vol = E - KE_vol
        
        # Mixture properties
        A = eta / (self.pp.gamma0 - 1.0) + (1.0 - eta) / (self.pp.gamma1 - 1.0)
        B = (eta * self.pp.gamma0 * self.pp.p0_0) / (self.pp.gamma0 - 1.0) + \
            ((1.0 - eta) * self.pp.gamma1 * self.pp.p0_1) / (self.pp.gamma1 - 1.0)
        
        gamma_eff = 1.0 + (1.0 / A)
        p0_eff = (B / A) / gamma_eff
        
        # Pressure (Tammann EOS)
        p = (gamma_eff - 1.0) * UE_vol - gamma_eff * p0_eff + self.pp.pref
        
        # Speed of sound
        a = np.sqrt(gamma_eff * (p + p0_eff) / (rho + self.pp.small))
        
        # Compute curvature and gradients
        kappa, grad_eta_x, grad_eta_y, grad_eta_mag = self.compute_curvature(eta)
        
        # Surface tension force
        Fsv_x = np.zeros_like(eta)
        Fsv_y = np.zeros_like(eta)
        
        if self.pp.apply_surface_tension:
            Fsv_x = self.pp.sigma * kappa * grad_eta_x * self.pp.epsilon
            Fsv_y = self.pp.sigma * kappa * grad_eta_y * self.pp.epsilon
        
        # Compute fluxes using HLLC solver
        rho_flux = np.zeros_like(rho)
        Mx_flux = np.zeros_like(Mx)
        My_flux = np.zeros_like(My)
        E_flux = np.zeros_like(E)
        
        # X-direction fluxes
        for i in range(1, self.np.ny-1):
            for j in range(1, self.np.nx-2):
                # Left and right states
                rho_L = rho[i, j]
                rho_R = rho[i, j+1]
                Mn_L = Mx[i, j]
                Mn_R = Mx[i, j+1]
                Mt_L = My[i, j]
                Mt_R = My[i, j+1]
                E_L = E[i, j]
                E_R = E[i, j+1]
                gamma_L = gamma_eff[i, j]
                gamma_R = gamma_eff[i, j+1]
                p0_L = p0_eff[i, j]
                p0_R = p0_eff[i, j+1]
                
                # Solve Riemann problem
                f_mass, f_mom_n, f_mom_t, f_energy = HLLCSolver.solve(
                    rho_L, Mn_L, Mt_L, E_L, gamma_L, p0_L,
                    rho_R, Mn_R, Mt_R, E_R, gamma_R, p0_R,
                    self.pp.pref, self.pp.small
                )
                
                # Flux differences
                if j == 0:
                    rho_flux[i, j] -= f_mass / self.np.dx
                    Mx_flux[i, j] -= f_mom_n / self.np.dx
                    My_flux[i, j] -= f_mom_t / self.np.dx
                    E_flux[i, j] -= f_energy / self.np.dx
                
                if j < self.np.nx-2:
                    rho_flux[i, j+1] += f_mass / self.np.dx
                    Mx_flux[i, j+1] += f_mom_n / self.np.dx
                    My_flux[i, j+1] += f_mom_t / self.np.dx
                    E_flux[i, j+1] += f_energy / self.np.dx
        
        # Y-direction fluxes
        for i in range(1, self.np.ny-2):
            for j in range(1, self.np.nx-1):
                # Bottom and top states
                rho_L = rho[i, j]
                rho_R = rho[i+1, j]
                Mn_L = My[i, j]
                Mn_R = My[i+1, j]
                Mt_L = Mx[i, j]
                Mt_R = Mx[i+1, j]
                E_L = E[i, j]
                E_R = E[i+1, j]
                gamma_L = gamma_eff[i, j]
                gamma_R = gamma_eff[i+1, j]
                p0_L = p0_eff[i, j]
                p0_R = p0_eff[i+1, j]
                
                # Solve Riemann problem
                f_mass, f_mom_n, f_mom_t, f_energy = HLLCSolver.solve(
                    rho_L, Mn_L, Mt_L, E_L, gamma_L, p0_L,
                    rho_R, Mn_R, Mt_R, E_R, gamma_R, p0_R,
                    self.pp.pref, self.pp.small
                )
                
                # Flux differences
                if i == 0:
                    rho_flux[i, j] -= f_mass / self.np.dy
                    My_flux[i, j] -= f_mom_n / self.np.dy
                    Mx_flux[i, j] -= f_mom_t / self.np.dy
                    E_flux[i, j] -= f_energy / self.np.dy
                
                if i < self.np.ny-2:
                    rho_flux[i+1, j] += f_mass / self.np.dy
                    My_flux[i+1, j] += f_mom_n / self.np.dy
                    Mx_flux[i+1, j] += f_mom_t / self.np.dy
                    E_flux[i+1, j] += f_energy / self.np.dy
        
        # Compute RHS
        eta_rhs = np.zeros_like(eta)
        if not self.pp.static_eta:
            # Cahn-Hilliard: d(eta)/dt = -u*grad(eta)
            eta_rhs = -(u * grad_eta_x + v * grad_eta_y)
        
        rho_rhs = rho_flux
        Mx_rhs = Mx_flux + Fsv_x
        My_rhs = My_flux + Fsv_y
        E_rhs = E_flux + u * Fsv_x + v * Fsv_y
        
        return eta_rhs, rho_rhs, Mx_rhs, My_rhs, E_rhs
    
    def compute_timestep(self):
        """Compute CFL-limited time step"""
        u = self.Mx / (self.rho + self.pp.small)
        v = self.My / (self.rho + self.pp.small)
        
        # Compute speed of sound
        KE_vol = 0.5 * self.rho * (u**2 + v**2)
        UE_vol = self.E - KE_vol
        
        A = self.eta / (self.pp.gamma0 - 1.0) + (1.0 - self.eta) / (self.pp.gamma1 - 1.0)
        B = (self.eta * self.pp.gamma0 * self.pp.p0_0) / (self.pp.gamma0 - 1.0) + \
            ((1.0 - self.eta) * self.pp.gamma1 * self.pp.p0_1) / (self.pp.gamma1 - 1.0)
        
        gamma_eff = 1.0 + (1.0 / A)
        p0_eff = (B / A) / gamma_eff
        p = (gamma_eff - 1.0) * UE_vol - gamma_eff * p0_eff + self.pp.pref
        
        a = np.sqrt(gamma_eff * (p + p0_eff) / (self.rho + self.pp.small))
        
        # CFL condition
        u_max = np.max(np.abs(u))
        v_max = np.max(np.abs(v))
        a_max = np.max(a)
        
        wave_speed = a_max + np.sqrt(u_max**2 + v_max**2)
        dx_min = min(self.np.dx, self.np.dy)
        
        dt_acoustic = self.np.cfl * dx_min / (wave_speed + self.pp.small)
        
        # Viscous CFL
        mu_max = max(self.pp.mu0, self.pp.mu1)
        rho_min = np.min(self.rho)
        dt_viscous = self.np.cfl_v * rho_min * dx_min**2 / (mu_max + self.pp.small)
        
        dt = min(dt_acoustic, dt_viscous, self.np.dt_max) * 0.9
        
        return dt
    
    def compute_bubble_radius(self):
        """Compute equivalent bubble radius from phase field"""
        # Find area where eta < 0.5 (gas phase)
        gas_area = np.sum(self.eta < 0.5) * self.np.dx * self.np.dy
        
        # Equivalent radius
        R_eq = np.sqrt(gas_area / np.pi)
        return R_eq
    
    def step(self):
        """Perform one time step using forward Euler"""
        dt = self.compute_timestep()
        
        # Compute RHS
        eta_rhs, rho_rhs, Mx_rhs, My_rhs, E_rhs = self.compute_rhs(
            self.eta, self.rho, self.Mx, self.My, self.E
        )
        
        # Update
        self.eta = self.eta + dt * eta_rhs
        self.rho = self.rho + dt * rho_rhs
        self.Mx = self.Mx + dt * Mx_rhs
        self.My = self.My + dt * My_rhs
        self.E = self.E + dt * E_rhs
        
        # Apply cutoffs
        self.eta = np.clip((self.eta - self.pp.cutoff) / (1.0 - 2.0 * self.pp.cutoff), 0.0, 1.0)
        self.rho = np.maximum(self.rho, self.pp.small)
        
        # Update velocities
        self.u = self.Mx / (self.rho + self.pp.small)
        self.v = self.My / (self.rho + self.pp.small)
        
        return dt
    
    def run_simulation(self):
        """Run the full simulation"""
        print("\n" + "="*70)
        print("Starting diffuse interface simulation...")
        print("="*70)
        
        start_time = time.time()
        
        t = 0.0
        n = 0
        
        while t < self.np.t_end:
            dt = self.step()
            t += dt
            n += 1
            
            # Save data
            if n % self.np.save_interval == 0:
                R = self.compute_bubble_radius()
                self.time_history.append(t)
                self.radius_history.append(R)
                
                if n % (self.np.save_interval * 10) == 0:
                    print(f"Step {n}, t={t*1e6:.2f} us, R={R*1e6:.2f} um, dt={dt*1e9:.2f} ns")
        
        elapsed = time.time() - start_time
        print(f"\nSimulation completed in {elapsed:.2f}s")
        print(f"Total steps: {n}")
        
        return np.array(self.time_history), np.array(self.radius_history)

def rayleigh_plesset_2d(R, t, p_bubble, p_inf, rho_l, sigma, mu_l):
    """2D Rayleigh-Plesset equation"""
    R_val, R_dot = R
    
    if R_val < 1e-10:
        R_val = 1e-10
    
    delta_p = p_bubble - p_inf
    sigma_term = sigma / R_val
    visc_term = 2.0 * mu_l * R_dot / R_val
    
    R_ddot = ((delta_p - sigma_term - visc_term) / rho_l - R_dot**2) / R_val
    
    return [R_dot, R_ddot]

def solve_rayleigh_plesset(phys_params, t_span):
    """Solve the 2D Rayleigh-Plesset equation"""
    print("\nSolving 2D Rayleigh-Plesset equation...")
    
    y0 = [phys_params.R0, 0.0]
    
    solution = odeint(
        rayleigh_plesset_2d, 
        y0, 
        t_span,
        args=(phys_params.p_bubble, phys_params.p_inf, 1000.0, 
              phys_params.sigma, phys_params.mu1)
    )
    
    R_rpe = solution[:, 0]
    R_dot_rpe = solution[:, 1]
    
    print(f"RPE solution computed for {len(t_span)} time points")
    
    return R_rpe, R_dot_rpe

def plot_results(t_di, R_di, t_rpe, R_rpe, phys_params):
    """Plot comparison between diffuse interface and RPE results"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Radius vs time
    ax1 = axes[0]
    ax1.plot(t_di * 1e6, R_di * 1e6, 'b-', linewidth=2.5, label='Diffuse Interface (C++ Port)')
    ax1.plot(t_rpe * 1e6, R_rpe * 1e6, 'r--', linewidth=2, label='2D Rayleigh-Plesset')
    ax1.set_xlabel('Time [microseconds]', fontsize=12)
    ax1.set_ylabel('Bubble Radius [microns]', fontsize=12)
    ax1.set_title('Compressed Bubble Dynamics: C++ Port vs 2D RPE', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Relative difference
    ax2 = axes[1]
    R_rpe_interp = np.interp(t_di, t_rpe, R_rpe)
    rel_diff = np.abs(R_di - R_rpe_interp) / phys_params.R0 * 100
    ax2.plot(t_di * 1e6, rel_diff, 'g-', linewidth=2)
    ax2.set_xlabel('Time [microseconds]', fontsize=12)
    ax2.set_ylabel('Relative Difference [%]', fontsize=12)
    ax2.set_title('Relative Difference: |R_DI - R_RPE| / R_0', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bubble_comparison_cpp_port.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'bubble_comparison_cpp_port.png'")
    plt.show()

def main():
    """Main execution function"""
    print("="*70)
    print("C++ Diffuse Interface Solver - Python Port")
    print("Compressed Bubble: 2 bar above atmosphere")
    print("="*70)
    
    # Initialize parameters
    phys_params = PhysicalParams()
    num_params = NumericalParams(phys_params.R0)
    
    print(f"\nPhysical Parameters:")
    print(f"  Initial bubble radius: {phys_params.R0*1e6:.1f} microns")
    print(f"  Bubble pressure: {phys_params.p_bubble/1e5:.2f} bar")
    print(f"  Ambient pressure: {phys_params.p_inf/1e5:.2f} bar")
    print(f"  Pressure difference: {(phys_params.p_bubble - phys_params.p_inf)/1e5:.2f} bar")
    print(f"  Surface tension: {phys_params.sigma:.4f} N/m")
    
    # Run diffuse interface simulation
    solver = DiffuseInterfaceSolver(phys_params, num_params)
    t_di, R_di = solver.run_simulation()
    
    # Solve Rayleigh-Plesset equation
    t_rpe = np.linspace(0, num_params.t_end, 500)
    R_rpe, R_dot_rpe = solve_rayleigh_plesset(phys_params, t_rpe)
    
    # Plot comparison
    plot_results(t_di, R_di, t_rpe, R_rpe, phys_params)
    
    # Print summary statistics
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Initial radius: {phys_params.R0*1e6:.2f} microns")
    print(f"Final radius (DI): {R_di[-1]*1e6:.2f} microns")
    print(f"Final radius (RPE): {R_rpe[-1]*1e6:.2f} microns")
    print(f"Relative difference: {abs(R_di[-1] - R_rpe[-1])/phys_params.R0*100:.2f}%")
    print("="*70)

if __name__ == "__main__":
    main()
