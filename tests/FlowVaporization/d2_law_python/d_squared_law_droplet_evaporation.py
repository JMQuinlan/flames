"""
D² Law for Spherical Droplet Evaporation
=========================================

This script implements the classical D² evaporation law for spherical droplets.
This is the most widely used benchmark in droplet combustion and spray modeling.

Physical Model:
- Spherical droplet of initial diameter d0
- Quasi-steady gas phase (Lewis number >> 1)
- Droplet diameter evolves as: d²(t) = d0² - K*t
- Evaporation constant: K = (8*rho_g*D_v/rho_l) * ln(1 + B_M)

Equations Used from Document:
- Eq. (3): Mass flux formulation (adapted to spherical geometry)
- Eq. (4): Spalding number B_M = (Y_vs - Y_inf)/(1 - Y_vs)
- Eq. (6): Surface vapor mass fraction from saturation pressure

Key Assumptions:
- Spherically symmetric (no deformation)
- Quasi-steady gas phase
- Constant properties
- No internal circulation
- Uniform droplet temperature

This is the foundational model for validating your diffuse-interface code.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')

# Set up nice plotting defaults
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['axes.grid'] = True


class DropletEvaporation:
    """
    Implements the classical D² law for droplet evaporation.
    
    This class solves the quasi-steady evaporation of a spherical droplet
    in a quiescent or convective environment using the Spalding number
    formulation from your document.
    """
    
    def __init__(self, fuel_type='ethanol'):
        """
        Initialize droplet evaporation model with fuel properties.
        
        Parameters:
        -----------
        fuel_type : str
            Either 'ethanol' or 'n-heptane'
        """
        self.fuel_type = fuel_type
        self.setup_properties()
        
    def setup_properties(self):
        """
        Set up all physical properties for the chosen fuel.
        Values from standard combustion references and your document.
        """
        
        if self.fuel_type == 'ethanol':
            # Ethanol (C2H5OH) properties
            self.W_v = 46.07e-3      # kg/mol, molecular weight of ethanol vapor
            self.rho_l = 789.0       # kg/m³, liquid density at 20°C
            self.cp_l = 2440.0       # J/(kg·K), liquid specific heat
            self.k_l = 0.169         # W/(m·K), liquid thermal conductivity
            self.L_v = 838e3         # J/kg, latent heat of vaporization
            self.T_boil = 351.4      # K, boiling point at 1 atm
            
            # Antoine equation coefficients (pressure in Pa, T in K)
            # log10(p_sat[Pa]) = A - B/(C + T[K])
            self.antoine_A = 5.24677
            self.antoine_B = 1598.673
            self.antoine_C = -46.424
            
            # Gas phase properties (from your document)
            self.D_v = 1.0e-5        # m²/s, binary diffusivity ethanol vapor in air
            
        elif self.fuel_type == 'n-heptane':
            # n-Heptane (C7H16) properties
            self.W_v = 100.2e-3      # kg/mol
            self.rho_l = 684.0       # kg/m³
            self.cp_l = 2240.0       # J/(kg·K)
            self.k_l = 0.124         # W/(m·K)
            self.L_v = 318e3         # J/kg
            self.T_boil = 371.6      # K
            
            # Antoine coefficients
            self.antoine_A = 4.02832
            self.antoine_B = 1268.636
            self.antoine_C = -56.199
            
            self.D_v = 6.0e-6        # m²/s (from your document)
        
        # Air properties (carrier gas)
        self.W_g = 28.97e-3          # kg/mol, molecular weight of air
        self.rho_g = 1.2             # kg/m³, gas density at ambient
        self.k_g = 0.026             # W/(m·K), gas thermal conductivity
        self.cp_g = 1005.0           # J/(kg·K), gas specific heat
        self.mu_g = 1.8e-5           # Pa·s, dynamic viscosity of air
        
        # Compute thermal diffusivity
        self.alpha_g = self.k_g / (self.rho_g * self.cp_g)
        
        # Schmidt and Prandtl numbers
        self.Sc = self.mu_g / (self.rho_g * self.D_v)  # Schmidt number
        self.Pr = self.mu_g * self.cp_g / self.k_g     # Prandtl number
        
        print(f"\n{'='*70}")
        print(f"D² Law Droplet Evaporation: {self.fuel_type.upper()}")
        print(f"{'='*70}")
        print(f"Fuel Properties:")
        print(f"  Liquid density:      ρ_l = {self.rho_l:.1f} kg/m³")
        print(f"  Molecular weight:    W_v = {self.W_v*1e3:.2f} g/mol")
        print(f"  Latent heat:         L_v = {self.L_v/1e3:.1f} kJ/kg")
        print(f"  Boiling point:       T_b = {self.T_boil:.1f} K")
        print(f"\nGas Phase Properties:")
        print(f"  Gas density:         ρ_g = {self.rho_g:.2f} kg/m³")
        print(f"  Vapor diffusivity:   D_v = {self.D_v*1e6:.2f} × 10⁻⁶ m²/s")
        print(f"  Schmidt number:      Sc  = {self.Sc:.2f}")
        print(f"  Prandtl number:      Pr  = {self.Pr:.2f}")
        print(f"{'='*70}\n")
    
    def saturation_pressure(self, T):
        """
        Calculate saturation pressure using Antoine equation.
        Required for Equation (6) from your document.
        
        Parameters:
        -----------
        T : float or array
            Temperature in Kelvin
            
        Returns:
        --------
        p_sat : float or array
            Saturation pressure in Pa
        """
        log10_p = self.antoine_A - self.antoine_B / (self.antoine_C + T)
        p_sat = 10**log10_p
        return p_sat
    
    def vapor_mass_fraction_surface(self, T_s, p_ambient):
        """
        Calculate surface vapor mass fraction using Equation (6).
        
        Implements your document's Equation (6):
        Y_vs = (W_v * x_vs) / (W_v * x_vs + W_g * (1 - x_vs))
        where x_vs = p_sat(T_s) / p
        
        Parameters:
        -----------
        T_s : float
            Surface temperature in K
        p_ambient : float
            Ambient pressure in Pa
            
        Returns:
        --------
        Y_vs : float
            Vapor mass fraction at surface
        """
        p_sat = self.saturation_pressure(T_s)
        x_vs = min(p_sat / p_ambient, 1.0)  # Mole fraction (capped at 1)
        
        # Convert to mass fraction (your Equation 6)
        numerator = self.W_v * x_vs
        denominator = self.W_v * x_vs + self.W_g * (1.0 - x_vs)
        Y_vs = numerator / denominator
        
        return Y_vs
    
    def spalding_number(self, T_s, p_ambient, Y_infinity=0.0):
        """
        Calculate Spalding mass transfer number using Equation (4).
        
        Implements your document's Equation (4):
        B_M = (Y_vs - Y_inf) / (1 - Y_vs)
        
        Parameters:
        -----------
        T_s : float
            Surface temperature in K
        p_ambient : float
            Ambient pressure in Pa
        Y_infinity : float
            Far-field vapor mass fraction
            
        Returns:
        --------
        B_M : float
            Spalding number
        """
        Y_vs = self.vapor_mass_fraction_surface(T_s, p_ambient)
        
        denominator = 1.0 - Y_vs
        if denominator < 1e-10:
            denominator = 1e-10
        
        B_M = (Y_vs - Y_infinity) / denominator
        
        return B_M
    
    def evaporation_constant_classical(self, T_s, T_inf, p_ambient, Y_infinity=0.0):
        """
        Calculate evaporation constant K using classical D² law.
        
        This uses the logarithmic form (standard in literature):
        K = (8 * rho_g * D_v / rho_l) * ln(1 + B_M)
        
        Note: Your document's Equation (3) uses B_M/(1+B_M) which is 
        approximately ln(1+B_M) for small B_M. The logarithmic form is
        more accurate and standard for D² law.
        
        Parameters:
        -----------
        T_s : float
            Droplet surface temperature in K
        T_inf : float
            Ambient gas temperature in K
        p_ambient : float
            Ambient pressure in Pa
        Y_infinity : float
            Far-field vapor mass fraction
            
        Returns:
        --------
        K : float
            Evaporation constant in m²/s
        """
        B_M = self.spalding_number(T_s, p_ambient, Y_infinity)
        
        # Classical D² law evaporation constant
        K = (8.0 * self.rho_g * self.D_v / self.rho_l) * np.log(1.0 + B_M)
        
        return K
    
    def evaporation_constant_convective(self, T_s, T_inf, p_ambient, 
                                       d_current, u_rel, Y_infinity=0.0):
        """
        Calculate evaporation constant with convective corrections.
        
        Uses Ranz-Marshall correlation for Sherwood number:
        Sh = 2 + 0.6 * Re^0.5 * Sc^0.33
        
        Then: K = (8 * rho_g * D_v / rho_l) * (Sh/2) * ln(1 + B_M)
        
        Parameters:
        -----------
        T_s : float
            Droplet surface temperature in K
        T_inf : float
            Ambient temperature in K
        p_ambient : float
            Ambient pressure in Pa
        d_current : float
            Current droplet diameter in m
        u_rel : float
            Relative velocity between droplet and gas in m/s
        Y_infinity : float
            Far-field vapor mass fraction
            
        Returns:
        --------
        K : float
            Evaporation constant with convection in m²/s
        """
        B_M = self.spalding_number(T_s, p_ambient, Y_infinity)
        
        # Reynolds number
        Re = self.rho_g * u_rel * d_current / self.mu_g
        
        # Ranz-Marshall correlation for Sherwood number
        Sh = 2.0 + 0.6 * Re**0.5 * self.Sc**(1.0/3.0)
        
        # Evaporation constant with convection
        K = (8.0 * self.rho_g * self.D_v / self.rho_l) * (Sh / 2.0) * np.log(1.0 + B_M)
        
        return K
    
    def droplet_temperature_equilibrium(self, T_inf, p_ambient):
        """
        Calculate equilibrium droplet temperature (wet-bulb temperature).
        
        For quasi-steady evaporation, the droplet reaches a temperature
        where heat conducted to the surface equals latent heat consumption.
        
        This is an iterative solution of the energy balance.
        
        Parameters:
        -----------
        T_inf : float
            Ambient gas temperature in K
        p_ambient : float
            Ambient pressure in Pa
            
        Returns:
        --------
        T_s : float
            Equilibrium surface temperature in K
        """
        
        def energy_balance(T_s):
            """
            Energy balance residual: heat_in - heat_out = 0
            """
            # Spalding number at this temperature
            B_M = self.spalding_number(T_s, p_ambient)
            
            # Heat transfer number (thermal analog of B_M)
            cp_avg = 0.5 * (self.cp_l + self.cp_g)
            B_T = cp_avg * (T_inf - T_s) / self.L_v
            
            # For equilibrium evaporation: B_M ≈ B_T
            # Residual to minimize
            residual = B_M - B_T
            
            return residual
        
        # Initial guess: somewhere between boiling point and ambient
        T_guess = min(self.T_boil, 0.7 * T_inf + 0.3 * self.T_boil)
        
        # Solve for equilibrium temperature
        T_s = fsolve(energy_balance, T_guess)[0]
        
        # Ensure physical bounds
        T_s = np.clip(T_s, 273.15, min(T_inf, self.T_boil + 20))
        
        return T_s
    
    def solve_quiescent(self, d0, T_inf, p_ambient=101325.0, 
                       Y_infinity=0.0, use_equilibrium_temp=True):
        """
        Solve D² law for a droplet in quiescent (still) gas.
        
        Integrates: d(d²)/dt = -K
        
        Parameters:
        -----------
        d0 : float
            Initial droplet diameter in meters
        T_inf : float
            Ambient gas temperature in K
        p_ambient : float
            Ambient pressure in Pa
        Y_infinity : float
            Far-field vapor mass fraction
        use_equilibrium_temp : bool
            If True, calculate equilibrium droplet temperature
            If False, use boiling point
            
        Returns:
        --------
        solution : dict
            Dictionary with time history and derived quantities
        """
        
        print(f"Solving D² Law (Quiescent Case)...")
        print(f"Initial diameter: d₀ = {d0*1e6:.1f} μm")
        print(f"Ambient temp:     T∞ = {T_inf:.1f} K")
        print(f"Ambient pressure: p  = {p_ambient/1e3:.1f} kPa\n")
        
        # Determine droplet surface temperature
        if use_equilibrium_temp:
            T_s = self.droplet_temperature_equilibrium(T_inf, p_ambient)
            print(f"Equilibrium droplet temperature: T_s = {T_s:.2f} K")
        else:
            T_s = self.T_boil
            print(f"Using boiling point: T_s = {T_s:.2f} K")
        
        # Calculate evaporation constant
        K = self.evaporation_constant_classical(T_s, T_inf, p_ambient, Y_infinity)
        print(f"Evaporation constant: K = {K*1e9:.4f} × 10⁻⁹ m²/s")
        
        # Calculate Spalding number
        B_M = self.spalding_number(T_s, p_ambient, Y_infinity)
        print(f"Spalding number: B_M = {B_M:.4f}")
        
        # Calculate droplet lifetime
        t_life = d0**2 / K
        print(f"Droplet lifetime: t_life = {t_life:.4f} s")
        
        # Time array for solution
        n_points = 1000
        t = np.linspace(0, t_life * 0.999, n_points)  # Stop just before complete evaporation
        
        # Analytical solution: d²(t) = d0² - K*t
        d_squared = d0**2 - K * t
        d = np.sqrt(d_squared)
        
        # Derived quantities
        surface_area = np.pi * d**2
        volume = (np.pi / 6.0) * d**3
        mass = self.rho_l * volume
        
        # Evaporation rate: dm/dt = -rho_l * pi * d * dd/dt
        # From d²(t) = d0² - K*t, we get dd/dt = -K/(2*d)
        dd_dt = -K / (2.0 * d)
        dm_dt = -self.rho_l * np.pi * d * dd_dt
        
        # Mass flux per unit area (your Equation 3 adapted to spherical geometry)
        mdot_per_area = dm_dt / surface_area
        
        # Normalized quantities
        d_normalized = d / d0
        d_squared_normalized = d_squared / d0**2
        
        print(f"\n✓ Solution complete!")
        print(f"  Final diameter: d_f = {d[-1]*1e6:.2f} μm")
        print(f"  Mass evaporated: {(mass[0] - mass[-1])*1e9:.2f} ng")
        print(f"  Average evap rate: {np.mean(dm_dt)*1e9:.2f} ng/s\n")
        
        # Package solution
        solution = {
            't': t,
            'd': d,
            'd_squared': d_squared,
            'd_normalized': d_normalized,
            'd_squared_normalized': d_squared_normalized,
            'surface_area': surface_area,
            'volume': volume,
            'mass': mass,
            'dd_dt': dd_dt,
            'dm_dt': dm_dt,
            'mdot_per_area': mdot_per_area,
            'K': K,
            'B_M': B_M,
            'T_s': T_s,
            'T_inf': T_inf,
            'd0': d0,
            't_life': t_life,
            'is_convective': False
        }
        
        return solution
    
    def solve_convective(self, d0, T_inf, u_rel, p_ambient=101325.0, 
                        Y_infinity=0.0, use_equilibrium_temp=True):
        """
        Solve D² law with convective effects (moving droplet).
        
        K varies with droplet size due to Reynolds number dependence.
        Must integrate ODE: d(d²)/dt = -K(d)
        
        Parameters:
        -----------
        d0 : float
            Initial droplet diameter in meters
        T_inf : float
            Ambient gas temperature in K
        u_rel : float
            Relative velocity in m/s
        p_ambient : float
            Ambient pressure in Pa
        Y_infinity : float
            Far-field vapor mass fraction
        use_equilibrium_temp : bool
            Use equilibrium temperature or boiling point
            
        Returns:
        --------
        solution : dict
            Dictionary with time history
        """
        
        print(f"Solving D² Law (Convective Case)...")
        print(f"Initial diameter: d₀ = {d0*1e6:.1f} μm")
        print(f"Ambient temp:     T∞ = {T_inf:.1f} K")
        print(f"Relative velocity: u = {u_rel:.2f} m/s\n")
        
        # Determine droplet temperature
        if use_equilibrium_temp:
            T_s = self.droplet_temperature_equilibrium(T_inf, p_ambient)
            print(f"Equilibrium droplet temperature: T_s = {T_s:.2f} K")
        else:
            T_s = self.T_boil
            print(f"Using boiling point: T_s = {T_s:.2f} K")
        
        # Initial evaporation constant
        K0 = self.evaporation_constant_convective(T_s, T_inf, p_ambient, 
                                                   d0, u_rel, Y_infinity)
        print(f"Initial evaporation constant: K₀ = {K0*1e9:.4f} × 10⁻⁹ m²/s")
        
        # Estimate lifetime (approximate, since K varies)
        t_life_approx = d0**2 / K0
        print(f"Approximate lifetime: t_life ≈ {t_life_approx:.4f} s")
        
        def rhs(t, y):
            """
            Right-hand side: d(d²)/dt = -K(d)
            y[0] = d²
            """
            d_squared = y[0]
            
            if d_squared <= 0:
                return [0.0]
            
            d_current = np.sqrt(d_squared)
            
            # Calculate K at current diameter
            K = self.evaporation_constant_convective(T_s, T_inf, p_ambient,
                                                     d_current, u_rel, Y_infinity)
            
            return [-K]
        
        # Solve ODE
        print("Integrating droplet evaporation...")
        t_span = [0, t_life_approx * 1.5]
        t_eval = np.linspace(0, t_life_approx * 0.999, 1000)
        
        sol = solve_ivp(rhs, t_span, [d0**2], t_eval=t_eval, 
                       method='RK45', rtol=1e-8, atol=1e-12,
                       events=lambda t, y: y[0] - 1e-20)  # Stop when d² → 0
        
        t = sol.t
        d_squared = sol.y[0, :]
        d = np.sqrt(d_squared)
        
        # Derived quantities
        surface_area = np.pi * d**2
        volume = (np.pi / 6.0) * d**3
        mass = self.rho_l * volume
        
        # Calculate K at each time point
        K_history = np.zeros_like(d)
        for i, d_i in enumerate(d):
            K_history[i] = self.evaporation_constant_convective(
                T_s, T_inf, p_ambient, d_i, u_rel, Y_infinity)
        
        dd_dt = -K_history / (2.0 * d)
        dm_dt = -self.rho_l * np.pi * d * dd_dt
        mdot_per_area = dm_dt / surface_area
        
        d_normalized = d / d0
        d_squared_normalized = d_squared / d0**2
        
        t_life = t[-1]
        
        print(f"\n✓ Solution complete!")
        print(f"  Actual lifetime: t_life = {t_life:.4f} s")
        print(f"  Final diameter: d_f = {d[-1]*1e6:.2f} μm")
        print(f"  Mass evaporated: {(mass[0] - mass[-1])*1e9:.2f} ng\n")
        
        solution = {
            't': t,
            'd': d,
            'd_squared': d_squared,
            'd_normalized': d_normalized,
            'd_squared_normalized': d_squared_normalized,
            'surface_area': surface_area,
            'volume': volume,
            'mass': mass,
            'dd_dt': dd_dt,
            'dm_dt': dm_dt,
            'mdot_per_area': mdot_per_area,
            'K': K_history,
            'B_M': self.spalding_number(T_s, p_ambient, Y_infinity),
            'T_s': T_s,
            'T_inf': T_inf,
            'd0': d0,
            't_life': t_life,
            'u_rel': u_rel,
            'is_convective': True
        }
        
        return solution
    
    def plot_results(self, solution, save_prefix='d2law'):
        """
        Create comprehensive plots for D² law validation.
        
        Generates all classic plots from droplet evaporation literature.
        """
        
        print("Generating plots...")
        
        t = solution['t']
        d = solution['d']
        d_squared = solution['d_squared']
        d_normalized = solution['d_normalized']
        d_squared_normalized = solution['d_squared_normalized']
        mass = solution['mass']
        dm_dt = solution['dm_dt']
        mdot_per_area = solution['mdot_per_area']
        
        # Handle K being either scalar or array
        is_convective = solution.get('is_convective', False)
        if is_convective:
            K_value = solution['K'][0]  # Use initial value for display
            K_label = f"K₀ = {K_value*1e9:.4f}×10⁻⁹ m²/s (varies)"
        else:
            K_value = solution['K']
            K_label = f"K = {K_value*1e9:.4f}×10⁻⁹ m²/s"
        
        fig = plt.figure(figsize=(18, 12))
        
        # Plot 1: d² vs t (THE classic D² law plot - must be linear!)
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(t, d_squared*1e12, 'b-', linewidth=3, label='d²(t)')
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('d² (μm²)', fontsize=12)
        ax1.set_title('D² Law: d²(t) = d₀² - Kt', fontweight='bold', fontsize=13)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Add linear fit to verify D² law
        coeffs = np.polyfit(t, d_squared*1e12, 1)
        K_fitted = -coeffs[0] * 1e-12
        ax1.plot(t, np.polyval(coeffs, t), 'r--', linewidth=2, 
                label=f'Linear fit: K = {K_fitted*1e9:.3f}×10⁻⁹ m²/s')
        ax1.legend(fontsize=10)
        
        # Plot 2: Diameter vs time
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(t, d*1e6, 'g-', linewidth=3)
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Diameter (μm)', fontsize=12)
        ax2.set_title('Droplet Diameter Evolution', fontweight='bold', fontsize=13)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Normalized d² vs normalized time
        ax3 = plt.subplot(3, 3, 3)
        t_normalized = t / solution['t_life']
        ax3.plot(t_normalized, d_squared_normalized, 'purple', linewidth=3)
        ax3.set_xlabel('Normalized Time (t/t_life)', fontsize=12)
        ax3.set_ylabel('Normalized d² (d²/d₀²)', fontsize=12)
        ax3.set_title('Universal D² Law Curve', fontweight='bold', fontsize=13)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Mass vs time
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(t, mass*1e9, 'orange', linewidth=3)
        ax4.set_xlabel('Time (s)', fontsize=12)
        ax4.set_ylabel('Droplet Mass (ng)', fontsize=12)
        ax4.set_title('Mass Evolution', fontweight='bold', fontsize=13)
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Evaporation rate dm/dt
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(t, -dm_dt*1e9, 'red', linewidth=3)  # Negative sign for positive rate
        ax5.set_xlabel('Time (s)', fontsize=12)
        ax5.set_ylabel('Evaporation Rate (ng/s)', fontsize=12)
        ax5.set_title('Mass Loss Rate: |dm/dt|', fontweight='bold', fontsize=13)
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Mass flux per unit area (your Equation 3)
        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(t, mdot_per_area*1e3, 'cyan', linewidth=3)
        ax6.set_xlabel('Time (s)', fontsize=12)
        ax6.set_ylabel('Mass Flux (g/(m²·s))', fontsize=12)
        ax6.set_title('Interfacial Mass Flux (Eq. 3)', fontweight='bold', fontsize=13)
        ax6.grid(True, alpha=0.3)
        
        # Plot 7: Surface area vs time
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(t, solution['surface_area']*1e12, 'brown', linewidth=3)
        ax7.set_xlabel('Time (s)', fontsize=12)
        ax7.set_ylabel('Surface Area (μm²)', fontsize=12)
        ax7.set_title('Droplet Surface Area', fontweight='bold', fontsize=13)
        ax7.grid(True, alpha=0.3)
        
        # Plot 8: Regression rate dd/dt
        ax8 = plt.subplot(3, 3, 8)
        ax8.plot(t, -solution['dd_dt']*1e6, 'magenta', linewidth=3)
        ax8.set_xlabel('Time (s)', fontsize=12)
        ax8.set_ylabel('Regression Rate (μm/s)', fontsize=12)
        ax8.set_title('Diameter Regression: |dd/dt|', fontweight='bold', fontsize=13)
        ax8.grid(True, alpha=0.3)
        
        # Plot 9: Summary info box
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        info_text = f"""
        D² LAW SUMMARY
        {'='*40}
        
        Fuel: {self.fuel_type.upper()}
        
        Initial Conditions:
          d₀ = {solution['d0']*1e6:.1f} μm
          T∞ = {solution['T_inf']:.1f} K
          T_s = {solution['T_s']:.1f} K
        
        Results:
          B_M = {solution['B_M']:.4f}
          {K_label}
          t_life = {solution['t_life']:.4f} s
          
        Mass:
          Initial: {mass[0]*1e9:.2f} ng
          Final: {mass[-1]*1e9:.2f} ng
          Evaporated: {(mass[0]-mass[-1])*1e9:.2f} ng
        
        Validation:
          Linear d² fit: {'PASS' if abs(K_fitted/K_value - 1) < 0.01 else 'CHECK'}
          R² > 0.999: {'PASS' if np.corrcoef(t, d_squared)[0,1]**2 > 0.999 else 'CHECK'}
        """
        
        ax9.text(0.1, 0.95, info_text, transform=ax9.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_all_plots.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_prefix}_all_plots.png")
        plt.show()
        
        # Create separate detailed D² plot
        self._plot_d_squared_detailed(solution, save_prefix)
    
    def _plot_d_squared_detailed(self, solution, save_prefix):
        """
        Create a detailed D² plot with annotations and validation.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        t = solution['t']
        d_squared = solution['d_squared']
        
        # Main data
        ax.plot(t, d_squared*1e12, 'bo-', linewidth=3, markersize=4, 
               markevery=50, label='Simulation')
        
        # Linear fit
        coeffs = np.polyfit(t, d_squared*1e12, 1)
        K_fitted = -coeffs[0] * 1e-12
        ax.plot(t, np.polyval(coeffs, t), 'r--', linewidth=2.5,
               label=f'Linear fit: K = {K_fitted*1e9:.4f}×10⁻⁹ m²/s')
        
        # Theoretical line
        d0_squared = solution['d0']**2
        is_convective = solution.get('is_convective', False)
        if is_convective:
            K_theory = solution['K'][0]
            theory_label = f'Theory (initial): K₀ = {K_theory*1e9:.4f}×10⁻⁹ m²/s'
        else:
            K_theory = solution['K']
            theory_label = f'Theory: K = {K_theory*1e9:.4f}×10⁻⁹ m²/s'
        
        d_squared_theory = (d0_squared - K_theory * t) * 1e12
        ax.plot(t, d_squared_theory, 'g:', linewidth=2.5, label=theory_label)
        
        ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
        ax.set_ylabel('d² (μm²)', fontsize=14, fontweight='bold')
        ax.set_title(f'D² Law Validation: {self.fuel_type.upper()} Droplet', 
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.4, linestyle='--')
        
        # Add text box with validation metrics
        R_squared = np.corrcoef(t, d_squared)[0, 1]**2
        error_percent = abs(K_fitted / K_theory - 1.0) * 100
        
        textstr = f'Validation Metrics:\n'
        textstr += f'R² = {R_squared:.6f}\n'
        textstr += f'K error = {error_percent:.2f}%\n'
        textstr += f'B_M = {solution["B_M"]:.4f}\n'
        textstr += f't_life = {solution["t_life"]:.4f} s'
        
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', horizontalalignment='right',
               bbox=props, family='monospace')
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_d_squared_validation.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_prefix}_d_squared_validation.png")
        plt.show()
    
    def create_animation(self, solution, save_prefix='d2law', fps=30):
        """
        Create animation showing shrinking droplet with live plots.
        
        Shows:
        - Visual representation of shrinking droplet
        - d² vs t plot building in real-time
        - Key parameters updating
        """
        
        print(f"\nCreating animation...")
        
        t = solution['t']
        d = solution['d']
        d_squared = solution['d_squared']
        
        fig = plt.figure(figsize=(16, 8))
        
        # Left panel: Droplet visualization
        ax1 = plt.subplot(1, 2, 1)
        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-1.2, 1.2)
        ax1.set_aspect('equal')
        ax1.axis('off')
        
        # Initial droplet circle
        circle = plt.Circle((0, 0), 1.0, color='blue', alpha=0.6)
        ax1.add_patch(circle)
        
        # Reference circle (initial size)
        ref_circle = plt.Circle((0, 0), 1.0, color='gray', 
                               fill=False, linestyle='--', linewidth=2)
        ax1.add_patch(ref_circle)
        
        # Text annotations
        title_text = ax1.text(0, 1.4, '', ha='center', fontsize=14, fontweight='bold')
        info_text = ax1.text(0, -1.4, '', ha='center', fontsize=11, family='monospace')
        
        # Right panel: d² vs t plot
        ax2 = plt.subplot(1, 2, 2)
        line_d2, = ax2.plot([], [], 'b-', linewidth=3)
        point_current, = ax2.plot([], [], 'ro', markersize=12)
        
        ax2.set_xlim(0, t[-1])
        ax2.set_ylim(0, d_squared[0]*1e12*1.1)
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('d² (μm²)', fontsize=12)
        ax2.set_title('D² Law: d²(t) = d₀² - Kt', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        def init():
            line_d2.set_data([], [])
            point_current.set_data([], [])
            return circle, line_d2, point_current, title_text, info_text
        
        def animate(frame):
            # Update droplet size
            d_current = d[frame]
            d_normalized = d_current / solution['d0']
            circle.set_radius(d_normalized)
            
            # Update title
            title_text.set_text(f'{self.fuel_type.upper()} Droplet Evaporation')
            
            # Update info
            info_str = f't = {t[frame]:.4f} s\n'
            info_str += f'd = {d_current*1e6:.2f} μm\n'
            info_str += f'd/d₀ = {d_normalized:.3f}'
            info_text.set_text(info_str)
            
            # Update d² plot
            line_d2.set_data(t[:frame+1], d_squared[:frame+1]*1e12)
            point_current.set_data([t[frame]], [d_squared[frame]*1e12])
            
            return circle, line_d2, point_current, title_text, info_text
        
        # Create animation
        n_frames = min(len(t), 300)  # Limit frames for reasonable file size
        frame_indices = np.linspace(0, len(t)-1, n_frames, dtype=int)
        
        anim = FuncAnimation(fig, animate, init_func=init,
                           frames=frame_indices, interval=1000/fps,
                           blit=True, repeat=True)
        
        # Save as GIF
        gif_filename = f'{save_prefix}_animation.gif'
        writer = PillowWriter(fps=fps)
        anim.save(gif_filename, writer=writer)
        print(f"✓ Saved: {gif_filename}")
        
        plt.close()
        print(f"✓ Animation complete!\n")


def main():
    """
    Main execution: Run D² law for multiple test cases.
    """
    
    print("\n" + "="*70)
    print(" D² LAW FOR DROPLET EVAPORATION ".center(70))
    print(" Classical Benchmark for Spray Combustion ".center(70))
    print(" Implementing Equations (3), (4), (6) from Document ".center(70))
    print("="*70 + "\n")
    
    # ========================================================================
    # CASE 1: Ethanol droplet in quiescent hot air
    # ========================================================================
    print("\n" + "─"*70)
    print(" CASE 1: ETHANOL - QUIESCENT ".center(70))
    print("─"*70)
    
    ethanol = DropletEvaporation(fuel_type='ethanol')
    
    sol_ethanol_quiescent = ethanol.solve_quiescent(
        d0=100e-6,          # 100 μm initial diameter
        T_inf=600.0,        # 600 K ambient (hot gas)
        p_ambient=101325,   # 1 atm
        Y_infinity=0.0,     # Dry air
        use_equilibrium_temp=True
    )
    
    ethanol.plot_results(sol_ethanol_quiescent, 
                        save_prefix='ethanol_quiescent')
    ethanol.create_animation(sol_ethanol_quiescent, 
                            save_prefix='ethanol_quiescent', fps=30)
    
    # ========================================================================
    # CASE 2: Ethanol droplet with convection
    # ========================================================================
    print("\n" + "─"*70)
    print(" CASE 2: ETHANOL - CONVECTIVE ".center(70))
    print("─"*70)
    
    sol_ethanol_convective = ethanol.solve_convective(
        d0=100e-6,
        T_inf=600.0,
        u_rel=5.0,          # 5 m/s relative velocity
        p_ambient=101325,
        Y_infinity=0.0,
        use_equilibrium_temp=True
    )
    
    ethanol.plot_results(sol_ethanol_convective,
                        save_prefix='ethanol_convective')
    ethanol.create_animation(sol_ethanol_convective,
                            save_prefix='ethanol_convective', fps=30)
    
    # ========================================================================
    # CASE 3: n-Heptane droplet in quiescent hot air
    # ========================================================================
    print("\n" + "─"*70)
    print(" CASE 3: N-HEPTANE - QUIESCENT ".center(70))
    print("─"*70)
    
    heptane = DropletEvaporation(fuel_type='n-heptane')
    
    sol_heptane_quiescent = heptane.solve_quiescent(
        d0=100e-6,
        T_inf=700.0,        # Higher temp for n-heptane
        p_ambient=101325,
        Y_infinity=0.0,
        use_equilibrium_temp=True
    )
    
    heptane.plot_results(sol_heptane_quiescent,
                        save_prefix='heptane_quiescent')
    heptane.create_animation(sol_heptane_quiescent,
                            save_prefix='heptane_quiescent', fps=30)
    
    # ========================================================================
    # CASE 4: n-Heptane with convection
    # ========================================================================
    print("\n" + "─"*70)
    print(" CASE 4: N-HEPTANE - CONVECTIVE ".center(70))
    print("─"*70)
    
    sol_heptane_convective = heptane.solve_convective(
        d0=100e-6,
        T_inf=700.0,
        u_rel=10.0,         # 10 m/s relative velocity
        p_ambient=101325,
        Y_infinity=0.0,
        use_equilibrium_temp=True
    )
    
    heptane.plot_results(sol_heptane_convective,
                        save_prefix='heptane_convective')
    heptane.create_animation(sol_heptane_convective,
                            save_prefix='heptane_convective', fps=30)
    
    # ========================================================================
    # Summary comparison
    # ========================================================================
    print("\n" + "="*70)
    print(" SUMMARY COMPARISON ".center(70))
    print("="*70)
    
    cases = [
        ('Ethanol Quiescent', sol_ethanol_quiescent),
        ('Ethanol Convective', sol_ethanol_convective),
        ('n-Heptane Quiescent', sol_heptane_quiescent),
        ('n-Heptane Convective', sol_heptane_convective)
    ]
    
    print(f"\n{'Case':<25} {'K (×10⁻⁹ m²/s)':<18} {'B_M':<10} {'t_life (s)':<12}")
    print("─"*70)
    for name, sol in cases:
        K_val = sol['K'] if isinstance(sol['K'], float) else sol['K'][0]
        print(f"{name:<25} {K_val*1e9:<18.4f} {sol['B_M']:<10.4f} {sol['t_life']:<12.4f}")
    
    print("\n" + "="*70)
    print(" ALL SIMULATIONS COMPLETE ".center(70))
    print("="*70)
    print("\nGenerated files:")
    print("  Ethanol Quiescent:")
    print("    • ethanol_quiescent_all_plots.png")
    print("    • ethanol_quiescent_d_squared_validation.png")
    print("    • ethanol_quiescent_animation.gif")
    print("  Ethanol Convective:")
    print("    • ethanol_convective_all_plots.png")
    print("    • ethanol_convective_d_squared_validation.png")
    print("    • ethanol_convective_animation.gif")
    print("  n-Heptane Quiescent:")
    print("    • heptane_quiescent_all_plots.png")
    print("    • heptane_quiescent_d_squared_validation.png")
    print("    • heptane_quiescent_animation.gif")
    print("  n-Heptane Convective:")
    print("    • heptane_convective_all_plots.png")
    print("    • heptane_convective_d_squared_validation.png")
    print("    • heptane_convective_animation.gif")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
