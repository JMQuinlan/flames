"""
Classic Analytical D² Law for Water Droplet Evaporation in Air
================================================================

Pure analytical solution - no numerical integration.

Textbook formulas:
    d²(t) = d₀² - K·t
    K = (8·ρ_g·D_v/ρ_l)·ln(1 + B_M)
    dd/dt = -K/(2d)

Fuel: Water
Ambient: Air
"""

import numpy as np
import matplotlib.pyplot as plt

# Plotting setup
plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2.5


class WaterDropletD2Law:
    """
    Analytical D² law for water droplet evaporation in air.
    """
    
    def __init__(self):
        """Initialize water and air properties."""
        
        # Water properties
        self.W_v = 18.015e-3      # kg/mol (molecular weight)
        self.rho_l = 997.0        # kg/m³ (liquid density at 25°C)
        self.L_v = 2257e3         # J/kg (latent heat of vaporization)
        self.T_boil = 373.15      # K (boiling point at 1 atm)
        
        # Antoine coefficients for water (log10(p[Pa]) = A - B/(C + T[K]))
        self.antoine_A = 5.40221
        self.antoine_B = 1838.675
        self.antoine_C = -31.737
        
        # Diffusion coefficient of water vapor in air
        self.D_v = 2.6e-5         # m²/s (at ~300K)
        
        # Air properties
        self.W_g = 28.97e-3       # kg/mol (molecular weight)
        self.rho_g = 1.184        # kg/m³ (density at 25°C)
        
        print("\n" + "="*70)
        print(" WATER DROPLET EVAPORATION - ANALYTICAL D² LAW ".center(70))
        print("="*70)
        print(f"Water Properties:")
        print(f"  ρ_l = {self.rho_l:.1f} kg/m³")
        print(f"  W_v = {self.W_v*1e3:.3f} g/mol")
        print(f"  L_v = {self.L_v/1e3:.0f} kJ/kg")
        print(f"  T_boil = {self.T_boil:.2f} K")
        print(f"  D_v = {self.D_v*1e6:.1f} × 10⁻⁶ m²/s")
        print(f"\nAir Properties:")
        print(f"  ρ_g = {self.rho_g:.3f} kg/m³")
        print(f"  W_g = {self.W_g*1e3:.2f} g/mol")
        print("="*70 + "\n")
    
    def saturation_pressure(self, T):
        """
        Antoine equation for water vapor saturation pressure.
        
        Parameters:
        -----------
        T : float
            Temperature in K
            
        Returns:
        --------
        p_sat : float
            Saturation pressure in Pa
        """
        log10_p = self.antoine_A - self.antoine_B / (self.antoine_C + T)
        return 10**log10_p
    
    def vapor_mass_fraction_surface(self, T_s, p_ambient):
        """
        Surface vapor mass fraction.
        
        Y_vs = (W_v * x_vs) / (W_v * x_vs + W_g * (1 - x_vs))
        where x_vs = p_sat(T_s) / p
        """
        p_sat = self.saturation_pressure(T_s)
        x_vs = min(p_sat / p_ambient, 1.0)
        
        numerator = self.W_v * x_vs
        denominator = self.W_v * x_vs + self.W_g * (1.0 - x_vs)
        Y_vs = numerator / denominator
        
        return Y_vs
    
    def spalding_number(self, T_s, p_ambient, Y_infinity=0.0):
        """
        Spalding mass transfer number.
        
        B_M = (Y_vs - Y_inf) / (1 - Y_vs)
        """
        Y_vs = self.vapor_mass_fraction_surface(T_s, p_ambient)
        
        denominator = 1.0 - Y_vs
        if denominator < 1e-10:
            denominator = 1e-10
        
        B_M = (Y_vs - Y_infinity) / denominator
        
        return B_M
    
    def evaporation_constant(self, T_s, p_ambient, Y_infinity=0.0):
        """
        Classic D² law evaporation constant.
        
        K = (8 * ρ_g * D_v / ρ_l) * ln(1 + B_M)
        
        Returns:
        --------
        K : float
            Evaporation constant in m²/s
        """
        B_M = self.spalding_number(T_s, p_ambient, Y_infinity)
        K = (8.0 * self.rho_g * self.D_v / self.rho_l) * np.log(1.0 + B_M)
        return K
    
    def solve_analytical(self, d0, T_s, T_inf, p_ambient=101325.0, 
                        Y_infinity=0.0, n_points=1000):
        """
        Solve using pure analytical formula: d²(t) = d₀² - K·t
        
        Parameters:
        -----------
        d0 : float
            Initial diameter in meters
        T_s : float
            Droplet surface temperature in K
        T_inf : float
            Ambient temperature in K
        p_ambient : float
            Ambient pressure in Pa
        Y_infinity : float
            Far-field vapor mass fraction
        n_points : int
            Number of time points
            
        Returns:
        --------
        solution : dict
            Time, diameter, d², and regression rate
        """
        
        print(f"Solving Analytical D² Law...")
        print(f"Initial diameter: d₀ = {d0*1e6:.1f} μm")
        print(f"Surface temp:     T_s = {T_s:.1f} K")
        print(f"Ambient temp:     T∞ = {T_inf:.1f} K")
        print(f"Ambient pressure: p = {p_ambient/1e3:.1f} kPa\n")
        
        # Calculate Spalding number
        B_M = self.spalding_number(T_s, p_ambient, Y_infinity)
        print(f"Spalding number: B_M = {B_M:.6f}")
        
        # Calculate evaporation constant
        K = self.evaporation_constant(T_s, p_ambient, Y_infinity)
        print(f"Evaporation constant: K = {K*1e9:.6f} × 10⁻⁹ m²/s")
        
        # Calculate droplet lifetime
        t_life = d0**2 / K
        print(f"Droplet lifetime: t_life = {t_life:.4f} s")
        
        # Create time array
        t = np.linspace(0, t_life * 0.999, n_points)
        
        # ANALYTICAL SOLUTION
        d_squared = d0**2 - K * t
        d = np.sqrt(d_squared)
        
        # Regression rate: dd/dt = -K/(2d)
        dd_dt = -K / (2.0 * d)
        
        # Surface vapor mass fraction
        Y_vs = self.vapor_mass_fraction_surface(T_s, p_ambient)
        
        print(f"Surface vapor mass fraction: Y_vs = {Y_vs:.6f}")
        print(f"\n✓ Analytical solution complete!\n")
        
        # Package solution
        solution = {
            't': t,
            'd': d,
            'd_squared': d_squared,
            'dd_dt': dd_dt,
            'K': K,
            'B_M': B_M,
            'Y_vs': Y_vs,
            'T_s': T_s,
            'T_inf': T_inf,
            'd0': d0,
            't_life': t_life,
            'p_ambient': p_ambient
        }
        
        return solution
    
    def plot_d_squared(self, solution):
        """
        Plot 1: d² vs time (classic D² law plot)
        """
        t = solution['t']
        d_squared = solution['d_squared']
        K = solution['K']
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Plot analytical solution
        ax.plot(t, d_squared*1e12, 'b-', linewidth=3, label='Analytical: d²(t) = d₀² - Kt')
        
        # Linear fit to verify linearity
        coeffs = np.polyfit(t, d_squared*1e12, 1)
        K_fitted = -coeffs[0] * 1e-12
        ax.plot(t, np.polyval(coeffs, t), 'r--', linewidth=2, alpha=0.7,
               label=f'Linear fit: K = {K_fitted*1e9:.6f}×10⁻⁹ m²/s')
        
        ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
        ax.set_ylabel('d² (μm²)', fontsize=14, fontweight='bold')
        ax.set_title('Classic D² Law: Water Droplet Evaporation in Air', 
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.4, linestyle='--')
        
        plt.tight_layout()
        plt.savefig('water_d2_law.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: water_d2_law.png")
        plt.show()
    
    def plot_regression_rate(self, solution):
        """
        Plot 2: Diameter regression rate vs time
        """
        t = solution['t']
        dd_dt = solution['dd_dt']
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Plot regression rate (absolute value)
        ax.plot(t, -dd_dt*1e6, 'r-', linewidth=3, label='|dd/dt| = K/(2d)')
        
        ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Regression Rate (μm/s)', fontsize=14, fontweight='bold')
        ax.set_title('Diameter Regression Rate: Water Droplet in Air', 
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.4, linestyle='--')
        
        plt.tight_layout()
        plt.savefig('water_regression_rate.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: water_regression_rate.png")
        plt.show()


def main():
    """
    Run analytical D² law for water droplet in air.
    """
    
    # Initialize water droplet model
    water = WaterDropletD2Law()
    
    # Solve for water droplet at ambient conditions
    # Using surface temperature = boiling point (quasi-steady assumption)
    solution = water.solve_analytical(
        d0=100e-6,          # 100 μm initial diameter
        T_s=373.15,         # Surface temp = boiling point (K)
        T_inf=400.0,        # Ambient temperature (K)
        p_ambient=101325,   # 1 atm (Pa)
        Y_infinity=0.0      # Dry air
    )
    
    # Generate plots
    print("Generating plots...\n")
    water.plot_d_squared(solution)
    water.plot_regression_rate(solution)
    
    print("\n" + "="*70)
    print(" COMPLETE ".center(70))
    print("="*70)
    print("\nGenerated files:")
    print("  • water_d2_law.png")
    print("  • water_regression_rate.png")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
