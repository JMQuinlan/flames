"""Size the MarmPaper tests with Keller-Miksis + Marmottant (2 um coated bubble)."""
import os
import sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "reference"))
from marmottant_rpe_km import Shell, Liquid, Gas, solve_km, natural_frequency

R0 = 2.0e-6
sh = Shell(chi=1.0, R_buck=1.96e-6, sigma_break=0.073, sigma_water=0.0728, kappa_s=7.2e-9)
c_l = np.sqrt(2.35 * (1e5 + 1e9) / 1000.0)
def liq(p): return Liquid(rho=1000.0, mu=1.0e-3, c=c_l, p_inf=p)
p0 = 1.0e5
p_eq = p0 + 2 * sh.sigma(R0) / R0
w0 = natural_frequency(R0, sh, liq(p0), Gas(p_g0=p_eq))
f0 = w0 / 2 / np.pi
print(f"c_l {c_l:.1f}  sigma(R0) {sh.sigma(R0):.4f}  R_rup/R0 {sh.R_rupture/R0:.4f}  p_eq {p_eq:.4e}")
print(f"f0 {f0/1e6:.3f} MHz  T0 {1/f0*1e9:.1f} ns  Minnaert {np.sqrt(3*1.4*p0/1000)/(2*np.pi*R0)/1e6:.3f} MHz")
d_sh = 4 * sh.kappa_s / (1000 * w0 * R0**3); d_mu = 4e-3 / (1000 * w0 * R0**2)
print(f"damping ratio-ish: shell {d_sh:.3f}  viscous {d_mu:.3f}")

def run(p_inf, p_b, tend, drive=None, n=40001):
    t = np.linspace(0, tend, n)
    tt, R, Rd = solve_km(R0, 0.0, (0, tend), sh, liq(p_inf), Gas(p_g0=p_b), drive=drive, t_eval=t)
    return tt, R / R0

# 1 free oscillation: Light_Mili pressures
t, R = run(1e5, 5.8307e4, 30 / f0)
print(f"\n[free osc] p_inf 1e5 p_b 5.83e4: Rmin {R.min():.3f} Rmax {R.max():.3f}  R_end {R[-1]:.4f}")
# 2 free collapse: scan p_inf at p_b = 5.8307e4 (Light_Mili gas)
print("\n[free collapse] p_b 5.83e4, scan p_inf:")
for pinf in [3e5, 5e5, 1e6, 2e6]:
    t, R = run(pinf, 5.8307e4, 3 * np.sqrt(1000 / pinf) * R0 * 0.915 * 4)
    i = np.argmin(R); tc = 0.915 * R0 * np.sqrt(1000 / pinf)
    print(f"  p_inf {pinf:.0e}: Rmin {R.min():.3f} at t {t[i]*1e9:.1f} ns (tau_c {tc*1e9:.1f} ns)  Rmax_rebound {R[i:].max():.3f}")
# 3 forced linear: f = f0/pi, start at equilibrium
print("\n[forced linear] f = f0/pi, from equilibrium:")
for A in [5e3, 1e4, 2e4]:
    t, R = run(p0, p_eq, 20 * np.pi / f0, drive=(A, w0 / np.pi))
    late = R[len(R)//2:]
    print(f"  A {A:.0e}: R range late {late.min():.4f}-{late.max():.4f}  (elastic band {sh.R_buck/R0:.3f}-{sh.R_rupture/R0:.3f})")
# 4 forced nonlinear: scan f/f0 and A; subharmonic content and Rmin
print("\n[forced nonlinear] 40 drive cycles, analyse last 20:")
for fr in [0.8, 0.9, 1.0, 1.5, 2.0]:
    for A in [3e4, 5e4, 8e4, 1.2e5]:
        wd = fr * w0; T = 2 * np.pi / wd; N = 40
        t, R = run(p0, p_eq, N * T, drive=(A, wd), n=N * 400 + 1)
        m = t > 20 * T
        x = R[m] - R[m].mean(); F = np.abs(np.fft.rfft(x * np.hanning(len(x)))); fq = np.fft.rfftfreq(len(x), t[1] - t[0])
        fd = wd / 2 / np.pi
        def amp(f): return F[np.argmin(np.abs(fq - f))]
        sub = amp(fd / 2) / max(amp(fd), 1e-30)
        ult = amp(1.5 * fd) / max(amp(fd), 1e-30)
        print(f"  f/f0 {fr:.1f} A {A:.0e}: Rmin {R.min():.3f} Rmax {R.max():.3f}  sub(f/2)/f {sub:.3f}  (3f/2)/f {ult:.3f}  2f/f {amp(2*fd)/max(amp(fd),1e-30):.3f}")
