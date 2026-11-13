#!/usr/bin/env python3
"""
Shock / small fuel-column decks (n-dodecane, n-pentane) for the AIAA SciTech 2027 evaporation study, and their
INCLINE SLURM files.  Same set-up and analysis as the 4.8 mm water column (../ShockWaterColumn/make_decks.py):
half domain (symmetry plane y = 0), x in [-4D, 8.8D], y in [0, 8D], base D/dx = 25, AMR on grad(eta) and shocks,
HLLC / minmod / SSP-RK3, surface tension on, inviscid (VISCOUS = False).  Re_D is only ~250-3000 at D = 10 um, but
the viscous runs (air: Sutherland at the post-shock T, liquid: NIST) blew up after ~160 steps at Ms 10 (2026-11-04,
with and without vaporization; diag/ runs), so the study is inviscid like the water column until that is fixed, run to t* = 2.2.  The column diameter D is smaller so
that evaporation is visible on the breakup time scale.

Vaporization uses the Abramzon-Sirignano film model (vaporization.model = 3) with
p_sat from an Antoine fit, film thicknesses from Churchill & Bernstein (1977) at the post-shock state
(1/3-rule film temperature), vapour diffusivity in air from Fuller, Schettler & Giddings (1966)
(Poling, Prausnitz & O'Connell 2001, eq. 11-4.4), air mu and k from Sutherland's law.

  python3 make_fuel_decks.py     (writes input_* here and the .sh files in TryonSlurmFiles/AIAA2027/FuelDroplet)
"""
import os, math

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SLURM_DIR = os.path.join(REPO, "TryonSlurmFiles", "AIAA2027", "FuelDroplet")
INCLINE_REPO = "/home/ttryon/flames"
INCLINE_OUT = "/mmfs1/home/ttryon/flames/bin/tests/AIAA2027/FuelDroplet"

# ---- fluids --------------------------------------------------------------------------------
# Every entry needs a source; FLUIDS is filled from the papers (see SOURCES.md next to this script).
FLUIDS = {}
exec(open(os.path.join(HERE, "fluids.py")).read())      # defines FLUIDS[name] = dict(...)

# ---- air -----------------------------------------------------------------------------------
g, p1, r1, cv_air = 1.4, 101325.0, 1.225, 717.86
cp_air = g * cv_air
R_air = cp_air - cv_air
T1 = p1 / (r1 * R_air)
a1 = math.sqrt(g * p1 / r1)
W_AIR, V_AIR = 28.965, 19.7                     # g/mol; Fuller diffusion volume of air


def post_shock(Ms):
    p2 = p1 * (1 + 2 * g / (g + 1) * (Ms * Ms - 1))
    r2 = r1 * (g + 1) * Ms * Ms / ((g - 1) * Ms * Ms + 2)
    u2 = a1 * 2 / (g + 1) * (Ms - 1 / Ms)
    return p2, r2, u2, p2 / (r2 * R_air)


def mu_air(T): return 1.716e-5 * (T / 273.15) ** 1.5 * (273.15 + 110.4) / (T + 110.4)     # Sutherland
def k_air(T):  return 0.0241 * (T / 273.15) ** 1.5 * (273.15 + 194.0) / (T + 194.0)


def fuller_D(T, p, W, V):
    """Binary diffusivity of vapour (molar mass W, diffusion volume V) in air [m^2/s]."""
    Wab = 2.0 / (1.0 / W + 1.0 / W_AIR)
    return 1.43e-3 * T ** 1.75 / ((p / 1e5) * math.sqrt(Wab) * (V ** (1 / 3) + V_AIR ** (1 / 3)) ** 2) * 1e-4


def churchill_bernstein(Re, Pr):
    return 0.3 + 0.62 * Re ** 0.5 * Pr ** (1 / 3) / (1 + (0.4 / Pr) ** (2 / 3)) ** 0.25 * (1 + (Re / 282000.0) ** 0.625) ** 0.8


def film(fl, Ms, D):
    """Film-model inputs at the post-shock state: Dv, delta, delta_T, k_gas, T_gas_max (stagnation T)."""
    p2, r2, u2, T2 = post_shock(Ms)
    Ts = fl["T0"]
    Tr = Ts + (T2 - Ts) / 3.0
    mu, k = mu_air(Tr), k_air(Tr)
    Dv = fuller_D(Tr, p2, fl["W"], fl["V_fuller"])
    rho_f = p2 / (R_air * Tr)
    Re = r2 * u2 * D / mu
    Sc, Pr = mu / (rho_f * Dv), cp_air * mu / k
    Sh, Nu = churchill_bernstein(Re, Sc), churchill_bernstein(Re, Pr)
    M2 = u2 / math.sqrt(g * p2 / r2)
    T0 = T2 * (1 + 0.5 * (g - 1) * M2 * M2)
    return dict(T2=T2, p2=p2, r2=r2, u2=u2, Re=Re, Sc=Sc, Pr=Pr, Sh=Sh, Nu=Nu, Dv=Dv, k=k,
                delta=D / Sh, delta_T=D / Nu, T0=T0, M2=M2)


# ---- grid / run ----------------------------------------------------------------------------
VISCOUS = False
# Geometry / refinement layouts (lengths in D).  SHORT is the original breakup study (t* = 2.2); its decks are unchanged.
# LONG (2026-11-12) follows the fragments downstream: base D/dx 12.5 so the long far field is cheap, shocks capped at
# level 1 (D/dx 25; p_refinement_max_level), only the interface reaches the finest level, run to t* = 8.  In the
# t* = 2.2 runs the liquid centroid moved ~0.65 (t/t*)^2 D, i.e. ~42 D by t* = 8; the domain runs to 60 D.
SHORT = dict(tag="", base_N=25, xlo=-4.0, xhi=8.8, yhi=8.0, t_end=2.2, nbuf=4, p_max_level=-1)
LONG = dict(tag="_long", base_N=12.5, xlo=-4.0, xhi=60.0, yhi=9.6, t_end=8.0, nbuf=3, p_max_level=1)
T_END_TSTAR = SHORT["t_end"]
N_FRAMES = 200
P_REFINE_FRAC = 0.05
RATE_PER_RANK = 1.13e6 / 12

DECK = open(os.path.join(HERE, "deck_template.txt")).read()
SLURM = """#!/bin/bash

#SBATCH --job-name=FD_{name}
#SBATCH -o /mmfs1/home/ttryon/FLAMES_out/flame_%j_stdout
#SBATCH -e /mmfs1/home/ttryon/FLAMES_out/err_flame_%j_stdout
#SBATCH -N 1
#SBATCH --ntasks-per-node={ntasks}
#SBATCH --partition=compute-long
#SBATCH -t {walltime}

# AIAA SciTech 2027: Mach {Ms} shock on a {D_um:g} um {fluid} column, finest D/dx = {N}{vap_note}.
# Deck generated by tests/AIAA2027/FuelDroplet/make_fuel_decks.py.  Output: {plot_file}

module purge
module load gnu9 mpich

mkdir -p {outdir}
srun --mpi=pmi2 {repo}/bin/hydro2-2d-g++ {repo}/tests/AIAA2027/FuelDroplet/input_{name}
"""


def cost_hours(fl, Ms, D, lev, vap, ntasks, geo=SHORT):
    p2, r2, u2, T2 = post_shock(Ms)
    tstar = D / u2 * math.sqrt(fl["rho"] / r2)
    dxf = D / geo["base_N"] / 2 ** lev
    smax = max(fl["c"], u2 + math.sqrt(g * p2 / r2))
    steps = geo["t_end"] * tstar / (0.3 * dxf / smax)
    nx0, ny0 = grid(geo)
    # fraction of each finer level that is refined (SHORT: measured ~0.35; LONG: interface only, rough guess 0.08)
    fr = 0.35 if geo is SHORT else 0.08
    cells = nx0 * ny0 * sum((4 ** l) * (1.0 if l == 0 else fr) for l in range(lev + 1))
    return steps * cells / (RATE_PER_RANK * ntasks * 0.7) / 3600 * (2.5 if vap else 1.0)


def grid(geo):
    nx0 = round((geo["xhi"] - geo["xlo"]) * geo["base_N"]); ny0 = round(geo["yhi"] * geo["base_N"])
    assert nx0 % 8 == 0 and ny0 % 8 == 0, (nx0, ny0)      # blocking_factor 8
    return nx0, ny0


def make(fluid, D, Ms, lev, vap, geo=SHORT):
    fl = FLUIDS[fluid]
    N = round(geo["base_N"] * 2 ** lev)
    name = f"{fl['short']}_D{D * 1e6:g}um_Ms{Ms:g}_N{N:03d}" + ("_vap" if vap else "") + geo["tag"]
    f = film(fl, Ms, D)
    tstar = D / f["u2"] * math.sqrt(fl["rho"] / f["r2"])
    stop = geo["t_end"] * tstar
    nx0, ny0 = grid(geo)
    dx0 = D / geo["base_N"]
    dxf = dx0 / 2 ** lev
    eps = 0.75 * dxf
    R0 = D / 2
    xs = -R0 - 0.25 * D
    xlo, ylo = geo["xlo"] * D, 0.0
    ntasks = 64 if (lev <= 2 and geo is SHORT) else 128
    hrs = cost_hours(fl, Ms, D, lev, vap, ntasks, geo)
    wall = "%d:00:00" % min(100, max(4, math.ceil(2.0 * hrs)))
    plot_file = f"{INCLINE_OUT}/output_{name}"
    vapblock = ""
    if vap:
        vapblock = (f"vaporization.model = 3\n"
                    f"vaporization.film_thickness = {f['delta']:.5e}\n"
                    f"vaporization.film_thickness_T = {f['delta_T']:.5e}\n"
                    f"vaporization.k_gas = {f['k']:.5g}\n"
                    f"vaporization.cp_vapor = {fl['cp_vapor']:g}\n"
                    f"vaporization.W_vapor = {fl['W']:g}\n"
                    f"vaporization.W_gas = {W_AIR:g}\n"
                    f"vaporization.psat = antoine\n"
                    f"vaporization.antoine_A = {fl['antoine'][0]:g}\n"
                    f"vaporization.antoine_B = {fl['antoine'][1]:g}\n"
                    f"vaporization.antoine_C = {fl['antoine'][2]:g}\n"
                    f"vaporization.T_gas_max = {f['T0']:.1f}\n")
    deck = DECK.format(
        name=name, fluid=fl["name"], D=D, D_um=D * 1e6, N=N, lev=lev, eps=eps, eps_um=eps * 1e6, Ms=Ms,
        vap=vap, vap_word="ON (film model)" if vap else "off", r1=r1, p1=p1, T1=T1,
        r2=f["r2"], u2=f["u2"], p2=f["p2"], T2=f["T2"], M2=f["M2"], Re=f["Re"],
        bow=" (supersonic: a bow shock forms ahead of the column)" if f["M2"] > 1 else "",
        tstar_us=tstar * 1e6, stop="%.6e" % stop, stop_us=stop * 1e6, stop_tstar=geo["t_end"],
        xs=xs, xs_um=xs * 1e6, t_hit_us=(-R0 - xs) / (Ms * a1) * 1e6,
        plot_file=plot_file, plot_dt="%.6e" % (stop / N_FRAMES), mgs=64 if lev >= 4 else 128,
        nx0=nx0, ny0=ny0, xlo=xlo, ylo=ylo, xhi=xlo + nx0 * dx0, yhi=ylo + ny0 * dx0, R0=R0,
        base_N=geo["base_N"], xlo_D=geo["xlo"], xhi_D=geo["xhi"], yhi_D=geo["yhi"], nbuf=geo["nbuf"],
        long_note="" if geo is SHORT else "\n#   Long domain: follows the fragments downstream (LONG in make_fuel_decks.py).",
        refine_note=("# grad(p)*2*dr > 5% of the incident shock jump (p2 - p1): incident, reflected, transmitted and bow\n"
                     "# shocks are refined to the finest level; weak acoustic waves are not.") if geo["p_max_level"] < 0 else
                    (f"# Interface (grad eta) refined to the finest level.  Shocks (grad(p)*2*dr > 5% of p2 - p1) are refined\n"
                     f"# only up to level {geo['p_max_level']} (D/dx = {geo['base_N'] * 2 ** geo['p_max_level']:g}): they set the post-shock state but need not be sharp\n"
                     f"# at the finest level, which in the t* = 2.2 runs held ~70% of the work."),
        prefine_extra="" if geo["p_max_level"] < 0 else f"p_refinement_max_level = {geo['p_max_level']}\n",
        gamma_l=fl["gamma"], pinf_l=fl["pinf"], cv_l=fl["cv"], cp_l=fl["gamma"] * fl["cv"], rho_l=fl["rho"],
        T_l=fl["T0"], sigma=fl["sigma"], Lv=fl["Lv"] if vap else 0.0, Dv=f["Dv"],
        mu_g=mu_air(f['T2']) if VISCOUS else 0.0, mu_l=fl['mu'] if VISCOUS else 0.0,
        visc_word='viscous' if VISCOUS else 'inviscid', vapblock=vapblock, pcrit=P_REFINE_FRAC * (f["p2"] - p1), src=fl["src"])
    with open(os.path.join(HERE, f"input_{name}"), "w") as fh: fh.write(deck)
    sh = SLURM.format(name=name, ntasks=ntasks, walltime=wall, N=N, Ms=Ms, D_um=D * 1e6, fluid=fl["name"],
                      plot_file=plot_file, outdir=INCLINE_OUT, repo=INCLINE_REPO,
                      vap_note=", film-model vaporization on" if vap else "")
    p = os.path.join(SLURM_DIR, f"RunFD_{name}.sh")
    with open(p, "w") as fh: fh.write(sh)
    os.chmod(p, 0o755)
    print(f"{name:34s} t* {tstar * 1e6:7.3f} us stop {stop * 1e6:7.2f} us  T2 {f['T2']:6.0f} K  Re {f['Re']:8.0f}  "
          f"Dv {f['Dv']:.3e}  delta {f['delta'] * 1e6:6.3f} um  dT {f['delta_T'] * 1e6:6.3f} um  "
          f"est {hrs:5.1f} h on {ntasks} (wall {wall})")
    return name, f, tstar


CASES = []      # (fluid, D, Ms, max_level, vap); filled in cases.py
exec(open(os.path.join(HERE, "cases.py")).read())


def main():
    os.makedirs(SLURM_DIR, exist_ok=True)
    rows = [make(*c) for c in CASES]
    with open(os.path.join(HERE, "film_params.txt"), "w") as fh:
        fh.write("# name  t*[us]  T2  p2  u2  Re  Sc  Pr  Sh  Nu  Dv  k  delta  delta_T  T0   (make_fuel_decks.py)\n")
        for name, f, ts in rows:
            fh.write(f"{name} {ts * 1e6:.4f} {f['T2']:.1f} {f['p2']:.1f} {f['u2']:.2f} {f['Re']:.4g} {f['Sc']:.3f} "
                     f"{f['Pr']:.3f} {f['Sh']:.2f} {f['Nu']:.2f} {f['Dv']:.4e} {f['k']:.4g} {f['delta']:.5e} "
                     f"{f['delta_T']:.5e} {f['T0']:.1f}\n")


if __name__ == "__main__":
    main()
