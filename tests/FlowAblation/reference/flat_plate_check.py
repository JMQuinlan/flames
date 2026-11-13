#!/usr/bin/env python3
"""
Laminar boundary-layer heat flux on a flat plate at Mach 2 (tests/FlowAblation/input_2D_FlatPlate).

  python3 flat_plate_check.py <run_dir> [--outdir DIR] [--twall TW | --adiabatic]

--twall TW    wall temperature of the run if not the deck's 1.0 (e.g. 2.5: hot plate heating the gas, q_w < 0)
--adiabatic   run without an isothermal wall (ablation.on = 0, no solid.T_wall): the check is then the RECOVERY
              temperature, wall temperature against T_aw = T_e (1 + sqrt(Pr) (gamma - 1)/2 M^2), and the skin friction.

Reference (compressible laminar flat plate, Eckert reference temperature):
    q_w(x) = St rho_e U cp (T_aw - T_w),   St = 0.332 sqrt(C*) Pr^(-2/3)/sqrt(Re_x),
    T_aw = T_e (1 + sqrt(Pr) (gamma - 1)/2 M^2),  T* = T_e + 0.5 (T_w - T_e) + 0.22 (T_aw - T_e),
    C* = rho* mu*/(rho_e mu_e) = T_e/T*   (constant viscosity).
Run: `ablation_q` on the wall-adjacent fluid cells of the last plot files (time average over the
last third of the run); also the fitted exponent n of q_w ~ x^n over 0.2 < x < 0.9 (theory -1/2)
and the skin friction c_f = 0.664 sqrt(C*)/sqrt(Re_x) from mu du/dy at the wall.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt
yt.funcs.mylog.setLevel(40)
GAM, CP, MU, PR, RHO, TE, TW, MACH = 1.4, 3.5, 2.0e-4, 0.72, 1.0, 1.0, 1.0, 2.0


def main():
    d = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = (sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv
              else os.path.join(here, "Images", os.path.basename(os.path.normpath(d))))
    os.makedirs(outdir, exist_ok=True)
    adiabatic = "--adiabatic" in sys.argv
    TW = float(sys.argv[sys.argv.index("--twall") + 1]) if "--twall" in sys.argv else globals()["TW"]
    U = MACH * np.sqrt(GAM * 1.0 / RHO)
    Taw = TE * (1 + np.sqrt(PR) * (GAM - 1) / 2 * MACH**2)
    if adiabatic: TW = Taw
    Ts = TE + 0.5 * (TW - TE) + 0.22 * (Taw - TE); Cs = TE / Ts
    pfs = sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))
    use = pfs[-max(1, len(pfs) // 3):]; Q = []; TAU = []; Y0 = []; TWALL = []
    for pf in use:
        ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        nx, ny = ds.domain_dimensions[:2]; x = (np.arange(nx) + 0.5) / nx
        y = float(ds.domain_left_edge[1]) + (np.arange(ny) + 0.5) * float(ds.domain_width[1]) / ny
        q = np.asarray(cg[("boxlib", "ablation_q")])[:, :, 0] if ("boxlib", "ablation_q") in ds.field_list else np.zeros(tuple(ds.domain_dimensions[:2]))
        Tf = np.asarray(cg[("boxlib", "T")])[:, :, 0]; phi = np.asarray(cg[("boxlib", "phi")])[:, :, 0]; u = np.asarray(cg[("boxlib", "velocityx")])[:, :, 0]
        j = np.where(phi[nx // 2] >= 0.5)[0][0]                 # first fluid row
        TWALL.append(1.5 * Tf[:, j] - 0.5 * Tf[:, j + 1])       # gas temperature extrapolated to the wall face
        Q.append(q[:, j]); TAU.append(MU * u[:, j] / y[j])      # wall at y = 0, cell centre at y[j] = dy/2
        dy = y[1] - y[0]; Y0.append((y[j] - u[:, j] * dy / np.maximum(u[:, j + 1] - u[:, j], 1e-30)) / dy)   # where the first two fluid cells put u = 0, in cells
    qw = np.mean(Q, axis=0); tau = np.mean(TAU, axis=0); t0, t1 = float(yt.load(use[0]).current_time), float(yt.load(use[-1]).current_time)
    Rex = RHO * U * x / MU
    qref = 0.332 * np.sqrt(Cs) * PR**(-2.0 / 3.0) / np.sqrt(Rex) * RHO * U * CP * (Taw - TW)
    cfref = 0.664 * np.sqrt(Cs) / np.sqrt(Rex); cf = tau / (0.5 * RHO * U * U)
    m = (x > 0.2) & (x < 0.9); Twn = np.mean(TWALL, axis=0)
    print(f"{d}: averaged over t = {t0:.2f} .. {t1:.2f} ({len(use)} plot files); U = {U:.4f}, T_aw = {Taw:.4f}, T_w = {TW:.4f}, T* = {Ts:.4f}, C* = {Cs:.4f}")
    if adiabatic:
        r_num = (Twn - TE) / (TE * (GAM - 1) / 2 * MACH**2)
        for xx in (0.2, 0.4, 0.6, 0.8, 0.9):
            i = np.argmin(np.abs(x - xx))
            print(f"  x = {x[i]:.2f}: wall temperature {Twn[i]:.4f}   T_aw {Taw:.4f}   ({100 * (Twn[i] - Taw) / (Taw - TE):+.1f} % of T_aw - T_e);"
                  f"  recovery factor {r_num[i]:.4f} (sqrt Pr = {np.sqrt(PR):.4f});   c_f = {cf[i]:.5f}   reference {cfref[i]:.5f}   ({100 * (cf[i] / cfref[i] - 1):+.1f} %)")
        print(f"  mean recovery factor over 0.2 < x < 0.9: {r_num[m].mean():.4f}   (sqrt Pr = {np.sqrt(PR):.4f}, {100 * (r_num[m].mean() / np.sqrt(PR) - 1):+.1f} %)")
    else:
        n_fit = np.polyfit(np.log(x[m]), np.log(np.abs(qw[m])), 1)[0]
        for xx in (0.1, 0.2, 0.4, 0.6, 0.8, 0.9):
            i = np.argmin(np.abs(x - xx))
            print(f"  x = {x[i]:.2f} (Re_x = {Rex[i]:7.0f}): q_w = {qw[i]:+.5f}   reference {qref[i]:+.5f}   ({100 * (qw[i] / qref[i] - 1):+.1f} %);"
                  f"   c_f = {cf[i]:.5f}   reference {cfref[i]:.5f}   ({100 * (cf[i] / cfref[i] - 1):+.1f} %)")
        print(f"  |q_w| ~ x^n over 0.2 < x < 0.9: n = {n_fit:.3f}   (theory -0.500);   mean q_w / reference there: {np.mean(qw[m] / qref[m]):.3f}"
              f"   ({'gas heats the plate' if qref[m].mean() > 0 else 'plate heats the gas'})")
    print(f"  no-slip position implied by the first two fluid cells, 0.2 < x < 0.9: y = {np.mean(np.mean(Y0, axis=0)[m]):+.2f} dy   (0 = on the wall face)")
    REF = "Reference Temperature Method"
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3)); st = max(1, len(x) // 40)
    if adiabatic:
        ax[0].axhline(Taw, color="C3", lw=1.6, label=r"Recovery Temperature, $r = \sqrt{Pr}$"); ax[0].plot(x[::st], Twn[::st], "ko", ms=4, mfc="none", label="Numerical")
        ax[0].set_ylabel("Wall Temperature [K]"); ax[0].set_title("Adiabatic Wall Temperature"); ax[0].set_xlim(0, 1); ax[0].set_ylim(TE, Taw + 0.3 * (Taw - TE))
    else:
        ax[0].loglog(x, np.abs(qref), "C3-", lw=1.6, label=REF); ax[0].loglog(x[::st], np.abs(qw[::st]), "ko", ms=4, mfc="none", label="Numerical")
        ax[0].set_ylabel(r"Wall Heat Flux Magnitude [W/m$^2$]"); ax[0].set_title("Heat Flux into the Plate" if qref[m].mean() > 0 else "Heat Flux from the Plate into the Gas"); ax[0].set_xlim(0.02, 1)
    ax[1].loglog(x, cfref, "C3-", lw=1.6, label=REF); ax[1].loglog(x[::st], cf[::st], "ko", ms=4, mfc="none", label="Numerical")
    ax[1].set_ylabel("Skin Friction Coefficient [-]"); ax[1].set_title("Skin Friction"); ax[1].set_xlim(0.02, 1)
    for a in ax: a.set_xlabel("Distance from the Leading Edge [m]"); a.grid(alpha=0.3, which="both"); a.legend(fontsize=9)
    fig.suptitle(f"Laminar Boundary Layer on a Flat Plate at Mach {MACH:g}, " + ("Adiabatic Wall" if adiabatic else rf"$T_w/T_\infty$ = {TW / TE:g}"))
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "flat_plate.png"), dpi=200); plt.close(fig)
    print(f"  wrote {outdir}/flat_plate.png")


if __name__ == "__main__":
    main()
