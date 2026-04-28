"""
NSCBC / NSCBC4 boundary-condition diagnostic for the FlowAcoustic pulse tests.

Reads the AMReX plot files produced by:
    tests/FlowAcoustic/input_Acoustic_Pulse_Neumann          (transmissive baseline)
    tests/FlowAcoustic/input_Acoustic_Pulse_Wall             (true reflecting wall)
    tests/FlowAcoustic/input_Acoustic_Pulse_NSCBC            (target_p = base)
    tests/FlowAcoustic/input_Acoustic_Pulse_NSCBC4           (target_p = base, nghost=4)
    tests/FlowAcoustic/input_Acoustic_Pulse_NSCBC_PtargetDrift   (target_p = 0.8)
    tests/FlowAcoustic/input_Acoustic_Pulse_NSCBC4_PtargetDrift  (target_p = 0.8, nghost=4)

Produces plots that make it obvious whether
  (a) the NSCBC / NSCBC4 outflow is actually non-reflecting compared to a
      true reflecting wall, and
  (b) the LODI relaxation term that uses target_p is actually driving the
      boundary state toward target_p (the user-reported "p_target does
      nothing" bug). The drift cases use target_p = 0.8 (BELOW p_base) on
      purpose - target_p > p_base would reverse the flow at xhi and the
      subsonic-outflow handler does not gracefully transition to inflow,
      so it explodes.

Run from the repo root, e.g. (Windows / Git Bash):
    python tests/FlowAcoustic/reference/acoustic_bc_compare.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import yt

yt.funcs.mylog.setLevel(40)


# ============================================================================
# CONFIGURATION
# ============================================================================

# All paths are resolved relative to this script's location so it works
# regardless of where the user invokes it from.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BIN = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "bin",
                                     "tests", "FlowAcoustic"))
_OUT = os.path.join(_HERE, "Images")
os.makedirs(_OUT, exist_ok=True)

CASES = [
    {
        "label":   "Neumann (transmissive)",
        "key":     "neumann",
        "dir":     os.path.join(_BIN, "output_Pulse_Neumann"),
        "color":   "tab:red",
        "ls":      "-",
        "target_p": None,
    },
    {
        "label":   "Reflecting wall (REFLECT_EVEN/ODD)",
        "key":     "wall",
        "dir":     os.path.join(_BIN, "output_Pulse_Wall"),
        "color":   "tab:gray",
        "ls":      "-",
        "target_p": None,
    },
    {
        "label":   "NSCBC outflow (target_p=p_base)",
        "key":     "nscbc",
        "dir":     os.path.join(_BIN, "output_Pulse_NSCBC"),
        "color":   "tab:blue",
        "ls":      "-",
        "target_p": 1.0,
    },
    {
        "label":   "NSCBC4 outflow (target_p=p_base)",
        "key":     "nscbc4",
        "dir":     os.path.join(_BIN, "output_Pulse_NSCBC4"),
        "color":   "tab:green",
        "ls":      "-",
        "target_p": 1.0,
    },
    {
        "label":   "NSCBC drift (target_p=0.8)",
        "key":     "nscbc_drift",
        "dir":     os.path.join(_BIN, "output_Pulse_NSCBC_PtargetDrift"),
        "color":   "tab:purple",
        "ls":      "--",
        "target_p": 0.8,
    },
    {
        "label":   "NSCBC4 drift (target_p=0.8)",
        "key":     "nscbc4_drift",
        "dir":     os.path.join(_BIN, "output_Pulse_NSCBC4_PtargetDrift"),
        "color":   "tab:olive",
        "ls":      "--",
        "target_p": 0.8,
    },
]

# Domain (must match the input files)
X_LO, X_HI = 0.0, 1.0
Y_MID = 0.0

# Snapshot times we want to draw on the multi-panel time-evolution plot.
# Pulse center starts at x=0.3, c~1.183, so it reaches xhi at t~0.59 s.
SNAPSHOT_TIMES = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25]


# ============================================================================
# DATA LOADING
# ============================================================================

def list_plot_files(case_dir):
    """Return sorted list of *cell directories (AMReX plot files)."""
    if not os.path.isdir(case_dir):
        return []
    plots = [os.path.join(case_dir, d) for d in os.listdir(case_dir)
             if os.path.isdir(os.path.join(case_dir, d)) and d.endswith("cell")]
    plots.sort()
    return plots


def detect_velocity_field(ds):
    """yt sometimes exposes velocity as 'velocityx' and sometimes 'velocity_x'."""
    fl = {f for f in ds.field_list}
    for cand in [("boxlib", "velocity_x"), ("boxlib", "velocityx"),
                 ("amrex",  "velocity_x"), ("amrex",  "velocityx")]:
        if cand in fl:
            return cand[1]
    return "velocity_x"  # fallback


def detect_momentum_field(ds):
    fl = {f for f in ds.field_list}
    for cand in [("boxlib", "momentum_x"), ("boxlib", "momentumx"),
                 ("amrex",  "momentum_x"), ("amrex",  "momentumx")]:
        if cand in fl:
            return cand[1]
    return "momentum_x"


def extract_ray(ds, vx_name, mx_name):
    """Extract a 1D ray along x at y=0 and return sorted arrays."""
    ray_start = ds.arr([X_LO,  Y_MID, 0.0], "code_length")
    ray_end   = ds.arr([X_HI,  Y_MID, 0.0], "code_length")
    ray = ds.ray(ray_start, ray_end)

    order = np.argsort(np.array(ray["x"]))
    x   = np.array(ray["x"])[order]
    rho = np.array(ray["density"])[order]
    p   = np.array(ray["pressure"])[order]
    u   = np.array(ray[vx_name])[order]
    M   = np.array(ray[mx_name])[order]
    return x, rho, p, u, M


def load_case(case):
    """Load every snapshot for a case. Returns dict of arrays keyed by time."""
    plots = list_plot_files(case["dir"])
    if not plots:
        print(f"  [WARN] no plot files in {case['dir']}")
        return None

    times = []
    snaps = []  # list of (x, rho, p, u, M)
    vx_name = mx_name = None

    for pf in plots:
        ds = yt.load(pf)
        if vx_name is None:
            vx_name = detect_velocity_field(ds)
            mx_name = detect_momentum_field(ds)
        x, rho, p, u, M = extract_ray(ds, vx_name, mx_name)
        times.append(float(ds.current_time))
        snaps.append((x, rho, p, u, M))

    return {
        "times": np.array(times),
        "x":     snaps[0][0],
        "rho":   np.array([s[1] for s in snaps]),
        "p":     np.array([s[2] for s in snaps]),
        "u":     np.array([s[3] for s in snaps]),
        "M":     np.array([s[4] for s in snaps]),
    }


# ============================================================================
# PLOTTING HELPERS
# ============================================================================

def closest_snapshot(data, t_target):
    """Return (idx, t_actual) of the snapshot nearest to t_target."""
    idx = int(np.argmin(np.abs(data["times"] - t_target)))
    return idx, data["times"][idx]


def plot_time_snapshots(loaded, field, ylabel, fname):
    """Multi-panel: each row is a snapshot time, all cases overlaid."""
    n = len(SNAPSHOT_TIMES)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.0 * n + 1), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, t_req in zip(axes, SNAPSHOT_TIMES):
        for case in CASES:
            data = loaded.get(case["key"])
            if data is None:
                continue
            idx, t_act = closest_snapshot(data, t_req)
            y = data[field][idx]
            ax.plot(data["x"], y,
                    color=case["color"], linestyle=case["ls"],
                    linewidth=1.6, alpha=0.85,
                    label=f"{case['label']} (t={t_act:.3f}s)")
        ax.axvline(X_HI, color="k", linewidth=0.5, alpha=0.5)
        ax.set_title(f"t ~ {t_req:.2f} s", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper left", ncol=2)

    axes[-1].set_xlabel("x (m)")
    fig.suptitle(f"{ylabel} - time evolution by BC", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(_OUT, fname)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_boundary_history(loaded, fname):
    """Track values at xhi (right boundary) and 5% inboard over time.

    This is the headline plot for diagnosing the p_target bug:
      * Neumann: p stays around p_base after pulse, then bounces
                 back and forth (reflections).
      * NSCBC w/ target_p=1.0: p settles back to ~1.0 quickly with
                 little reflection.
      * NSCBC w/ target_p=1.5: p drifts up toward 1.5 over time.
                 If it does NOT drift, p_target is broken.
    """
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    fields = [("p",        "Pressure at boundary"),
              ("u",         "Velocity at boundary  (u<0 => INFLOW => runaway)"),
              ("rho",       "Density at boundary"),
              ("u_signed",  "Sign(u) at boundary (1=outflow, -1=inflow)")]

    # Pick the cell nearest to xhi (last interior cell).
    for ax, (field, title) in zip(axes, fields):
        for case in CASES:
            data = loaded.get(case["key"])
            if data is None:
                continue
            i_b = int(np.argmin(np.abs(data["x"] - X_HI)))  # nearest to xhi
            if field == "u_signed":
                y_b = np.sign(data["u"][:, i_b])
            else:
                y_b = data[field][:, i_b]
            ax.plot(data["times"], y_b,
                    color=case["color"], linestyle=case["ls"],
                    linewidth=1.8, label=case["label"])
            if field == "p" and case["target_p"] is not None:
                ax.axhline(case["target_p"],
                           color=case["color"], linestyle=":", linewidth=1.0,
                           alpha=0.6,
                           label=f"target_p ({case['label']}) = {case['target_p']}")
        if field == "u":
            ax.axhline(0.0, color="k", linewidth=0.5, alpha=0.5)
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best", ncol=2)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Right-boundary state vs time (last interior cell at x = xhi)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(_OUT, fname)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_domain_integrals(loaded, fname):
    """Integrate mass, momentum, energy proxies over the domain to show
    that NSCBC actually lets stuff *leave* (mass / momentum drop) while
    Neumann conserves them."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    titles = ["sum(rho)*dx  (~total mass)",
              "sum(M)*dx    (~total x-momentum)",
              "max(|p - <p_base>|) over domain"]

    for case in CASES:
        data = loaded.get(case["key"])
        if data is None:
            continue
        x = data["x"]
        dx = x[1] - x[0]
        mass = np.sum(data["rho"], axis=1) * dx
        mom  = np.sum(data["M"],   axis=1) * dx
        # Use 1.0 as nominal base pressure regardless of target_p drift case.
        max_dp = np.max(np.abs(data["p"] - 1.0), axis=1)

        axes[0].plot(data["times"], mass, color=case["color"],
                     linestyle=case["ls"], label=case["label"])
        axes[1].plot(data["times"], mom, color=case["color"],
                     linestyle=case["ls"], label=case["label"])
        axes[2].plot(data["times"], max_dp, color=case["color"],
                     linestyle=case["ls"], label=case["label"])

    for ax, t in zip(axes, titles):
        ax.set_ylabel(t)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Domain integrals - Neumann conserves; NSCBC bleeds out",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(_OUT, fname)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_xt_diagram(loaded, field, label, fname):
    """One x-t (waterfall) image per case so reflections are obvious."""
    n = sum(1 for c in CASES if loaded.get(c["key"]) is not None)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5.5), sharey=True)
    if n == 1:
        axes = [axes]

    cases_present = [c for c in CASES if loaded.get(c["key"]) is not None]

    # Use a common color scale so cases are directly comparable.
    vmin = min(np.min(loaded[c["key"]][field]) for c in cases_present)
    vmax = max(np.max(loaded[c["key"]][field]) for c in cases_present)

    for ax, case in zip(axes, cases_present):
        data = loaded[case["key"]]
        im = ax.imshow(
            data[field],
            aspect="auto",
            origin="lower",
            extent=[data["x"][0], data["x"][-1],
                    data["times"][0], data["times"][-1]],
            cmap="RdBu_r",
            vmin=vmin, vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(case["label"], fontsize=10)
        ax.set_xlabel("x (m)")
        ax.axvline(X_HI, color="k", linewidth=0.6, alpha=0.6)

    axes[0].set_ylabel("time (s)")
    cbar = fig.colorbar(im, ax=axes, orientation="vertical",
                        fraction=0.025, pad=0.02)
    cbar.set_label(label)
    fig.suptitle(f"x-t diagram: {label}", fontsize=12, fontweight="bold")
    out = os.path.join(_OUT, fname)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def print_diagnostic_summary(loaded):
    """Spit out numbers the user can quote when filing the bug report."""
    print("\n" + "=" * 72)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 72)
    for case in CASES:
        data = loaded.get(case["key"])
        if data is None:
            print(f"\n[{case['label']}]   no data")
            continue
        i_b = int(np.argmin(np.abs(data["x"] - X_HI)))
        p_b0 = data["p"][0,  i_b]
        p_bf = data["p"][-1, i_b]
        u_bf = data["u"][-1, i_b]
        u_b_traj = data["u"][:, i_b]
        flow_reversed = bool(np.any(u_b_traj < -1e-3))
        max_dp_final = np.max(np.abs(data["p"][-1] - 1.0))
        max_p_anywhere = float(np.max(data["p"]))
        min_p_anywhere = float(np.min(data["p"]))
        # If the run terminated early (sim crashed) the t_final will be
        # less than what the input file requested.
        print(f"\n[{case['label']}]")
        print(f"  N_snapshots             : {len(data['times'])}")
        print(f"  t_final reached         : {data['times'][-1]:.4f} s")
        print(f"  p(xhi, t=0)             : {p_b0:.5f}")
        print(f"  p(xhi, t=final)         : {p_bf:.5f}")
        print(f"  u(xhi, t=final)         : {u_bf:.5e}")
        print(f"  flow reversed at xhi    : {flow_reversed}"
              + (" <-- RUNAWAY HAZARD" if flow_reversed else ""))
        print(f"  max p anywhere, anytime : {max_p_anywhere:.5f}")
        print(f"  min p anywhere, anytime : {min_p_anywhere:.5f}")
        print(f"  max |p - 1| at t_final  : {max_dp_final:.5e}")
        if case["target_p"] is not None:
            drift = p_bf - case["target_p"]
            verdict = "CLOSE TO TARGET" if abs(drift) < 0.02 else "NOT AT TARGET"
            print(f"  target_p                : {case['target_p']:.5f}")
            print(f"  p_b - target_p (final)  : {drift:+.5e}   ({verdict})")
            # For the drift cases, also report the *fraction* of the
            # mismatch that has been closed - this is the cleanest
            # number for arguing that p_target does (or doesn't) work.
            mismatch0 = 1.0 - case["target_p"]
            if abs(mismatch0) > 1e-6:
                closed = 1.0 - drift / mismatch0
                print(f"  fraction of (p_base - target_p) closed: {closed*100:.1f}%")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 72)
    print("LOADING CASES")
    print("=" * 72)
    loaded = {}
    for case in CASES:
        print(f"\n[{case['label']}] -> {case['dir']}")
        d = load_case(case)
        if d is not None:
            loaded[case["key"]] = d
            print(f"  loaded {len(d['times'])} snapshots, "
                  f"t in [{d['times'][0]:.4f}, {d['times'][-1]:.4f}]")

    if not loaded:
        print("\nNo cases loaded - run the simulations first.")
        sys.exit(1)

    print("\n" + "=" * 72)
    print("PLOTS")
    print("=" * 72)
    plot_time_snapshots(loaded, "p",   "Pressure",   "snapshots_pressure.png")
    plot_time_snapshots(loaded, "rho", "Density",    "snapshots_density.png")
    plot_time_snapshots(loaded, "u",   "Velocity",   "snapshots_velocity.png")
    plot_time_snapshots(loaded, "M",   "Momentum_x", "snapshots_momentum.png")

    plot_boundary_history(loaded, "boundary_history.png")
    plot_domain_integrals(loaded, "domain_integrals.png")
    plot_xt_diagram(loaded, "p", "Pressure",   "xt_pressure.png")
    plot_xt_diagram(loaded, "u", "Velocity_x", "xt_velocity.png")

    print_diagnostic_summary(loaded)

    print("\nDone. Plots in:", _OUT)


if __name__ == "__main__":
    main()
