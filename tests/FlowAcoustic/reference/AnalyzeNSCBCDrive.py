"""NSCBC acoustic drive -- 1D tube unit tests.

Drives a tube through the NSCBC incoming characteristic (the path the driven
bubble uses) and compares against the d'Alembert analytic wave.  The drive is
read straight from each input, so the Fourier-series form needs no bookkeeping
here: S(t) = sum_k A_k sin(w_k t + phi_k).

  Images/NSCBCDrive_<case>.png   transient p(x) vs analytic + error history
  Images/NSCBCDrive_<case>.gif   animated p(x,t)
  Images/NSCBCDrive_transfer.png relaxation transfer function vs theory
"""
import glob, os, re, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import yt; yt.funcs.mylog.setLevel(50)

HERE = os.path.dirname(os.path.abspath(__file__))
IMG  = os.path.join(HERE, "Images"); os.makedirs(IMG, exist_ok=True)
TEST = os.path.normpath(os.path.join(HERE, ".."))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "bin", "tests", "FlowAcoustic"))
CASES = ["Pulse", "Sine", "Fourier"]

def parse(path):
    kv = {}
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1); kv[k.strip()] = v.strip()
    return kv

def drive_of(kv):
    """(A_k, w_k, phi_k) arrays from nscbc.xlo.drive_*; scalar = one term."""
    f = lambda k: [float(t) for t in kv.get(k, "").split()] if kv.get(k) else []
    A, W, P = f("nscbc.xlo.drive_amp"), f("nscbc.xlo.drive_omega"), f("nscbc.xlo.drive_phase")
    if not P: P = [0.0]*len(A)
    return list(zip(A, W, P))

def S(t, terms):
    return sum(a*np.sin(w*t + p) for a, w, p in terms)

def load(case):
    pfs = sorted(glob.glob(os.path.join(ROOT, f"output_NSCBCDrive_{case}", "*cell")),
                 key=lambda s: int(re.search(r"(\d+)cell", s).group(1)))
    out = []
    for pf in pfs:
        ds = yt.load(pf); ad = ds.all_data()
        x = np.array(ad["x"], float); p = np.array(ad["pressure"], float)
        xs = np.unique(np.round(x, 9))
        out.append((float(ds.current_time), xs,
                    np.array([p[np.isclose(x, v)].mean() for v in xs])))
    return out

C0 = math.sqrt(1.4); P0 = 1.0
summary = []
for case in CASES:
    kv = parse(os.path.join(TEST, f"input_NSCBCDrive_{case}"))
    terms = drive_of(kv); fr = load(case)
    if not fr:
        print(f"{case}: no plotfiles"); continue
    sig = kv.get("nscbc.xlo.sigma", "?"); lref = float(kv.get("nscbc.xlo.L_ref", "1"))
    K = float(sig)*C0/lref
    exact = lambda t, x: P0 + S(t - x/C0, terms)*(t - x/C0 > 0)

    fig, ax = plt.subplots(1, 2, figsize=(12.2, 4.2))
    sel = [fr[i] for i in np.linspace(1, len(fr)-1, 6).astype(int)]
    cm = plt.get_cmap("viridis")
    for j,(t,x,p) in enumerate(sel):
        ax[0].plot(x, p, color=cm(j/5), lw=1.5, label=f"t={t:.2f}")
        ax[0].plot(x, exact(t,x), color="0.55", lw=0.9, ls=":", zorder=0)
    ax[0].set_xlabel("x"); ax[0].set_ylabel("pressure"); ax[0].grid(alpha=.25, lw=.6)
    ax[0].set_title(f"{case}: simulation (color) vs analytic (dotted)", fontsize=10)
    ax[0].legend(fontsize=7.5, frameon=False, ncol=2)
    ts = np.array([f[0] for f in fr])
    l2 = np.array([np.sqrt(np.mean((f[2]-exact(f[0],f[1]))**2)) for f in fr])
    amp = np.array([(f[2].max()-f[2].min())/2 for f in fr])
    # Peak of the actual waveform, not sum|A_k|: for a Fourier partial sum the
    # terms do not peak together, so sum|A_k| overstates the target by ~30%.
    _tt = np.linspace(0.0, 2*np.pi/min(w for _,w,_ in terms), 20001)
    tgt = float(np.max(np.abs(S(_tt, terms))))
    ax[1].semilogy(ts, l2, "-", color="#D55E00", lw=1.5, label=r"$L_2$ error vs analytic")
    ax[1].set_xlabel("t"); ax[1].set_ylabel(r"$L_2$ error", color="#D55E00")
    ax[1].tick_params(axis="y", colors="#D55E00"); ax[1].grid(alpha=.25, lw=.6)
    a2 = ax[1].twinx()
    a2.plot(ts, amp/tgt, "-", color="#0072B2", lw=1.5, label="amplitude / target")
    a2.axhline(1.0, color="0.6", ls="--", lw=.9)
    a2.set_ylabel("amplitude / target", color="#0072B2"); a2.tick_params(axis="y", colors="#0072B2")
    a2.set_ylim(0, 1.3)
    ax[1].set_title(f"K = {K:.0f} 1/s,  w/K = {terms[0][1]/K:.3f}", fontsize=10)
    fig.suptitle(f"NSCBC drive -- {case}", fontsize=12); fig.tight_layout()
    fig.savefig(os.path.join(IMG, f"NSCBCDrive_{case}.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    summary.append((case, amp[-1]/tgt, l2[-1]))
    print(f"wrote Images/NSCBCDrive_{case}.png   final amp/target={amp[-1]/tgt:.3f}  L2={l2[-1]:.2e}")

    figg, axg = plt.subplots(figsize=(7.2, 3.6))
    axg.set_xlim(0,1); axg.set_ylim(P0-1.6*tgt, P0+1.6*tgt)
    axg.set_xlabel("x"); axg.set_ylabel("pressure"); axg.grid(alpha=.25, lw=.6)
    ln, = axg.plot([],[], lw=1.9, color="#0072B2", label="simulation")
    lx, = axg.plot([],[], lw=1.1, color="0.55", ls=":", label="analytic")
    axg.legend(fontsize=8, frameon=False, loc="upper right"); tt = axg.set_title("")
    def upd(i):
        t,x,p = fr[i]; ln.set_data(x,p); lx.set_data(x,exact(t,x))
        tt.set_text(f"{case}   t = {t:.3f}"); return ln,lx
    FuncAnimation(figg, upd, frames=len(fr), blit=False).save(
        os.path.join(IMG, f"NSCBCDrive_{case}.gif"), writer=PillowWriter(fps=10))
    plt.close(figg)
    print(f"wrote Images/NSCBCDrive_{case}.gif")

print("\n%-10s %14s %12s" % ("case","amp/target","L2 err"))
for c,a,l in summary: print("%-10s %14.3f %12.2e" % (c,a,l))
