"""NSCBC acoustic drive vs the primitive-Dirichlet drive, 1D tube.

Left column  : Dirichlet drive (input_AcousticBC)      -- the working reference
Right column : NSCBC drive     (input_NSCBCDrive_Sine2) -- identical input except
                                                           the boundary block

Writes  Images/NSCBCDrive_compare.png   transient p(x) + error vs analytic
        Images/NSCBCDrive_wave.gif      animated p(x,t) for both
"""
import glob, os, re, sys, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import yt; yt.funcs.mylog.setLevel(50)

HERE = os.path.dirname(os.path.abspath(__file__))
IMG  = os.path.join(HERE, "Images"); os.makedirs(IMG, exist_ok=True)
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "bin", "tests", "FlowAcoustic"))
C0, AMP, OM, P0 = math.sqrt(1.4), 0.01, 31.415926536, 1.0

def load(d):
    pfs = sorted(glob.glob(os.path.join(ROOT, d, "*cell")),
                 key=lambda s: int(re.search(r"(\d+)cell", s).group(1)))
    out = []
    for pf in pfs:
        ds = yt.load(pf); ad = ds.all_data()
        x = np.array(ad["x"], float); p = np.array(ad["pressure"], float)
        xs = np.unique(np.round(x, 9))
        out.append((float(ds.current_time), xs,
                    np.array([p[np.isclose(x, v)].mean() for v in xs])))
    return out

def exact(t, x):
    ph = t - x / C0
    return P0 + AMP * np.sin(OM * ph) * (ph > 0)

CASES = [("out_ref2", "Dirichlet drive  (input_AcousticBC)"),
         ("out_s2",   "NSCBC drive  (only the BC block differs)")]
data = {d: load(d) for d, _ in CASES}
for d, lab in CASES:
    print(f"{lab}: {len(data[d])} frames")

# ---------------------------------------------------------------- figure ---
fig, ax = plt.subplots(2, 2, figsize=(12.5, 7.0))
for col, (d, lab) in enumerate(CASES):
    fr = data[d]
    if not fr:
        ax[0][col].text(.5,.5,"no data",transform=ax[0][col].transAxes,ha="center"); continue
    sel = [fr[i] for i in np.linspace(0, len(fr)-1, min(6, len(fr))).astype(int)]
    cm = plt.get_cmap("viridis")
    for j,(t,x,p) in enumerate(sel):
        ax[0][col].plot(x, p, color=cm(j/max(len(sel)-1,1)), lw=1.4, label=f"t={t:.3f}")
        ax[0][col].plot(x, exact(t,x), color="0.6", lw=0.9, ls=":", zorder=0)
    ax[0][col].set_title(lab, fontsize=10)
    ax[0][col].set_xlabel("x"); ax[0][col].set_ylabel("pressure")
    ax[0][col].legend(fontsize=7, frameon=False, ncol=2)
    ax[0][col].grid(alpha=.25, lw=.6)
    # domain-mean pressure and L2 error vs the analytic wave
    ts = [f[0] for f in fr]
    pm = [f[2].mean() for f in fr]
    l2 = [np.sqrt(np.mean((f[2]-exact(f[0],f[1]))**2)) for f in fr]
    ax[1][col].plot(ts, pm, "o-", color="#0072B2", ms=3.5, lw=1.4, label=r"domain-mean $p$")
    ax[1][col].axhline(P0, color="0.6", lw=.9, ls="--", label=r"$p_0$ (should hold)")
    ax[1][col].set_xlabel("t"); ax[1][col].set_ylabel(r"mean $p$")
    a2 = ax[1][col].twinx()
    a2.semilogy(ts, l2, "s--", color="#D55E00", ms=3.5, lw=1.2, label=r"$L_2$ err vs analytic")
    a2.set_ylabel(r"$L_2$ error", color="#D55E00"); a2.tick_params(axis="y", colors="#D55E00")
    ax[1][col].legend(fontsize=7.5, frameon=False, loc="lower left")
    ax[1][col].grid(alpha=.25, lw=.6)
fig.suptitle("NSCBC acoustic drive: the boundary block is the only difference", fontsize=12)
fig.tight_layout()
for e in ("png","pdf"):
    fig.savefig(os.path.join(IMG, f"NSCBCDrive_compare.{e}"), dpi=180, bbox_inches="tight")
print("wrote", os.path.join(IMG, "NSCBCDrive_compare.png"))
plt.close(fig)

# ------------------------------------------------------------------- gif ---
n = max(len(data[d]) for d,_ in CASES)
fig, axg = plt.subplots(1, 2, figsize=(12, 4.0))
lines = []
for col,(d,lab) in enumerate(CASES):
    axg[col].set_xlim(0,1); axg[col].set_title(lab, fontsize=10)
    axg[col].set_xlabel("x"); axg[col].set_ylabel("pressure"); axg[col].grid(alpha=.25, lw=.6)
    axg[col].set_ylim(0.0, 1.05)
    ln,  = axg[col].plot([],[], lw=1.8, color="#0072B2", label="simulation")
    lnx, = axg[col].plot([],[], lw=1.0, color="0.55", ls=":", label="analytic")
    axg[col].legend(fontsize=8, frameon=False, loc="lower left")
    lines.append((ln,lnx))
ttl = fig.suptitle("")
def upd(i):
    for col,(d,_) in enumerate(CASES):
        fr = data[d]
        if not fr: continue
        t,x,p = fr[min(i,len(fr)-1)]
        lines[col][0].set_data(x,p); lines[col][1].set_data(x,exact(t,x))
    ttl.set_text(f"t = {data[CASES[0][0]][min(i,len(data[CASES[0][0]])-1)][0]:.3f}")
    return [l for pair in lines for l in pair]
anim = FuncAnimation(fig, upd, frames=n, blit=False)
anim.save(os.path.join(IMG,"NSCBCDrive_wave.gif"), writer=PillowWriter(fps=4))
print("wrote", os.path.join(IMG,"NSCBCDrive_wave.gif"))
