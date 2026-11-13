"""
R29 Couette validation: error tables + paper-ready figures (single- and two-phase).

New script (2026-10-29); the older couette_analysis.py / couette_analysis_multi.py
are unchanged and still analyse a single output directory each.

Usage (from anywhere):
    python3 couette_R29_analysis.py                # all R29_v3 runs, default paths
    python3 couette_R29_analysis.py <runs_root>    # root that holds the R29_v3_* dirs

What it does
  * loads every plotfile of each run, averages u over x (periodic direction)
  * steady-state check: max |u(t_last) - u(t_prev)| / U_top
  * errors are normalised by the plate speed U_top (u vanishes at the lower plate,
    so a pointwise relative error is undefined there):
        max  = max_j |u_j - u_ex(y_j)| / U_top
        L2   = sqrt(mean_j (u_j - u_ex)^2) / U_top
  * two-phase: errors over the whole channel, outside the diffuse band
    |y - h/2| > 3 eps, and inside it; interface velocity u(h/2) (linear interpolation
    between the two cells adjacent to y = h/2); shear stress
    tau_j = mu(eta_j) (u_{j+1} - u_{j-1}) / (2 dy) with the code's mixture rule
    mu = eta mu0 + (1 - eta) mu1 (arithmetic; Hydro2.cpp, RHS viscous block).
  * diagnostics that decompose the error (printed, written to the table file):
        - "ghost-wall" exact: the Dirichlet momentum BC sets the GHOST cell to the
          wall value (src/BC/Expression.cpp), i.e. the plates sit at y = -dy/2 and
          y = h + dy/2, not on the domain faces;
        - "diffuse" exact: steady 1D Couette with the diffuse mu(y) from the
          plotfile eta, constant stress, plates at the ghost centres:
          u(y) = U int_{y_lo}^{y} dy'/mu / int_{y_lo}^{y_hi} dy'/mu.
"""

import os
import sys
import glob
import numpy as np
import yt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

yt.funcs.mylog.setLevel(40)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '../../../bin/tests/FlowCouette')
IMG = os.path.join(HERE, 'Images', 'R29' + os.environ.get('COUETTE_SUFFIX', ''))
os.makedirs(IMG, exist_ok=True)

H = 3.0
U = 0.1
Y_INT = 1.5
MU_UP, MU_LO = 2.0, 1.0      # mu0 (eta = 1, upper), mu1 (eta = 0, lower)  -- input_hydro2_multi
MU_SINGLE = 10.0

plt.rcParams.update({
    'font.size': 14, 'axes.titlesize': 15, 'axes.labelsize': 15,
    'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 13,
    'font.family': 'serif', 'mathtext.fontset': 'cm',
})
MARKERS = ['o', 's', '^', 'D', 'v']


# ----------------------------------------------------------------------------
def load_run(path):
    files = sorted(glob.glob(os.path.join(path, 'output', '*cell')),
                   key=lambda f: int(os.path.basename(f)[:-4]))   # numeric: 95000cell < 200000cell
    if not files:
        raise SystemExit(f'no plotfiles in {path}')

    def prof(f):
        ds = yt.load(f)
        g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        avg = lambda n: np.array(g[n])[:, :, 0].mean(axis=0)
        return dict(t=float(ds.current_time), y=avg('y'), u=avg('velocityx'),
                    v=avg('velocityy'), eta=avg('eta'),
                    dy=float(ds.domain_width[1] / ds.domain_dimensions[1]),
                    ylo=float(ds.domain_left_edge[1]), yhi=float(ds.domain_right_edge[1]))
    last, prev = prof(files[-1]), prof(files[-2])
    last['du_last'] = np.max(np.abs(last['u'] - prev['u'])) / U
    last['dt_last'] = last['t'] - prev['t']
    last['files'] = files
    return last


def exact_single(y, y0=0.0, y1=H):
    return U * (y - y0) / (y1 - y0)


def exact_two_layer(y, y0=0.0, y1=H, yi=Y_INT):
    # sharp interface, plates at y0 (u=0) and y1 (u=U), mu_lo below yi, mu_up above
    R = (yi - y0) / MU_LO + (y1 - yi) / MU_UP
    tau = U / R
    return np.where(y <= yi, tau * (y - y0) / MU_LO, tau * (yi - y0) / MU_LO + tau * (y - yi) / MU_UP), tau


def exact_diffuse(y, eps, y0, y1, rule='arith', yi=Y_INT):
    yy = np.linspace(y0, y1, 200001)
    e = 0.5 * (1.0 + np.tanh((yy - yi) / eps))
    mu = e * MU_UP + (1 - e) * MU_LO if rule == 'arith' else 1.0 / (e / MU_UP + (1 - e) / MU_LO)
    I = np.concatenate([[0.0], np.cumsum(0.5 * (1 / mu[1:] + 1 / mu[:-1]) * np.diff(yy))])
    return U * np.interp(y, yy, I) / I[-1], U / I[-1]


def errs(u, ue, mask=None):
    d = (u - ue) / U
    if mask is not None:
        d = d[mask]
    return 100 * np.max(np.abs(d)), 100 * np.sqrt(np.mean(d ** 2))


def interp_at(y, f, y0):
    return float(np.interp(y0, y, f))


def save(fig, name):
    for ext in ('png', 'eps'):
        fig.savefig(os.path.join(IMG, f'{name}.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  wrote', os.path.join(IMG, name + '.{png,eps}'))


lines = []
def out(s=''):
    print(s)
    lines.append(s)


# ============================================================================
# SINGLE PHASE
# ============================================================================
single_runs = [('R29_v3_single_n64', 64), ('R29_v3_single_n128', 128)] if os.environ.get('COUETTE_SUFFIX', '') != '_harm_face' else [('R30_single_n64_face', 64), ('R30_single_n128_face', 128)]
single_wall = [('R29_v3_single_n64_wallaligned', 64), ('R29_v3_single_n128_wallaligned', 128)]

out('=' * 78)
out('SINGLE-PHASE COUETTE  (input_hydro2_single, mu = 10, rho = 100, U_top = 0.1, h = 3)')
out('errors in % of U_top;  "ghost-wall" = exact line between the ghost-cell centres')
out(f'{"run":34s} {"Ny":>4s} {"t_end":>6s} {"dU_last%":>9s} {"max%":>8s} {"L2%":>8s} {"maxGW%":>8s} {"|v|max":>9s}')
S = {}
for name, n in single_runs + single_wall:
    p = os.path.join(ROOT, name)
    if not os.path.isdir(os.path.join(p, 'output')):
        out(f'{name:34s} MISSING'); continue
    r = load_run(p)
    ue = exact_single(r['y'])
    ugw = exact_single(r['y'], r['ylo'] - r['dy'] / 2, r['yhi'] + r['dy'] / 2)
    mx, l2 = errs(r['u'], ue)
    mgw, _ = errs(r['u'], ugw)
    r.update(ue=ue, mx=mx, l2=l2)
    S[name] = r
    out(f'{name:34s} {n:4d} {r["t"]:6.1f} {100*r["du_last"]:9.2e} {mx:8.4f} {l2:8.4f} {mgw:8.2e} {np.max(np.abs(r["v"])):9.2e}')

base = [k for k, _ in single_runs if k in S]
if base:
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    yf = np.linspace(0, H, 400)
    ax.plot(exact_single(yf) / U, yf / H, 'k-', lw=1.8, label='Exact')
    for i, k in enumerate(base):
        r = S[k]; n = len(r['y'])
        ax.plot(r['u'] / U, r['y'] / H, MARKERS[i], mfc='none', ms=7, mew=1.3,
                markevery=(i * max(1, n // 16) // 3, max(1, n // 16)), label=f'Numerical, $N_y = {n}$')
    ax.set_xlabel('Velocity, $u/U_{top}$'); ax.set_ylabel('Height, $y/h$')
    ax.set_title('Single-Phase Couette Velocity Profile')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(loc='upper left')
    save(fig, 'Couette_Single_velocity_profile')

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    for i, k in enumerate(base):
        r = S[k]; n = len(r['y'])
        ax.plot(100 * (r['u'] - r['ue']) / U, r['y'] / H, '-', marker=MARKERS[i], mfc='none', ms=7,
                mew=1.3, lw=1.0, markevery=(i * max(1, n // 16) // 3, max(1, n // 16)), label=f'Numerical, $N_y = {n}$')
    ax.axvline(0, color='k', lw=1.0)
    ax.set_xlabel('Error, $(u - u_{exact})/U_{top}$ [%]'); ax.set_ylabel('Height, $y/h$')
    ax.set_title('Single-Phase Couette Velocity Error')
    ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(loc='best')
    save(fig, 'Couette_Single_error')


# ============================================================================
# TWO PHASE
# ============================================================================
multi_runs = [  # name, Ny, eps
    ('R29_v3_multi_n64_eps0.1', 64, 0.1),
    ('R29_v3_multi_n128_eps0.1', 128, 0.1),
    ('R29_v3_multi_n256_eps0.1', 256, 0.1),
    ('R29_v3_multi_n128_eps0.05', 128, 0.05),
    ('R29_v3_multi_n256_eps0.025', 256, 0.025),
    ('R29_v3_multi_n64_eps0.1_deckIC', 64, 0.1),
    # sensitivity only (not in the figures): conservative face-stress viscous form, visc_flux_form = 1
    ('R29_v3_multi_n64_eps0.1_vff', 64, 0.1),
    ('R29_v3_multi_n128_eps0.05_vff', 128, 0.05),
    ('R29_v3_multi_n256_eps0.025_vff', 256, 0.025),
    # harmonic face viscosity: visc_flux_form = 1, visc_harmonic = 1 (added 2026-10-29)
    ('R29_v3_multi_n64_eps0.1_harm', 64, 0.1),
    ('R29_v3_multi_n128_eps0.05_harm', 128, 0.05),
    ('R29_v3_multi_n256_eps0.025_harm', 256, 0.025),
    # harmonic + domain shifted by dy/2 so the Dirichlet ghost centres sit on the plates
    ('R29_v3_multi_n64_eps0.1_harm_wallaligned', 64, 0.1),
    ('R29_v3_multi_n128_eps0.05_harm_wallaligned', 128, 0.05),
    # harmonic + walls on the face (momentum.bc.dirichlet_at_face = 2, the Hydro2 default from 2026-10-29)
    ('R29_v3_multi_n64_eps0.1_harm_face', 64, 0.1),
    ('R29_v3_multi_n128_eps0.05_harm_face', 128, 0.05),
    ('R29_v3_multi_n256_eps0.025_harm_face', 256, 0.025),
]
SUF = os.environ.get('COUETTE_SUFFIX', '')   # '_harm': figures from the harmonic-viscosity runs
out()
out('=' * 78)
out('TWO-PHASE COUETTE  (input_hydro2_multi: lower mu1 = 1, rho = 50; upper mu0 = 2, rho = 100; eps = tanh width)')
_, tau_ex = exact_two_layer(np.array([0.0]))
u_int_ex = float(exact_two_layer(np.array([Y_INT]))[0][0])
out(f'exact: u(h/2)/U = {u_int_ex/U:.6f},  tau = {tau_ex:.6e} Pa;  band = |y - h/2| < 3 eps;  errors in % of U_top')
hdr = (f'{"run":32s} {"Ny":>4s} {"eps":>6s} {"t_end":>6s} {"dU%":>8s} {"max%":>7s} {"L2%":>7s} '
       f'{"maxOut%":>8s} {"L2Out%":>7s} {"maxIn%":>7s} {"uInt%":>7s} {"tauLvl%":>8s} {"tauSpr%":>8s} {"tauSprOut%":>10s}')
out(hdr)
M = {}
diag = []
for name, n, eps in multi_runs:
    p = os.path.join(ROOT, name)
    if not os.path.isdir(os.path.join(p, 'output')):
        out(f'{name:32s} MISSING'); continue
    r = load_run(p)
    y, u, eta, dy = r['y'], r['u'], r['eta'], r['dy']
    ue, _ = exact_two_layer(y)
    band = np.abs(y - Y_INT) < 3 * eps
    mx, l2 = errs(u, ue); mxo, l2o = errs(u, ue, ~band); mxi, _ = errs(u, ue, band)
    uI = interp_at(y, u, Y_INT); eI = 100 * (uI - u_int_ex) / U
    mu = 1.0 / (eta / MU_UP + (1 - eta) / MU_LO) if ('_harm' in name) else eta * MU_UP + (1 - eta) * MU_LO
    tau = mu[1:-1] * (u[2:] - u[:-2]) / (2 * dy); yt_ = y[1:-1]
    lvl = 100 * (np.mean(tau) / tau_ex - 1)
    spr = 100 * (tau.max() - tau.min()) / tau_ex
    bo = np.abs(yt_ - Y_INT) >= 3 * eps
    spro = 100 * (tau[bo].max() - tau[bo].min()) / tau_ex
    # decomposition
    y0g, y1g = r['ylo'] - dy / 2, r['yhi'] + dy / 2
    ugw, _ = exact_two_layer(y, y0g, y1g)
    udf, tdf = exact_diffuse(y, eps, y0g, y1g, 'arith')
    uhm, _ = exact_diffuse(y, eps, y0g, y1g, 'harm')
    udf0, _ = exact_diffuse(y, eps, 0.0, H, 'arith')
    r.update(ue=ue, tau=tau, yt=yt_, eps=eps, band=band, mx=mx, mxo=mxo, uI=uI, udf=udf)
    M[name] = r
    out(f'{name:32s} {n:4d} {eps:6.3f} {r["t"]:6.1f} {100*r["du_last"]:8.1e} {mx:7.3f} {l2:7.3f} '
        f'{mxo:8.3f} {l2o:7.3f} {mxi:7.3f} {eI:7.3f} {lvl:8.3f} {spr:8.3f} {spro:10.3f}')
    ob = ~band
    diag.append((name, n, eps, errs(u, ugw, ob)[0], errs(u, udf)[0], errs(udf, ue, ob)[0], errs(udf0, ue, ob)[0],
                 errs(uhm, ugw, ob)[0], 100 * (tdf / tau_ex - 1), errs(eta, 0.5 * (1 + np.tanh((y - Y_INT) / eps)))[0] * U / 100))
out()
out('decomposition (max % of U_top; [out] = outside the band only):')
out('  A num-vs-sharp(ghost walls)[out] | B num-vs-diffuse-arith(ghost walls)[all y] | C diffuse-arith(ghost walls)-vs-sharp(faces)[out] |')
out('  D diffuse-arith(faces)-vs-sharp(faces)[out] | E diffuse-harmonic(ghost walls)-vs-sharp(ghost walls)[out] | F tau_diffuse-arith level % | G max|eta - eta_IC|')
out(f'  {"run":32s} {"Ny":>4s} {"eps":>6s} {"A":>8s} {"B":>8s} {"C":>8s} {"D":>8s} {"E":>8s} {"F":>8s} {"G":>9s}')
for d in diag:
    out(f'  {d[0]:32s} {d[1]:4d} {d[2]:6.3f} {d[3]:8.3f} {d[4]:8.4f} {d[5]:8.3f} {d[6]:8.3f} {d[7]:8.4f} {d[8]:8.3f} {d[9]:9.2e}')

# deck-time snapshot (deck stop_time = 120 s) from the ramp-IC run
k = 'R29_v3_multi_n64_eps0.1_deckIC'
if k in M:
    f120 = [f for f in M[k]['files'] if os.path.basename(f) == '120000cell']
    if f120:
        ds = yt.load(f120[0]); g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        u120 = np.array(g['velocityx'])[:, :, 0].mean(0); y120 = np.array(g['y'])[0, :, 0]
        out(f'\n  deck-IC run at the deck stop_time t = {float(ds.current_time):.0f} s: '
            f'max err = {errs(u120, exact_two_layer(y120)[0])[0]:.3f} %  '
            f'(vs t = {M[k]["t"]:.0f} s: {M[k]["mx"]:.3f} %;  '
            f'max|u120 - u_end|/U = {100*np.max(np.abs(u120 - M[k]["u"]))/U:.3f} %)')

with open(os.path.join(IMG, 'Couette_R29_errors.txt'), 'w') as fh:
    fh.write('\n'.join(lines) + '\n')
print('  wrote', os.path.join(IMG, 'Couette_R29_errors.txt'))

seq = [k + SUF for k in ('R29_v3_multi_n64_eps0.1', 'R29_v3_multi_n128_eps0.05', 'R29_v3_multi_n256_eps0.025') if k + SUF in M]
if seq:
    yf = np.linspace(0, H, 1001)
    uf, _ = exact_two_layer(yf)
    # (a) velocity profile
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    r0 = M[seq[0]]
    ax.axhspan((Y_INT - 3 * r0['eps']) / H, (Y_INT + 3 * r0['eps']) / H, color='0.88', zorder=0,
               label=rf'Diffuse band, $\epsilon = {r0["eps"]:g}$')
    ax.plot(uf / U, yf / H, 'k-', lw=1.8, label='Exact')
    for i, k in enumerate(seq):
        r = M[k]; n = len(r['y'])
        ax.plot(r['u'] / U, r['y'] / H, MARKERS[i], mfc='none', ms=7, mew=1.3, markevery=(i * max(1, n // 16) // 3, max(1, n // 16)),
                label=rf'Numerical, $N_y = {n}$, $\epsilon = {r["eps"]:g}$')
    ax.set_xlabel('Velocity, $u/U_{top}$'); ax.set_ylabel('Height, $y/h$')
    ax.set_title('Two-Phase Couette Velocity Profile')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(loc='upper left', fontsize=12)
    save(fig, 'Couette_Multi_velocity_profile')

    # (b) error profile
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    for i, k in enumerate(seq):
        r = M[k]; n = len(r['y'])
        ax.plot(100 * (r['u'] - r['ue']) / U, r['y'] / H, '-', marker=MARKERS[i], mfc='none', ms=7, mew=1.3,
                lw=1.0, markevery=(i * max(1, n // 16) // 3, max(1, n // 16)), label=rf'Numerical, $N_y = {n}$, $\epsilon = {r["eps"]:g}$')
    ax.axvline(0, color='k', lw=1.0); ax.axhline(0.5, color='k', ls=':', lw=1.0)
    ax.set_xlabel('Error, $(u - u_{exact})/U_{top}$ [%]'); ax.set_ylabel('Height, $y/h$')
    ax.set_title('Two-Phase Couette Velocity Error')
    ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(loc='best')
    save(fig, 'Couette_Multi_error')

    # (c) shear-stress balance
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.axvline(1.0, color='k', lw=1.8, label='Exact')
    for i, k in enumerate(seq):
        r = M[k]; n = len(r['yt'])
        ax.plot(r['tau'] / tau_ex, r['yt'] / H, marker=MARKERS[i], ls='-', lw=0.8, color=f'C{i}', mfc='none', ms=7, mew=1.3,
                markevery=(i * max(1, n // 24) // 3, max(1, n // 24)), label=rf'Numerical, $N_y = {n+2}$, $\epsilon = {r["eps"]:g}$')
    ax.set_xlabel(r'Shear Stress, $\mu\,(du/dy)\,/\,\tau_{exact}$'); ax.set_ylabel('Height, $y/h$')
    ax.set_title('Two-Phase Couette Shear-Stress Balance')
    ax.set_ylim(0, 1); ax.grid(alpha=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=12, frameon=False)
    save(fig, 'Couette_Multi_shear_stress')

    # (d) convergence of the errors
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for lab, ks, mk in ((r'$\epsilon \propto \Delta y$', seq, 'o'),
                        (r'$\epsilon = 0.1$ fixed', [k for k in ('R29_v3_multi_n64_eps0.1', 'R29_v3_multi_n128_eps0.1',
                                                                  'R29_v3_multi_n256_eps0.1') if k in M], 's')):
        if not ks: continue
        N = np.array([len(M[k]['y']) for k in ks])
        ax.loglog(N, [M[k]['mxo'] for k in ks], '-' + mk, mfc='none', ms=8, mew=1.3, label=f'Outside band, {lab}')
        ax.loglog(N, [abs(100 * (M[k]['uI'] - u_int_ex) / U) for k in ks], '--' + mk, mfc='none', ms=8, mew=1.3,
                  label=f'Interface $u(h/2)$, {lab}')
    Nr = np.array([64, 256]); ax.loglog(Nr, 0.8 * (Nr / 64.0) ** -1, 'k:', lw=1.2, label='First order')
    ax.set_xlabel('Cells Across Channel, $N_y$'); ax.set_ylabel('Max Error [%]')
    ax.set_title('Two-Phase Couette Error Convergence')
    ax.grid(alpha=0.3, which='both'); ax.legend(loc='best', fontsize=11)
    save(fig, 'Couette_Multi_convergence')
