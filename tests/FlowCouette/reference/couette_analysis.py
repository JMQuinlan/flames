"""
Single-phase planar Couette analysis (no-slip plates, steady-state).

Reference for input_hydro2_single:
  bottom wall (y = 0) : stationary, u = U_bot
  top    wall (y = H) : moving,     u = U_top
  pure single fluid (eta forced to 0, only phase 1 active).

Steady-state momentum balance + no-slip BCs => linear profile:
    u(y) = U_bot + (U_top - U_bot) * y / H
    du/dy = (U_top - U_bot) / H              (uniform)
    shear stress tau = mu * du/dy            (uniform)

This script extracts u(y) from the last plotfile, computes the analytical
linear profile, and reports L2 / Linf error plus a uniform-shear-rate
diagnostic.
"""

import yt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

yt.funcs.mylog.setLevel(40)

# ============================================================================
# USER CONFIGURATION
# ============================================================================

case_name        = 'Couette_Single'
# Output dir: the aggregator passes it as argv[1]; otherwise use a default.
# Toggle the default between INCLINE (/mmfs1, active) and DESKTOP (comment swap).
import sys
_OUT_INCLINE = r'/mmfs1/home/ttryon/flames/bin/tests/FlowCouette/UNIT_TEST_2D/output_single'
_OUT_DESKTOP = r'../../../bin/tests/FlowCouette/UNIT_TEST_2D/output_single'
amrex_output_dir = sys.argv[1] if len(sys.argv) > 1 else _OUT_INCLINE  # -> _OUT_DESKTOP for desktop

# Geometry (must match input_hydro2_single)
y_min  = 0.0
y_max  = 3.0
x_min  = 0.0
x_max  = 0.1875
H      = y_max - y_min

# Physical parameters (must match input_hydro2_single)
mu    = 10.0      # dynamic viscosity (phase 1 since eta = 0 everywhere)
rho   = 100.0
nu    = mu / rho  # kinematic viscosity = 0.1 m^2/s

# Plate velocities (must match Dirichlet momentum BCs in input file)
U_bot = 0.0
U_top = 0.1

# Slice location for 1D profile extraction
x_slice = 0.5 * (x_min + x_max)

# Output
images_dir = './Images'
os.makedirs(images_dir, exist_ok=True)


# ============================================================================
# ANALYTICAL HELPERS
# ============================================================================

def couette_single(y, U_top, U_bot, H):
    """Steady-state single-phase Couette profile."""
    return U_bot + (U_top - U_bot) * y / H


def shear_rate(U_top, U_bot, H):
    """Steady-state uniform shear rate du/dy."""
    return (U_top - U_bot) / H


# ============================================================================
# LOAD AMREX DATA
# ============================================================================

print("=" * 70)
print("SINGLE-PHASE COUETTE ANALYSIS  (input_hydro2_single)")
print("=" * 70)
print(f"\nProblem setup:")
print(f"  Domain     : y = [{y_min}, {y_max}] m,  x = [{x_min}, {x_max}] m")
print(f"  Fluid      : rho = {rho}, mu = {mu},  nu = {nu}")
print(f"  No-slip    : u(0) = {U_bot},  u(H) = {U_top}")
print(f"  Steady SS  : du/dy = {shear_rate(U_top, U_bot, H):.6f} 1/s (uniform)")
print(f"  tau_diff   : H^2 / (pi^2 * nu) = {H * H / (np.pi ** 2 * nu):.2f} s")

print(f"\nLoading {amrex_output_dir}...")
plot_files = sorted(
    os.path.join(amrex_output_dir, d)
    for d in os.listdir(amrex_output_dir)
    if os.path.isdir(os.path.join(amrex_output_dir, d)) and d.endswith('cell')
)
if not plot_files:
    print(f"  ERROR: no *cell directories found.")
    raise SystemExit(1)

last_plot = plot_files[-1]
ds        = yt.load(last_plot)
sim_time  = float(ds.current_time)
print(f"  using {os.path.basename(last_plot)}, t_sim = {sim_time:.4f} s")


# ============================================================================
# EXTRACT NUMERICAL PROFILE
# ============================================================================

# z midplane: 0.0 in 2D (domain z = [0,0]); the central z-plane in a 3D
# z-extension run (z=0 would be a boundary face).
zmid = lambda d: float(0.5 * (d.domain_left_edge[2] + d.domain_right_edge[2]))
ray = ds.ray(
    ds.arr([x_slice, y_min, zmid(ds)], 'code_length'),
    ds.arr([x_slice, y_max, zmid(ds)], 'code_length'),
)
order = np.argsort(ray['y'])
y_num = np.array(ray['y'][order])
u_num = np.array(ray['velocityx'][order])
v_num = np.array(ray['velocityy'][order])
try:
    eta_num = np.array(ray['eta'][order])
    has_eta = True
except Exception:
    has_eta = False

print(f"\n  extracted {len(y_num)} points along x = {x_slice} m")
print(f"  u range: [{np.min(u_num):.6f}, {np.max(u_num):.6f}] m/s")
print(f"  |v|max : {np.max(np.abs(v_num)):.3e} m/s (should be ~0)")

# No-slip enforcement check
print(f"\n  No-slip BC enforcement (at the first/last cells of the ray):")
print(f"    u(y ~= 0)  = {u_num[0]:.6e}   (target {U_bot})")
print(f"    u(y ~= H)  = {u_num[-1]:.6e}   (target {U_top})")


# ============================================================================
# ANALYTICAL PROFILE + ERROR
# ============================================================================

y_fine        = np.linspace(y_min, y_max, 500)
u_ana_fine    = couette_single(y_fine, U_top, U_bot, H)
u_ana_at_num  = couette_single(y_num,  U_top, U_bot, H)

err  = u_num - u_ana_at_num
L2   = np.sqrt(np.mean(err ** 2))
Linf = np.max(np.abs(err))
print(f"\n  L2 error  = {L2:.6e}")
print(f"  Linf err  = {Linf:.6e}")
print(f"  mean |err|= {np.mean(np.abs(err)):.6e}")

# Numerical shear rate diagnostic.
dudy_num = np.gradient(u_num, y_num)
print(f"\n  Numerical du/dy stats:")
print(f"    mean   : {np.mean(dudy_num):.6f}  (analytical {shear_rate(U_top, U_bot, H):.6f})")
print(f"    stddev : {np.std(dudy_num):.6e}    (should be ~0 -> uniform shear)")


# ============================================================================
# PLOTS
# ============================================================================

# Percent error normalized by the plate-velocity scale (U_top - U_bot).
# Using percent-of-velocity-difference (rather than dividing by u_ana which
# vanishes at y=0) keeps the metric finite and physically meaningful.
U_scale  = max(abs(U_top - U_bot), 1e-30)
pct_err  = 100.0 * err / U_scale
L2_pct   = np.sqrt(np.mean(pct_err ** 2))
Linf_pct = np.max(np.abs(pct_err))

# Velocity profile (final time)
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(u_ana_fine, y_fine, 'b-', lw=2.0, label=f'Analytical (linear)')
ax.plot(u_num,      y_num,  'r--', lw=1.4, label='Numerical', alpha=0.85)
ax.set_xlabel('u (m/s)', fontsize=12)
ax.set_ylabel('Height (m)', fontsize=12)
ax.set_title('Single-phase Couette velocity profile',
             fontsize=13, fontweight='bold')
ax.set_ylim([y_min, y_max])
ax.legend(fontsize=10, loc='best')
ax.grid(alpha=0.3)
plt.tight_layout()
out = os.path.join(images_dir, f'{case_name}_velocity_profile')
fig.savefig(out + '.png', dpi=300, bbox_inches='tight')
fig.savefig(out + '.eps',           bbox_inches='tight')
plt.close(fig)
print(f"\n  wrote {out}.png / .eps")

# Shear rate
fig, ax = plt.subplots(figsize=(8, 7))
dudy_ana = np.full_like(y_fine, shear_rate(U_top, U_bot, H))
ax.plot(dudy_ana, y_fine, 'b-', lw=2.0, label='Analytical (uniform)')
ax.plot(dudy_num, y_num,  'r--', lw=1.4, label='Numerical (centered diff)', alpha=0.85)
ax.set_xlabel('du/dy (1/s)', fontsize=12)
ax.set_ylabel('Height (m)', fontsize=12)
ax.set_title('Shear rate', fontsize=13, fontweight='bold')
ax.set_ylim([y_min, y_max])
ax.legend(fontsize=10, loc='upper center')
ax.grid(alpha=0.3)
plt.tight_layout()
out = os.path.join(images_dir, f'{case_name}_shear_rate')
fig.savefig(out + '.png', dpi=300, bbox_inches='tight')
fig.savefig(out + '.eps',           bbox_inches='tight')
plt.close(fig)
print(f"  wrote {out}.png / .eps")

# Percent error
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(pct_err, y_num, 'k-', lw=1.5)
ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Velocity Difference (%)', fontsize=12)
ax.set_ylabel('Height (m)', fontsize=12)
ax.set_title('Couette velocity error', fontsize=13, fontweight='bold')
ax.set_ylim([y_min, y_max])
ax.grid(alpha=0.3)
plt.tight_layout()
out = os.path.join(images_dir, f'{case_name}_error')
fig.savefig(out + '.png', dpi=300, bbox_inches='tight')
fig.savefig(out + '.eps',           bbox_inches='tight')
plt.close(fig)
print(f"  wrote {out}.png / .eps")

if has_eta:
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(eta_num, y_num, 'g-', lw=1.6)
    ax.set_xlabel('eta', fontsize=12)
    ax.set_ylabel('Height (m)', fontsize=12)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([y_min, y_max])
    ax.set_title('Phase fraction', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(images_dir, f'{case_name}_eta')
    fig.savefig(out + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(out + '.eps',           bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out}.png / .eps")


# ============================================================================
# TRANSIENT GIF (velocity profile vs analytical at every plotfile)
# ============================================================================

print(f"\n  building transient GIF from {len(plot_files)} plotfiles...")
frame_data = []
for pf in plot_files:
    try:
        ds_f      = yt.load(pf)
        ray_f     = ds_f.ray(
            ds_f.arr([x_slice, y_min, zmid(ds_f)], 'code_length'),
            ds_f.arr([x_slice, y_max, zmid(ds_f)], 'code_length'),
        )
        order_f   = np.argsort(ray_f['y'])
        y_f       = np.array(ray_f['y'][order_f])
        u_f       = np.array(ray_f['velocityx'][order_f])
        t_f       = float(ds_f.current_time)
        frame_data.append((t_f, y_f, u_f))
    except Exception as exc:
        print(f"    [skip] {os.path.basename(pf)}: {exc}")

# Sort by physical time (filenames are usually monotonic, but be safe).
frame_data.sort(key=lambda d: d[0])

x_lo = min(U_bot, U_top) - 0.02 * U_scale
x_hi = max(U_bot, U_top) + 0.02 * U_scale

fig_g, ax_g = plt.subplots(figsize=(8, 7))
ax_g.plot(u_ana_fine, y_fine, 'b-', lw=2.0, label='Analytical (steady)')
(line_num,) = ax_g.plot([], [], 'r--', lw=1.6, label='Numerical')
time_text  = ax_g.text(0.04, 0.96, '', transform=ax_g.transAxes,
                       fontsize=12, fontweight='bold', va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
ax_g.set_xlim([x_lo, x_hi])
ax_g.set_ylim([y_min, y_max])
ax_g.set_xlabel('u (m/s)', fontsize=12)
ax_g.set_ylabel('Height (m)', fontsize=12)
ax_g.set_title('Single-phase Couette transient', fontsize=13, fontweight='bold')
ax_g.legend(fontsize=10, loc='lower right')
ax_g.grid(alpha=0.3)

def _gif_init():
    line_num.set_data([], [])
    time_text.set_text('')
    return line_num, time_text

def _gif_update(i):
    t_i, y_i, u_i = frame_data[i]
    line_num.set_data(u_i, y_i)
    time_text.set_text(f't = {t_i:.3f} s')
    return line_num, time_text

if frame_data:
    anim = animation.FuncAnimation(
        fig_g, _gif_update, init_func=_gif_init,
        frames=len(frame_data), interval=120, blit=True,
    )
    gif_path = os.path.join(images_dir, f'{case_name}_transient.gif')
    try:
        anim.save(gif_path, writer=animation.PillowWriter(fps=8))
        print(f"  wrote {gif_path}  ({len(frame_data)} frames)")
    except Exception as exc:
        print(f"  [warn] gif save failed ({exc}); try `pip install pillow`.")
else:
    print("  [warn] no usable frames; gif not written.")
plt.close(fig_g)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
