"""
Two-fluid planar Couette analysis (steady-state, classical 2-layer).

Reference for input_hydro2_multi: two fluid layers stacked vertically with
no-slip plates at y=0 (u=0) and y=H (u=U_top).  Phase 1 (lower fluid B,
viscosity mu_B) occupies y in [0, y_int]; phase 0 (upper fluid A, viscosity
mu_A) occupies y in [y_int, H].

Steady-state momentum balance gives piecewise-linear u(y) with shear stress
tau = mu * du/dy continuous across the interface:

    A1 = U_top / [ y_int + (mu_B / mu_A) * (H - y_int) ]    # du/dy in lower layer
    A2 = (mu_B / mu_A) * A1                                 # du/dy in upper layer
    u(y) = A1 * y                                  for y <= y_int
    u(y) = A1 * y_int + A2 * (y - y_int)           for y >  y_int

This script extracts u(y) from the last plotfile, computes the analytical
profile, and reports the L2 / Linf error plus a shear-stress continuity
diagnostic at the interface.
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

case_name        = 'Couette_Multi'
amrex_output_dir = r'../../../bin/tests/FlowCouette/output_multi'

# Geometry (must match input_hydro2_multi)
y_min  = 0.0
y_max  = 3.0
x_min  = 0.0
x_max  = 0.1875
y_int  = 1.5       # interface height
H      = y_max - y_min

# Physical parameters (must match input_hydro2_multi)
# Phase 0 (upper, fluid A)
mu_A   = 2.0
rho_A  = 100.0
# Phase 1 (lower, fluid B)
mu_B   = 1.0
rho_B  = 50.0

# Boundary conditions
U_bot  = 0.0
U_top  = 0.1

# Slice location for 1D profile extraction
x_slice = 0.5 * (x_min + x_max)

# Output
images_dir = './Images'
os.makedirs(images_dir, exist_ok=True)


# ============================================================================
# ANALYTICAL HELPERS
# ============================================================================

def couette_2layer(y, U_top, U_bot, mu_A, mu_B, y_int, H):
    """Classical 2-layer Couette closed-form.

    Stress continuity:  mu_B * A1 = mu_A * A2
    Velocity continuity at y_int.
    Boundary conditions: u(0) = U_bot, u(H) = U_top.
    """
    delta_U = U_top - U_bot
    A1 = delta_U / (y_int + (mu_B / mu_A) * (H - y_int))
    A2 = (mu_B / mu_A) * A1
    u  = np.where(
        y <= y_int,
        U_bot + A1 * y,
        U_bot + A1 * y_int + A2 * (y - y_int),
    )
    return u, A1, A2


# ============================================================================
# LOAD AMREX DATA
# ============================================================================

print("=" * 70)
print("TWO-FLUID PLANAR COUETTE ANALYSIS  (input_hydro2_multi)")
print("=" * 70)
print(f"\nProblem setup:")
print(f"  Domain     : y = [{y_min}, {y_max}] m,  x = [{x_min}, {x_max}] m")
print(f"  Interface  : y_int = {y_int} m")
print(f"  Lower (B)  : rho = {rho_B}, mu = {mu_B}, nu = {mu_B/rho_B}")
print(f"  Upper (A)  : rho = {rho_A}, mu = {mu_A}, nu = {mu_A/rho_A}")
print(f"  Plates     : u(0) = {U_bot},  u(H) = {U_top}")

# Steady-state expectations
_, A1_exp, A2_exp = couette_2layer(np.array([y_int]), U_top, U_bot,
                                   mu_A, mu_B, y_int, H)
print(f"  Expected shear rates:")
print(f"    A1 (lower) = du/dy = {A1_exp:.6f} 1/s")
print(f"    A2 (upper) = du/dy = {A2_exp:.6f} 1/s")
print(f"  Stress check: mu_B * A1 = {mu_B * A1_exp:.6e}")
print(f"                mu_A * A2 = {mu_A * A2_exp:.6e}  (should match)")
print(f"  tau_diff lower : H^2 / nu_B = {H * H / (mu_B / rho_B):.2f} s")
print(f"  tau_diff upper : H^2 / nu_A = {H * H / (mu_A / rho_A):.2f} s")

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

ray = ds.ray(
    ds.arr([x_slice, y_min, 0.0], 'code_length'),
    ds.arr([x_slice, y_max, 0.0], 'code_length'),
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


# ============================================================================
# ANALYTICAL PROFILE + ERROR
# ============================================================================

y_fine        = np.linspace(y_min, y_max, 500)
u_ana_fine, A1, A2 = couette_2layer(y_fine, U_top, U_bot, mu_A, mu_B, y_int, H)
u_ana_at_num, _, _ = couette_2layer(y_num,  U_top, U_bot, mu_A, mu_B, y_int, H)

err  = u_num - u_ana_at_num
L2   = np.sqrt(np.mean(err ** 2))
Linf = np.max(np.abs(err))
print(f"\n  L2 error  = {L2:.6e}")
print(f"  Linf err  = {Linf:.6e}")
print(f"  mean |err|= {np.mean(np.abs(err)):.6e}")

# Numerical shear rate diagnostic (one-sided at interface from below / above).
dudy_num = np.gradient(u_num, y_num)
mask_lower = y_num < y_int
mask_upper = y_num > y_int
A1_num = float(np.mean(dudy_num[mask_lower])) if mask_lower.any() else np.nan
A2_num = float(np.mean(dudy_num[mask_upper])) if mask_upper.any() else np.nan
print(f"\n  Numerical mean du/dy:")
print(f"    lower layer : {A1_num:.6f}  (analytical {A1:.6f})")
print(f"    upper layer : {A2_num:.6f}  (analytical {A2:.6f})")
print(f"  Numerical stress continuity:")
print(f"    mu_B * A1_num = {mu_B * A1_num:.6e}")
print(f"    mu_A * A2_num = {mu_A * A2_num:.6e}  (should match)")


# ============================================================================
# PLOTS
# ============================================================================

# Percent error normalized by the plate-velocity scale (U_top - U_bot).
# Keeps the metric finite at y=0 where u_ana = 0.
U_scale  = max(abs(U_top - U_bot), 1e-30)
pct_err  = 100.0 * err / U_scale
L2_pct   = np.sqrt(np.mean(pct_err ** 2))
Linf_pct = np.max(np.abs(pct_err))

# Velocity profile
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(u_ana_fine, y_fine, 'b-', lw=2.0, label='Analytical (2-layer Couette)')
ax.plot(u_num,      y_num,  'r--', lw=1.4, label='Numerical', alpha=0.85)
ax.set_xlabel('u (m/s)', fontsize=12)
ax.set_ylabel('Height (m)', fontsize=12)
ax.set_title('Two-fluid Couette velocity profile', fontsize=13, fontweight='bold')
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
dudy_ana = np.where(y_fine <= y_int, A1, A2)
ax.plot(dudy_ana, y_fine, 'b-', lw=2.0, label='Analytical')
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

# Stress continuity check (mu * du/dy should be flat across the domain)
fig, ax = plt.subplots(figsize=(8, 7))
mu_profile = np.where(y_num <= y_int, mu_B, mu_A)
tau_num    = mu_profile * dudy_num
tau_ana    = np.full_like(y_fine, mu_B * A1)
ax.plot(tau_ana, y_fine, 'b-', lw=2.0, label=f'Analytical')
ax.plot(tau_num, y_num,  'r--', lw=1.4, label='Numerical mu * du/dy', alpha=0.85)
ax.set_xlabel('shear stress tau (Pa)', fontsize=12)
ax.set_ylabel('Height (m)', fontsize=12)
ax.set_title('Shear stress continuity', fontsize=13, fontweight='bold')
ax.set_ylim([y_min, y_max])
ax.legend(fontsize=10, loc='best')
ax.grid(alpha=0.3)
plt.tight_layout()
out = os.path.join(images_dir, f'{case_name}_stress_continuity')
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
    ax.set_xlabel('eta (alpha of phase 0 / upper fluid A)', fontsize=12)
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
            ds_f.arr([x_slice, y_min, 0.0], 'code_length'),
            ds_f.arr([x_slice, y_max, 0.0], 'code_length'),
        )
        order_f   = np.argsort(ray_f['y'])
        y_f       = np.array(ray_f['y'][order_f])
        u_f       = np.array(ray_f['velocityx'][order_f])
        t_f       = float(ds_f.current_time)
        frame_data.append((t_f, y_f, u_f))
    except Exception as exc:
        print(f"    [skip] {os.path.basename(pf)}: {exc}")

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
ax_g.set_title('Two-fluid Couette transient', fontsize=13, fontweight='bold')
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
