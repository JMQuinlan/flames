"""
Overlay 1D normal-direction slices from rotated Garrick Riemann problems.

For each angle theta in ANGLES, this loads the corresponding AMReX output, casts
a ray along n = (cos theta, sin theta) through the origin, and plots density,
normal velocity (u_n = vx*cos theta + vy*sin theta), and pressure against the
signed normal coordinate s = x*cos theta + y*sin theta.

If the hydro2 solver couples x and y correctly, all curves must overlay.
"""

import yt
import numpy as np
import matplotlib.pyplot as plt
import os

yt.funcs.mylog.setLevel(40)

# ============================================================================
# USER CONFIGURATION
# ============================================================================

ANGLES = [0, 15, 30, 45, 60, 75, 90]

# Output directories are written by hydro2 under bin/. The script lives at
# tests/FlowRiemannUnitTests/Angled/reference/, so go up four levels.
output_dir_template = r'..\..\..\..\bin\tests\FlowRiemannUnitTests\Angled\output_Garrick_Angled_{:02d}'

output_folder = r'./Images'
output_filename = 'Garrick_Angled_overlay'

# Half-length of the sampling ray along n. Must satisfy
#   S_MAX * max(|cos theta|, |sin theta|) <= 1
# to stay inside the [-1, 1]^2 domain. With unit max that's S_MAX <= 1.
S_MAX = 0.99

# ============================================================================

print("=" * 60)
print("OVERLAYING ROTATED GARRICK RIEMANN PROFILES")
print("=" * 60)

fig, axes = plt.subplots(3, 1, figsize=(12, 14))
colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(ANGLES)))

t_final = None
loaded = 0

for color, theta_deg in zip(colors, ANGLES):
    amrex_output_dir = output_dir_template.format(theta_deg)

    if not os.path.isdir(amrex_output_dir):
        print(f"  [skip] theta={theta_deg:>2d}deg: dir not found ({amrex_output_dir})")
        continue

    plot_files = sorted(
        os.path.join(amrex_output_dir, item)
        for item in os.listdir(amrex_output_dir)
        if os.path.isdir(os.path.join(amrex_output_dir, item)) and item.endswith('cell')
    )
    if not plot_files:
        print(f"  [skip] theta={theta_deg:>2d}deg: no *cell directories")
        continue

    last_plot = plot_files[-1]
    ds = yt.load(last_plot)
    if t_final is None:
        t_final = float(ds.current_time)

    theta = np.deg2rad(theta_deg)
    cth, sth = np.cos(theta), np.sin(theta)

    ray_start = ds.arr([-S_MAX * cth, -S_MAX * sth, 0.0], 'code_length')
    ray_end   = ds.arr([ S_MAX * cth,  S_MAX * sth, 0.0], 'code_length')
    ray = ds.ray(ray_start, ray_end)

    x   = np.array(ray['x'])
    y   = np.array(ray['y'])
    vx  = np.array(ray['velocityx'])
    vy  = np.array(ray['velocityy'])
    p   = np.array(ray['pressure'])
    rho = np.array(ray['density'])

    s   = x * cth + y * sth
    u_n = vx * cth + vy * sth

    order = np.argsort(s)
    s, rho, u_n, p = s[order], rho[order], u_n[order], p[order]

    label = f'theta = {theta_deg:>2d} deg'
    axes[0].plot(s, rho, '-', color=color, linewidth=1.5, label=label)
    axes[1].plot(s, u_n, '-', color=color, linewidth=1.5, label=label)
    axes[2].plot(s, p,   '-', color=color, linewidth=1.5, label=label)

    print(f"  [ok]   theta={theta_deg:>2d}deg: {len(s)} samples, "
          f"t = {float(ds.current_time):.6f}, "
          f"rho in [{rho.min():.4f}, {rho.max():.4f}], "
          f"p in [{p.min():.4e}, {p.max():.4e}]")
    loaded += 1

if loaded == 0:
    print("\nNo datasets loaded - check that the simulations have run.")
    raise SystemExit(1)

title = (f'Rotated Garrick Riemann - overlay of theta in {ANGLES} deg, '
         f't = {t_final:.4f} s')
axes[0].set_title(title, fontsize=14, fontweight='bold')

axes[0].set_xlabel(r'normal coordinate $s = x\cos\theta + y\sin\theta$', fontsize=12)
axes[0].set_ylabel(r'Density (kg/m$^3$)', fontsize=12)
axes[0].legend(fontsize=9, ncol=2)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([-S_MAX, S_MAX])

axes[1].set_xlabel(r'normal coordinate $s$', fontsize=12)
axes[1].set_ylabel(r'Normal velocity $u_n = v_x\cos\theta + v_y\sin\theta$ (m/s)',
                   fontsize=12)
axes[1].legend(fontsize=9, ncol=2)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([-S_MAX, S_MAX])

axes[2].set_xlabel(r'normal coordinate $s$', fontsize=12)
axes[2].set_ylabel(r'Pressure (Pa)', fontsize=12)
axes[2].legend(fontsize=9, ncol=2)
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim([-S_MAX, S_MAX])

plt.tight_layout()

os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, output_filename)
plt.savefig(output_path + '.png', format='png', dpi=300, bbox_inches='tight')
plt.savefig(output_path + '.eps', format='eps', bbox_inches='tight')

print(f"\nOverlay saved:")
print(f"  {output_path}.png")
print(f"  {output_path}.eps")

plt.show()

print("=" * 60)
print(f"Done. {loaded}/{len(ANGLES)} angles plotted.")
