"""Generate tests/FlowMarmottant/MarmPaper decks (coated 2 um bubble, RK3 + MUSCL2) and their slurm files.

One template, so the numerics, shell and BC blocks are identical across the five tests.  Physics numbers
come from Keller-Miksis + Marmottant sizing (~/claude_scratch/marmpaper/km_size.py, km_sub.py).
"""
import os, math
REPO = os.path.expanduser("~/Desktop/flames2")
DECKS = REPO + "/tests/FlowMarmottant/MarmPaper"
SLURM = REPO + "/TryonSlurmFiles/FlowMarmottant/FlowMarmPaper"
os.makedirs(DECKS, exist_ok=True); os.makedirs(SLURM, exist_ok=True)

R0 = 2.0e-6
P0 = 1.0e5
SIG_R0 = 1.0 * ((R0 / 1.96e-6) ** 2 - 1.0)           # Marmottant sigma(R0) = 0.0412 N/m
P_EQ = P0 + 2.0 * SIG_R0 / R0                         # equilibrium gas pressure 1.4123e5
F0 = 4.084e6                                          # KM linear natural frequency [Hz] (shell-stiffened)
W0 = 2 * math.pi * F0
# Box gain p_centre / drive_amp of the 20 R0 driven box (calibrated, ~/claude_scratch/marmpaper/gain, see README)
GAIN_LIN = float(os.environ.get("GAIN_LIN", "1.006"))  # 20 R0 box, f0/pi: centre 1.0064, mid-box 1.0072, phase 0.2 deg
GAIN_NL = float(os.environ.get("GAIN_NL", "1.107"))   # 20 R0 box, f0: centre 1.107, mid-box 1.089, phase 0.2 deg

# ---- _Mili decks: the same tests with every length and time x 1e4 (R0 = 2 cm) ----
# Pressures, densities, EOS, velocities, water mu and the thermal constants are unchanged.  kappa_s is scaled
# by S^2 to keep the dimensionless shell viscosity K* = kappa_s/(R0^2 sqrt(rho p_inf)) of the micro decks
# (7.2e-9 -> 0.72 kg/s, as in the old Sch20 *_Mili decks).  The Marmottant elastic shell (chi, sigma_break) and
# sigma_water keep their physical values, so at 2 cm they are inert (2 sigma/R <= 7 Pa): f0 drops to Minnaert.
S_MILI = 1.0e4
R0_MILI = R0 * S_MILI
F0_MILI = 163.128                                     # KM linear natural frequency at R0 = 2 cm [Hz] (~ Minnaert 163.09)
W0_MILI = 2 * math.pi * F0_MILI
P_EQ_MILI = P0 + 2.0 * SIG_R0 / R0_MILI               # 1.000041e5
GAIN_NL_MILI = float(os.environ.get("GAIN_NL_MILI", "0.970"))  # 20 R0 box at 2 f0_Mili = 326 Hz: centre 0.970, mid-box 0.974, phase -0.8 deg
                                                                # (liquid-only 32^3 run, ~/claude_scratch/mili/gain, 2026-11-08)

def fmt(x): return f"{x:.6e}"

def nscbc(faces, p_inf, L_ref, amp=0.0, omega=0.0, comment=""):
    s = []
    for f in faces:
        s += [f"{comment}nscbc.{f}.type     = outflow",
              f"{comment}nscbc.{f}.target_p = {fmt(p_inf)}",
              f"{comment}nscbc.{f}.sigma    = 0.6",
              f"{comment}nscbc.{f}.beta     = 0.5",
              f"{comment}nscbc.{f}.L_ref    = {fmt(L_ref)}"]
        if amp:
            s += [f"{comment}nscbc.{f}.drive_amp   = {fmt(amp)}",
                  f"{comment}nscbc.{f}.drive_omega = {fmt(omega)}"]
    return s

def bcs(octant, comment=""):
    c = comment
    L = []
    if octant:
        for fld in ("eta", "density", "energy"):
            for ax in "xyz":
                L.append(f"{c}{fld}.bc.type.{ax}lo = REFLECT_EVEN")
                L.append(f"{c}{fld}.bc.type.{ax}hi = " + ("neumann" if fld == "eta" else "nscbc_outflow"))
        L += [f"{c}momentum.bc.type.xlo = REFLECT_ODD  REFLECT_EVEN REFLECT_EVEN",
              f"{c}momentum.bc.type.xhi = nscbc_outflow nscbc_outflow nscbc_outflow",
              f"{c}momentum.bc.type.ylo = REFLECT_EVEN REFLECT_ODD  REFLECT_EVEN",
              f"{c}momentum.bc.type.yhi = nscbc_outflow nscbc_outflow nscbc_outflow",
              f"{c}momentum.bc.type.zlo = REFLECT_EVEN REFLECT_EVEN REFLECT_ODD",
              f"{c}momentum.bc.type.zhi = nscbc_outflow nscbc_outflow nscbc_outflow"]
    else:
        for fld in ("eta", "density", "energy"):
            for ax in "xyz":
                for s in ("lo", "hi"):
                    L.append(f"{c}{fld}.bc.type.{ax}{s} = " + ("neumann" if fld == "eta" else "nscbc_outflow"))
        for ax in "xyz":
            for s in ("lo", "hi"):
                L.append(f"{c}momentum.bc.type.{ax}{s} = nscbc_outflow nscbc_outflow nscbc_outflow")
    return L

def deck(name, title, header, *, p_inf, p_b, liquid_ic, max_level, stop, plot_dt, thermo_dt,
         drive=None, jet=None, full_only=False, half=None, n_oct=160, refine_box=None, mili=False):
    sc = S_MILI if mili else 1.0
    R0 = (R0_MILI if mili else globals()["R0"])
    f0 = F0_MILI if mili else F0
    if half is None: half = 100 * R0
    dx = half / n_oct / 2 ** max_level
    eps = 2.0 * dx
    L_ref = 3.125e-3 * half                          # L_ref/domain as in the 11_04 coated collapse deck
    amp, om = (drive if drive else (0.0, 0.0))
    o = []   # octant lines, f = full-domain lines (commented in octant decks)
    out = [f"# MarmPaper: {title}", "#"] + [f"# {l}" if l else "#" for l in header] + [
        "#",
    ] + ([
        "# COMMON (all MarmPaper decks): coated 2 um bubble, Marmottant shell chi 1.0 N/m, R_buck 0.98 R0",
        f"#   (sigma(R0) = {SIG_R0:.4f} N/m, elastic band R/R0 0.980-1.015), kappa_s 7.2e-9 kg/s, water mu 1e-3;",
        f"#   KM linear natural frequency f0 = {F0/1e6:.3f} MHz (T0 = {1e9/F0:.1f} ns); RK3 + MUSCL2 + HLLC, cfl 0.5;",
    ] if not mili else [
        "# MILI SCALE: the micro MarmPaper deck of the same number with every length and time x 1e4 (R0 = 2 cm).",
        "#   Pressures, densities, EOS, velocities, water mu 1e-3 and thermal constants unchanged.  kappa_s = 0.72 kg/s",
        "#   (= 7.2e-9 x 1e8: same dimensionless shell viscosity K* = kappa_s/(R0^2 sqrt(rho p_inf)) as the micro deck).",
        "#   Marmottant chi 1.0 N/m, sigma_break 0.073, sigma_water 0.0728 kept physical -> inert at 2 cm (2 sigma/R <= 7 Pa).",
        f"#   KM linear natural frequency f0 = {F0_MILI:.3f} Hz (T0 = {1e3/F0_MILI:.3f} ms, = Minnaert); RK3 + MUSCL2 + HLLC, cfl 0.5;",
    ]) + [
        f"#   eps/dx_finest = 2; R0/dx_finest = {R0/dx:.1f} (max_level {max_level}); NSCBC outflow far field.",
        "#   Interface jump, tensor curvature, relaxation closure etc. are built into the code (flag audit 2026-11-05).",
        "#   THINC (qq) lines below are commented out until the first round of results is in.",
        "", "alamo.program = hydro2",
        f"plot_file = /mmfs1/home/ttryon/flames/bin/tests/FlowMarmottant/MarmPaper/output_{name}", "",
        "### OUTPUT ###",
        "amr.plot_int       = -1",
        f"amr.plot_dt        = {fmt(plot_dt)}",
        f"amr.thermo.plot_dt = {fmt(thermo_dt)}    # thermo.dat: gas_volume (R_vol), gas_pressure_int, kinetic_energy, interface_area",
        "plot_minimal       = 1", "",
        "### MESH ###",
        "amr.max_grid_size  = 32", "amr.blocking_factor = 16", "amr.regrid_int     = 10", "amr.grid_eff       = 0.8",
        f"amr.max_level      = {max_level}", "amr.n_error_buf    = 16", "nghost = 4"]
    def geom_block(octant, c):
        lo = 0.0 if octant else -half
        n = n_oct if octant else 2 * n_oct
        L = [f"{c}geometry.prob_lo     = {fmt(lo)} {fmt(lo)} {fmt(lo)}",
             f"{c}geometry.prob_hi     = {fmt(half)} {fmt(half)} {fmt(half)}",
             f"{c}geometry.is_periodic = 0 0 0",
             f"{c}amr.n_cell           = {n} {n} {n}     # base dx {half/n_oct:.4e}; finest dx {dx:.4e}"]
        if refine_box:
            a, b = (0.0 if octant else -refine_box), refine_box
            L += [f"{c}refine_box.lo = {fmt(a)} {fmt(a)} {fmt(a)}", f"{c}refine_box.hi = {fmt(b)} {fmt(b)} {fmt(b)}"]
        return L
    out += ["", f"### GEOMETRY -- {'FULL DOMAIN' if full_only else 'OCTANT (active); full domain below, commented'} ###"]
    if full_only:
        out += geom_block(False, "")
    else:
        out += geom_block(True, "") + ["# --- full domain (swap with the octant block above; also swap the BC block) ---"] + geom_block(False, "# ")
    out += ["", "### TIME ###",
            "timestep                = " + ("1.0e-12" if not mili else "1.0e-8"),
            "dynamictimestep.on      = 1", "dynamictimestep.verbose = 0",
            "dynamictimestep.max     = " + ("1.0e-9" if not mili else "1.0e-5"),
            "dynamictimestep.min     = " + ("1.0e-18" if not mili else "1.0e-14"), "cfl                     = 0.5",
            f"stop_time               = {fmt(stop)}", "",
            "### INTERFACE ###",
            "eta.ic.type = expression",
            f"eta.ic.expression.constant.epsilon = {fmt(eps)}    # eps/dx_finest = 2",
            f"eta.ic.expression.constant.R0      = {fmt(R0)}",
            "eta.ic.expression.region0 = \"0.5*(1 + tanh((sqrt(x*x + y*y + z*z) - R0)/epsilon))\"",
            "epsilon = eta.ic.expression.constant.epsilon", "",
            "### EOS: 0 = water (stiffened gas), 1 = air ###",
            "eos0.gamma = 2.35", "eos0.p0    = 1.0e9", "eos0.cv    = 1781.0", "eos0.cp    = 4186.0",
            "eos1.gamma = 1.4", "eos1.p0    = 0.0", "eos1.cv    = 714.286", "eos1.cp    = 1000.0", "",
            "### INITIAL CONDITIONS ###",
            "density0.ic.type = expression", "density0.ic.expression.region0 = \"1000.0\"",
            "density1.ic.type = expression", "density1.ic.expression.region0 = \"1.0\""]
    if jet:
        U, zc, w, h = jet
        vz = (f"-U*exp(-(x*x + y*y)/(w*w) - (z - zc)*(z - zc)/(h*h))"
              f"*0.5*(1 + tanh((sqrt(x*x + y*y + z*z) - R0)/epsilon))")
        for ph in ("0", "1"):
            out += [f"velocity{ph}.ic.type = expression",
                    f"velocity{ph}.ic.expression.constant.U  = {fmt(U)}    # jet speed [m/s] (raise for a stronger jet)",
                    f"velocity{ph}.ic.expression.constant.zc = {fmt(zc)}    # patch centre above the top cap",
                    f"velocity{ph}.ic.expression.constant.w  = {fmt(w)}    # lateral 1/e radius",
                    f"velocity{ph}.ic.expression.constant.h  = {fmt(h)}    # axial 1/e half-length",
                    f"velocity{ph}.ic.expression.constant.R0 = {fmt(R0)}",
                    f"velocity{ph}.ic.expression.constant.epsilon = {fmt(eps)}",
                    f"velocity{ph}.ic.expression.region0 = \"0.0\"",
                    f"velocity{ph}.ic.expression.region1 = \"0.0\"",
                    f"velocity{ph}.ic.expression.region2 = \"{vz}\""]
    else:
        for ph in ("0", "1"):
            out += [f"velocity{ph}.ic.type = expression"] + [f"velocity{ph}.ic.expression.region{k} = \"0.0\"" for k in range(3)]
    out += ["pressure0.ic.type = expression",
            f"pressure0.ic.expression.constant.p_inf = {fmt(p_inf)}",
            f"pressure0.ic.expression.constant.p_b   = {fmt(p_b)}",
            f"pressure0.ic.expression.constant.R0    = {fmt(R0)}",
            f"pressure0.ic.expression.region0 = \"{liquid_ic}\"",
            "pressure1.ic.type = expression",
            f"pressure1.ic.expression.constant.p_b = {fmt(p_b)}",
            "pressure1.ic.expression.region0 = \"p_b\"",
            "equalize_ic_pressure = 1", "",
            "### MATERIAL / SHELL ###",
            "mu0   = 1.0e-3     # water", "mu0_b = 0.0", "mu1   = 0.0", "mu1_b = 0.0",
            "apply_surface_tension = 1",
            "sigma = 0.0728                   # water (= post-rupture tension)", "pref  = 0.0",
            "marmottant             = 1",
            f"marmottant.R0          = {fmt(R0)}  # shell unstrained at the IC radius",
            ("marmottant.R_buckling  = 1.96e-6       # 0.98 R0" if not mili else
             f"marmottant.R_buckling  = {fmt(0.98 * R0)}  # 0.98 R0"),
            "marmottant.chi         = 1.0           # [N/m]",
            "marmottant.sigma_break = 0.073         # [N/m]",
            ("shell.kappa_s = 7.2e-9                  # [kg/s] (micro-scale equivalent of kappa_s = 0.72 at R0 = 2 cm)" if not mili else
             "shell.kappa_s = 0.72                    # [kg/s] (= micro 7.2e-9 x 1e8: same K*)"),
            "shell.mu_s    = 0.0", ""]
    faces_oct = ["xhi", "yhi", "zhi"]
    faces_full = ["xlo", "xhi", "ylo", "yhi", "zlo", "zhi"]
    out += [f"### BOUNDARY CONDITIONS -- {'full domain' if full_only else 'octant: symmetry (lo) + NSCBC outflow (hi) (active)'} ###"]
    if drive:
        out += [f"# Drive: p_target(t) = p_inf + drive_amp sin(drive_omega t) on every NSCBC face.",
                f"# drive_amp = (target amplitude at the bubble) / (box gain, calibrated: see README).",
                f"# 20 R0 box (not 100 R0): at f0 the 100 R0 octant is near an acoustic resonance (gain 9.2, non-uniform)."]
    if full_only:
        out += nscbc(faces_full, p_inf, L_ref, amp, om) + bcs(False)
    else:
        out += nscbc(faces_oct, p_inf, L_ref, amp, om) + bcs(True)
        out += ["# --- full domain: NSCBC outflow on all six faces (swap with the octant BC block above) ---"]
        out += nscbc(["xlo", "ylo", "zlo"], p_inf, L_ref, amp, om, comment="# ") + bcs(False, comment="# ")
    out += ["", "### REFINEMENT (eta band) ###",
            "eta_refinement_criterion   = 1.0e-1", "omega_refinement_criterion = 1.0e+10",
            "p_refinement_criterion     = 1.0e+10", "rho_refinement_criterion   = 1.0e+10",
            "gradu_refinement_criterion = 1.0e+10", "",
            "### NUMERICS ###",
            "cutoff = 1.0e-4",
            "Riemann_Solver.type = hllc",
            "Limiter.type        = muscl2",
            "# THINC qq (second round; uncomment these three lines and comment out Limiter.type above):",
            "# Limiter.type       = thinc",
            "# Limiter.thinc.thinc_base = muscl2",
            "# Limiter.thinc.thinc_mode = qq       # key is Limiter.thinc.<key> (Limiter.thinc_mode is silently ignored)",
            "integration.type    = RungeKutta",
            "integration.rk.type = 3",
            "visc_subcycle              = 1       # bulk + shell viscous terms sub-cycled",
            "shell.kappa_s_subcycle_max = 8192",
            "shell.heat_split           = 2       # shell viscous heat split by thermal effusivity",
            "thermal.k0                 = 0.6     # water [W/m/K]",
            "thermal.k1                 = 0.026   # air   [W/m/K]",
            "relax_diag                 = 0",
            "apply_weight          = 0", "apply_vaporization    = 0", "grav                  = 0.0",
            "m0.ic.constant.value = 0.0", "u0.ic.constant.value = 0.0 0.0", "q.ic.constant.value  = 0.0 0.0"]
    open(f"{DECKS}/input_{name}", "w").write("\n".join(out) + "\n")
    return dict(name=name, dx=dx, eps=eps, stop=stop)

LIQ_PROFILE = "p_inf - (p_inf - p_b)*R0/max(sqrt(x*x + y*y + z*z), R0)"   # Light_Mili IC (liquid relaxes to p_b at R0)
LIQ_UNIFORM = "p_inf"                                                      # static equilibrium (gas at p_eq)
T0 = 1.0 / F0

runs = []
runs.append(deck("01_FreeOscillation", "free natural spherical oscillation (micro-scale Light_Mili)", [
    "Light_Mili deck shrunk to R0 = 2 um with the physical Marmottant shell: same p_inf = 1e5, p_b = 5.8307e4",
    "(gas below equilibrium, p_eq = 1.4123e5), same 100 R0 domain and R0/dx = 25.6.",
    "KM: R_min 0.829 R0 (shell buckles), decays to the new equilibrium R ~ 0.88 R0 in ~5 T0.",
    "stop_time = 12 T0."],
    p_inf=P0, p_b=5.8307e4, liquid_ic=LIQ_PROFILE, max_level=4, stop=12 * T0, plot_dt=T0 / 20, thermo_dt=T0 / 200))
runs.append(deck("02_FreeCollapse", "free natural spherical collapse (higher pressure ratio)", [
    "As 01 with p_inf raised to 1.0e6 (p_inf/p_b = 17.2), one level finer (R0/dx 51.2) for R_min.",
    "KM: R_min 0.321 R0 at t = 70.6 ns (tau_c = 57.9 ns), first rebound R_max 0.592 R0.",
    "stop_time = 400 ns (~7 tau_c: collapse + 2-3 rebounds)."],
    p_inf=1.0e6, p_b=5.8307e4, liquid_ic=LIQ_PROFILE, max_level=5, stop=4.0e-7, plot_dt=4.0e-9, thermo_dt=2.0e-10))
WL = W0 / math.pi
runs.append(deck("03_ForcedLinear", "forced LINEAR spherical oscillation (f_drive = f0/pi)", [
    f"Bubble starts at static equilibrium (liquid p_inf = 1e5, gas p_eq = {P_EQ:.4e}).",
    f"Drive f = f0/pi = {WL/2/math.pi/1e6:.4f} MHz (non-integer ratio, no harmonic resonance), amplitude 1.0e4 Pa at the bubble.",
    "KM: R/R0 = 1 +- 0.0042 (inside the elastic band 0.980-1.015), steady after 2 drive cycles.",
    f"stop_time = 8 drive periods = {8*2*math.pi/WL*1e6:.3f} us.  Wavelength {1533/(WL/2/math.pi)*1e3:.2f} mm >> box: uniform drive."],
    p_inf=P0, p_b=P_EQ, liquid_ic=LIQ_UNIFORM, max_level=4, stop=8 * 2 * math.pi / WL,
    plot_dt=2 * math.pi / WL / 16, thermo_dt=2 * math.pi / WL / 400, drive=(1.0e4 / GAIN_LIN, WL), half=20 * R0, n_oct=32))
runs.append(deck("04_ForcedNonlinear", "forced NONLINEAR oscillation near resonance (subharmonic)", [
    f"Bubble starts at static equilibrium.  Drive f = f0 = {F0/1e6:.3f} MHz, amplitude 4.0e5 Pa at the bubble (4 bar,",
    "liquid far field reaches -3 bar).  KM (km_sub.py): period doubling from the first cycles, locked in by",
    "cycle ~8: alternate cycles R_min 0.616 / 1.112 R0, R_max 1.491 R0, f/2 subharmonic 2.0x the drive line.",
    "Robust: period doubling persists for 3.6-4.5 bar and f/f0 0.9-1.1; R_min > 0.5 R0 throughout (validation target).",
    f"stop_time = 20 drive periods = {20*T0*1e6:.3f} us."],
    p_inf=P0, p_b=P_EQ, liquid_ic=LIQ_UNIFORM, max_level=4, stop=20 * T0, plot_dt=T0 / 16, thermo_dt=T0 / 400,
    drive=(4.0e5 / GAIN_NL, W0), half=20 * R0, n_oct=32))
runs.append(deck("05_Jetting", "jetting: liquid jet through the bubble, U shape -> torus (FULL DOMAIN)", [
    "Full 3D domain (no symmetry), +-8 R0 box with NSCBC outflow on all six faces, R0/dx 64.",
    "Bubble at static equilibrium; a downward (-z) Gaussian liquid velocity patch sits on the top cap",
    "(U = 50 m/s, centre 1.4 R0, lateral 1/e radius 0.4 R0, axial 0.5 R0, liquid side only).  The impulsive start",
    "sheds a short acoustic pulse (~rho c U/2, sub-ns, decays as 1/r); the solenoidal part of the patch is the jet.",
    "The jet dynamic pressure rho U^2/2 = 1.25e6 Pa >> gas 1.4e5 and Laplace 4e4 Pa: it indents the top cap (U shape),",
    "crosses the bubble (~2 R0 / U ~ 80-150 ns) and pierces the bottom -> torus.  Raise U for a faster jet.",
    "refine_box +-2 R0 keeps the finest level on the jet path.  stop_time = 400 ns."],
    p_inf=P0, p_b=P_EQ, liquid_ic=LIQ_UNIFORM, max_level=4, stop=4.0e-7, plot_dt=4.0e-9, thermo_dt=2.0e-10,
    jet=(50.0, 1.4 * R0, 0.4 * R0, 0.5 * R0), full_only=True, half=8 * R0, n_oct=32, refine_box=2 * R0))

# ---- _Mili twins of 02, 04, 05 (R0 = 2 cm; see S_MILI above).  KM numbers: ~/claude_scratch/marmpaper_mili_km.py,
# marmpaper_mili_sub.py (same Keller-Miksis + Marmottant model as km_size.py / km_sub.py, Mili parameters). ----
RM, T0M = R0_MILI, 1.0 / F0_MILI
runs.append(deck("02_FreeCollapse_Mili", "free natural spherical collapse, MILI scale (R0 = 2 cm)", [
    "02_FreeCollapse with every length and time x 1e4: p_inf 1.0e6, p_b 5.8307e4, R0/dx 51.2, 100 R0 octant.",
    "KM (Mili parameters: elastic shell inert, kappa_s 0.72): R_min 0.308 R0 at t = 0.696 ms (tau_c 0.579 ms),",
    "first rebound R_max 0.610 R0 (micro 02: 0.321 / 0.592 -- the shell viscosity dominates both).",
    "stop_time = 4 ms (~7 tau_c: collapse + 2-3 rebounds)."],
    p_inf=1.0e6, p_b=5.8307e4, liquid_ic=LIQ_PROFILE, max_level=5, stop=4.0e-3, plot_dt=4.0e-5, thermo_dt=2.0e-6,
    mili=True))
WNM = 2.0 * W0_MILI
runs.append(deck("04_ForcedNonlinear_Mili", "forced NONLINEAR oscillation, MILI scale (subharmonic at 2 f0)", [
    f"Bubble starts at static equilibrium (p_eq = {P_EQ_MILI:.6e}).  Without the elastic shell f0 = Minnaert = {F0_MILI:.1f} Hz and",
    "the micro drive (4 bar at f0) is chaotic here (KM R_max 3 R0), so the drive moves to the classical subharmonic",
    f"resonance: f = 2 f0 = {WNM/2/math.pi:.1f} Hz, 3.0e5 Pa at the bubble.  KM: period doubling, alternate-cycle R_min",
    "0.555 / 1.197 R0, R_max 1.60 R0, f/2 line 1.8x the drive line.  Robust for f/f0 1.8-2.2 and 2.5-3.5 bar;",
    "R_min > 0.5 R0 (validation target, as micro 04).",
    f"stop_time = 40 drive periods = 20 T0 = {20*T0M*1e3:.2f} ms."],
    p_inf=P0, p_b=P_EQ_MILI, liquid_ic=LIQ_UNIFORM, max_level=4, stop=20 * T0M, plot_dt=T0M / 16, thermo_dt=T0M / 800,
    drive=(3.0e5 / GAIN_NL_MILI, WNM), half=20 * RM, n_oct=32, mili=True))
runs.append(deck("05_Jetting_Mili", "jetting, MILI scale (R0 = 2 cm): liquid jet through the bubble, U shape -> torus (FULL DOMAIN)", [
    "05_Jetting with every length and time x 1e4: full 3D domain, +-8 R0 box, NSCBC outflow on all six faces, R0/dx 64.",
    "Bubble at static equilibrium; downward (-z) Gaussian liquid velocity patch on the top cap (U = 50 m/s,",
    "centre 1.4 R0, lateral 1/e radius 0.4 R0, axial 0.5 R0, liquid side only).  rho U^2/2 = 1.25e6 Pa >> gas 1e5 Pa:",
    "U-shaped indentation, jet crosses the bubble in ~2 R0 / U ~ 0.8-1.5 ms, pierces the bottom -> torus.",
    "refine_box +-2 R0.  stop_time = 4 ms."],
    p_inf=P0, p_b=P_EQ_MILI, liquid_ic=LIQ_UNIFORM, max_level=4, stop=4.0e-3, plot_dt=4.0e-5, thermo_dt=2.0e-6,
    jet=(50.0, 1.4 * RM, 0.4 * RM, 0.5 * RM), full_only=True, half=8 * RM, n_oct=32, refine_box=2 * RM, mili=True))

for r in runs:
    nodes = 4 if "Jetting" in r["name"] else 1
    s = f"""#!/bin/bash

#SBATCH --job-name=MP{r['name'][:2]}{r['name'][3:9]}{'M' if r['name'].endswith('_Mili') else ''}
#SBATCH -o /mmfs1/home/ttryon/FLAMES_out/flame_%j_stdout
#SBATCH -e /mmfs1/home/ttryon/FLAMES_out/err_flame_%j_stdout
#SBATCH -N {nodes}
#SBATCH --ntasks-per-node=128
#SBATCH --partition=compute-long
#SBATCH -t 500:00:00

module purge
module load gnu9 mpich

## MarmPaper {r['name']} (see the deck header; finest dx {r['dx']:.3e} m, eps {r['eps']:.3e} m) ##
# Needs hydro2 built from the 2026-11-05 flag-audit code (interface jump / tensor curvature built in).
cd /home/ttryon/flames

srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-3d-g++ /home/ttryon/flames/tests/FlowMarmottant/MarmPaper/input_{r['name']}
"""
    open(f"{SLURM}/{r['name']}.sh", "w").write(s)
    os.chmod(f"{SLURM}/{r['name']}.sh", 0o755)
print("\n".join(f"{r['name']}: dx {r['dx']:.3e} eps {r['eps']:.3e} stop {r['stop']:.3e}" for r in runs))
