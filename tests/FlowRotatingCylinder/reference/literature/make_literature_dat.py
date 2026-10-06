#!/usr/bin/env python3
"""
Writes the reference data files of this folder (rotating circular cylinder, 2D laminar, incompressible).

  python3 make_literature_dat.py

Sources
  Kang, Choi & Lee, "Laminar flow past a rotating circular cylinder", Phys. Fluids 11, 3312 (1999)
  Stojkovic, Breuer & Durst, "Effect of high rotation rates on the laminar flow around a circular cylinder",
      Phys. Fluids 14, 3160 (2002)
alpha = cylinder surface speed / free-stream speed in both (same as Omega R / U of the Hydro2 decks);
coefficients are normalised by rho U^2 D / 2.  Lift is NEGATIVE here (counter-clockwise spin, stream in +x), the
convention of Kang et al. and of the Hydro2 runs; Stojkovic et al. print its magnitude.

"table" / "equation" values are copied from the papers.  "digitized" values were read off the scanned figures on
2026-10-05 with digitize_tools.py (axes calibrated on the tick marks, marker centres from filled-hole blobs, x
crosses by template correlation).  Checks of the digitization against tabulated numbers:
  Kang Fig. 9, Re 100, alpha 1:  C_L' 0.3622 (Table I 0.3631), C_D' 0.1001 (0.0993)
  Kang Fig. 5, Re 100, alpha 1:  St 0.1656 (Table I 0.1655)
  Kang Fig. 8(b), Re 100, alpha 1: C_D 1.099 (Table I 1.1040)
  Stojkovic Fig. 13(a), alpha 2: 5.51 (Table II 5.48);  Kang's points re-plotted in Stojkovic Fig. 13 agree with
  the values digitized from Kang's own figures to 0.004 (amplitude) and 0.01 (mean lift at alpha = 1).
"""
import os
nan = float("nan")
HERE = os.path.dirname(os.path.abspath(__file__))


def write(name, header, cols, rows):
    with open(os.path.join(HERE, name), "w") as f:
        for l in header.strip("\n").split("\n"): f.write("# " + l + "\n")
        f.write("# columns: " + "  ".join(cols) + "\n")
        for r in rows: f.write("  ".join(f"{v:>10.5g}" if isinstance(v, float) else f"{v:>10}" for v in r) + "\n")
    print("wrote", name, len(rows), "rows")


# ----------------------------------------------------------------------------------------- Stojkovic et al. 2002
write("stojkovic2002_table2_Re100.dat", """
Stojkovic, Breuer & Durst, Phys. Fluids 14, 3160 (2002), TABLE II (copied; exact).  Re = 100.
Strouhal number, time-averaged lift and drag and their amplitudes (half peak-to-peak).  nan = not given
(alpha = 2: steady flow, no vortex shedding; drag and lift given to 2-3 digits only).
Lift negated to the Kang / Hydro2 sign convention.""",
      ["alpha", "Cl_mean", "Cl_amp", "Cd_mean", "Cd_amp", "St"],
      [(0.0, -0.0, 0.3259, 1.3371, 0.0091, 0.1650), (0.5, -1.220, 0.3420, 1.2770, 0.0513, 0.1657),
       (1.0, -2.504, 0.3616, 1.1080, 0.0986, 0.1658), (1.5, -3.900, 0.3180, 0.8180, 0.1140, 0.1626),
       (2.0, -5.48, nan, 0.46, nan, nan)])

T = "table"; Dg = "digitized"; I = "interpolated"
F13 = [  # alpha, C_Lav, source, C_Lam, source
    (0.0, 0.0, T, 0.3259, T), (0.5, 1.220, T, 0.3420, T), (1.0, 2.504, T, 0.3616, T), (1.5, 3.900, T, 0.3180, T),
    (2.0, 5.48, T, 0.0, Dg), (2.5, 7.61, Dg, 0.0, Dg), (3.0, 10.14, Dg, 0.0, Dg), (3.5, 13.36, Dg, 0.0, Dg),
    (4.0, 17.00, Dg, 0.0, Dg), (4.5, 21.35, Dg, 0.0, Dg), (4.75, 23.87, Dg, 0.0, Dg), (4.8, 24.43, Dg, 0.752, Dg),
    (4.9, 25.55, Dg, 1.142, Dg), (5.0, 26.53, Dg, 1.230, Dg), (5.1, 27.51, Dg, 1.194, Dg), (5.15, 28.6, Dg, 0.956, Dg),
    (5.2, 29.0, I, 0.0, Dg), (5.25, 29.3, Dg, 0.0, Dg), (5.5, 30.88, Dg, 0.0, Dg), (6.0, 34.38, Dg, 0.0, Dg),
    (6.5, 37.74, Dg, 0.0, Dg), (7.0, 41.39, Dg, 0.0, Dg), (8.0, 48.11, Dg, 0.0, Dg), (9.0, 54.84, Dg, 0.0, Dg),
    (10.0, 61.57, Dg, 0.0, Dg), (11.0, 68.01, Dg, 0.0, Dg), (12.0, 74.6, Dg, 0.0, Dg)]
write("stojkovic2002_fig13_Re100.dat", """
Stojkovic, Breuer & Durst, Phys. Fluids 14, 3160 (2002), FIG. 13: mean lift and lift amplitude (half
peak-to-peak) for Re = 100, 0 <= alpha <= 12.  alpha <= 2 from TABLE II; the rest digitized.
Digitization uncertainty: Cl_mean +-0.15 (1 pixel = 0.14; +-0.3 for the three overlapping points at
alpha = 5.15, 5.2, 5.25, whose spin ratios are inferred -- the paper gives the unsteady window as
4.8 <= alpha <= 5.15); Cl_amp +-0.003.  The flow is steady (Cl_amp = 0) for 2 <= alpha < 4.8 and alpha > 5.15.
Potential flow: Cl = -2 pi alpha.  Lift negated to the Kang / Hydro2 sign convention.""",
      ["alpha", "Cl_mean", "Cl_mean_source", "Cl_amp", "Cl_amp_source"],
      [(a, -cl if cl else -0.0, s1, am, s2) for a, cl, s1, am, s2 in F13])

write("stojkovic2002_fig13_kang_points_Re100.dat", """
Points labelled "Kang & Choi (1999)" in Stojkovic et al. (2002), FIG. 13, Re = 100 (digitized there; Cl_mean +-0.15,
Cl_amp +-0.003).  For comparison with kang1999_*.dat, which were digitized from Kang's own figures.""",
      ["alpha", "Cl_mean", "Cl_amp"], [(0.5, -1.17, 0.339), (1.0, -2.48, 0.360), (1.5, -3.69, 0.324)])

# ----------------------------------------------------------------------------------------------- Kang et al. 1999
write("kang1999_table1_Re100_alpha1.dat", """
Kang, Choi & Lee, Phys. Fluids 11, 3312 (1999), TABLE I (copied; exact): parametric study at Re = 100, alpha = 1.
R_D = domain radius / d, M x N = grid, dt = time step.  First row is the parameter set used in the paper.""",
      ["R_D", "M", "N", "dt", "St", "Cl_mean", "Cd_mean", "Cl_amp", "Cd_amp"],
      [(50, 241, 241, 0.02, 0.1655, -2.4881, 1.1040, 0.3631, 0.0993), (100, 285, 241, 0.02, 0.1650, -2.4833, 1.0979, 0.3603, 0.0988),
       (50, 301, 301, 0.02, 0.1656, -2.5027, 1.0993, 0.3576, 0.0980), (50, 241, 241, 0.01, 0.1656, -2.4881, 1.1039, 0.3630, 0.0991)])

write("kang1999_eq11_mean_lift_fit.dat", """
Kang, Choi & Lee, Phys. Fluids 11, 3312 (1999), Eq. (11) (copied; exact): least-squares fits of the mean lift for
alpha <= 1,  Cl_mean = slope * alpha, with its pressure and friction parts.
USE THIS for the mean lift: in Kang's FIG. 8(a) the curves for Re = 40 .. 160 lie within 2-5 pixels of each other
(1 pixel = 0.021) and their markers overlap, so the Reynolds numbers cannot be separated by digitizing.
Beyond the fit range the figure shows all four Re near Cl = -5.4 .. -5.65 at alpha = 2 and -7.55 at alpha = 2.5.""",
      ["Re", "slope_total", "slope_pressure", "slope_friction", "alpha_max"],
      [(40, -2.57, -2.31, -0.26, 1.0), (60, -2.50, -2.29, -0.21, 1.0), (100, -2.48, -2.31, -0.17, 1.0), (160, -2.46, -2.33, -0.13, 1.0)])

A = [0.0, 0.1, 0.2, 0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5]
def table(series):          # series: {Re: {alpha: value}} -> rows over the union grid
    return [tuple([a] + [float(series[r].get(a, nan)) for r in sorted(series)]) for a in A if any(a in series[r] for r in series)]

CD = {40: {0: 1.503, .1: 1.506, .2: 1.498, .5: 1.458, 1: 1.318, 1.5: 1.107, 2: .849, 2.5: .579},
      60: {0: 1.391, .1: 1.384, .2: 1.380, .5: 1.332, 1: 1.161, 1.1: 1.114, 1.2: 1.058, 1.3: .999, 1.4: .936, 1.5: .891, 2: .650, 2.5: .412},
      100: {0: 1.322, .1: 1.321, .2: 1.314, .5: 1.268, 1: 1.099, 1.5: .815, 1.6: .740, 1.7: .664, 1.8: .580, 1.9: .511, 2: .471, 2.5: .279},
      160: {0: 1.322, .1: 1.321, .2: 1.314, .5: 1.268, 1: 1.099, 1.5: .815, 1.6: .740, 1.7: .652, 1.8: .556, 1.9: .439, 2: .352, 2.5: .205}}
write("kang1999_fig8b_mean_drag.dat", """
Kang, Choi & Lee, Phys. Fluids 11, 3312 (1999), FIG. 8(b), total mean drag (solid lines), digitized.
Uncertainty +-0.006 for isolated markers (1 pixel = 0.004).
Re = 100 and Re = 160 OVERLAP for alpha <= 1.6 (the two markers print on top of each other): both columns
hold the centre of the merged marker there, uncertainty +-0.01 each; they separate from alpha = 1.7 on
(+-0.015 at 1.7 and 1.8, where the split is estimated from the height of the merged blob).
Re = 40 at alpha = 1.0 is merged with a filled square of unknown meaning (+-0.015).
Table I gives Re = 100, alpha = 1: 1.1040.  nan = no data point at that alpha.
The pressure and friction parts of FIG. 8 (dash-dot, dotted) were NOT digitized.""",
      ["alpha", "Cd_Re40", "Cd_Re60", "Cd_Re100", "Cd_Re160"], table(CD))

CLA = {60: {0: .1284, .1: .1281, .2: .1295, .5: .1327, 1: .1229, 1.1: .1116, 1.2: .0953, 1.3: .0665, 1.4: .0085, 1.5: 0, 2: 0, 2.5: 0},
       100: {0: .3249, .1: .3265, .2: .3281, .5: .3410, 1: .3622, 1.5: .3218, 1.6: .2868, 1.7: .2287, 1.8: .1204, 1.9: 0, 2: 0, 2.5: 0},
       160: {0: .5550, .1: .5556, .2: .5584, .5: .5721, 1: .5959, 1.5: .5686, 1.6: .5416, 1.7: .4938, 1.8: .4057, 1.9: .2214, 2: 0, 2.5: 0}}
CDA = {60: {0: .0008, .1: .0044, .2: .0079, .5: .0202, 1: .0346, 1.1: .0339, 1.2: .0304, 1.3: .0227, 1.4: .0036, 1.5: 0, 2: 0, 2.5: 0},
       100: {0: .0092, .1: .0163, .2: .0244, .5: .0521, 1: .1001, 1.5: .1163, 1.6: .1075, 1.7: .0877, 1.8: .0467, 1.9: 0, 2: 0, 2.5: 0},
       160: {0: .0294, .1: .0400, .2: .0513, .5: .0883, 1: .1588, 1.5: .2031, 1.6: .2007, 1.7: .1883, 1.8: .1576, 1.9: .0859, 2: 0, 2.5: 0}}
write("kang1999_fig9a_lift_amplitude.dat", """
Kang, Choi & Lee, Phys. Fluids 11, 3312 (1999), FIG. 9(a): amplitude of the lift fluctuation C_L', digitized.
Uncertainty +-0.003 (1 pixel = 0.0016).  0 = no vortex shedding (markers on the zero line; at alpha = 2 and 2.5
the three markers coincide).  Table I gives Re = 100, alpha = 1: 0.3631.  nan = no data point.""",
      ["alpha", "Cl_amp_Re60", "Cl_amp_Re100", "Cl_amp_Re160"], table(CLA))
write("kang1999_fig9b_drag_amplitude.dat", """
Kang, Choi & Lee, Phys. Fluids 11, 3312 (1999), FIG. 9(b): amplitude of the drag fluctuation C_D', digitized.
Uncertainty +-0.001 (1 pixel = 0.0006).  0 = no vortex shedding.  Table I gives Re = 100, alpha = 1: 0.0993.""",
      ["alpha", "Cd_amp_Re60", "Cd_amp_Re100", "Cd_amp_Re160"], table(CDA))

ST = {60: {0: .1358, .1: .1358, .2: .1359, .5: .1365, 1: .1378, 1.1: .1379, 1.2: .1377, 1.3: .1375, 1.4: .1363},
      100: {0: .1647, .1: .1647, .2: .1647, .5: .1650, 1: .1656, 1.5: .1635, 1.6: .1623, 1.7: .1607, 1.8: .1587},
      160: {0: .1864, .1: .1865, .2: .1862, .5: .1860, 1: .1856, 1.5: .1819, 1.6: .1803, 1.7: .1784, 1.8: .1763, 1.9: .1750}}
write("kang1999_fig5_strouhal.dat", """
Kang, Choi & Lee, Phys. Fluids 11, 3312 (1999), FIG. 5: Strouhal number vs alpha, digitized.
Uncertainty +-0.0005 (1 pixel = 0.00016).  nan = no vortex shedding / no data point (shedding stops above
alpha_L ~ 1.4 at Re = 60, ~1.8 at Re = 100, ~1.9 at Re = 160).  Table I gives Re = 100, alpha = 1: 0.1655.""",
      ["alpha", "St_Re60", "St_Re100", "St_Re160"], table(ST))
write("kang1999_fig5_hu1996_Re60.dat", """
Hu et al. (Kang's Ref. 17), Re = 60, as plotted in Kang, Choi & Lee (1999), FIG. 5 (diamonds), digitized, +-0.0005.
Kang et al. disagree with this curve; kept for completeness.""",
      ["alpha", "St"], [(0.0, .1528), (0.1, .1524), (0.2, .1517), (0.3, .1505), (0.4, .1488), (0.5, .1464), (0.6, .1432), (0.8, .1357), (1.0, .1261)])


# ============================================================== second batch (2026-10-05): steady low-Re regime
# Stojkovic Figs. 9 and 10 use large markers; each series was located with a template cut from the figure's own
# legend (digitize_tools.legend_template / refine), starting from a first reading.  "q" columns: template score,
# >= 0.78 = clean, lower = the marker is partly hidden by others (uncertainty roughly doubled).
RE9 = [1, 2, 3, 4, 5, 7.5, 10, 15, 20, 25, 30, 35, 40, 45]
P9 = {"a5_f": [5.275, 3.647, 3.088, 2.795, 2.618, 2.348, 2.188, 1.976, 1.837, 1.725, 1.635, 1.566, 1.507, 1.454],
      "a5_p": [2.643, -0.042, -0.994, -1.532, -1.871, -2.235, -2.343, -2.352, -2.265, -2.151, -2.054, -1.956, -1.878, -1.793],
      "a3_p": [4.319, 2.215, 1.363, 0.890, 0.586, 0.155, -0.079, -0.292, -0.393, -0.452, -0.478, -0.494, -0.505, -0.510],
      "a3_f": [5.164, 3.343, 2.635, 2.279, 2.029, 1.672, 1.465, 1.220, 1.076, 0.975, 0.906, 0.847, 0.805, 0.762],
      "a0.5_f": [5.177, 3.256, 2.524, 2.114, 1.842, 1.438, 1.209, 0.954, 0.799, 0.698, 0.624, 0.570, 0.522, 0.485],
      "a0.5_p": [nan, nan, nan, nan, nan, nan, nan, 1.287, 1.167, 1.081, 1.017, 0.975, 0.936, 0.906]}
write("stojkovic2002_fig9b_drag_parts_lowRe.dat", """
Stojkovic, Breuer & Durst, Phys. Fluids 14, 3160 (2002), FIG. 9(b): pressure (p) and friction (f) parts of the
drag coefficient, steady flow, 1 <= Re <= 45, alpha = 0.5, 3, 5.  Digitized (legend-template matching).
Uncertainty +-0.02 (1 pixel = 0.011); +-0.08 for the friction parts of alpha = 0.5 and 3 at Re <= 5, where the
solid markers overlap, and +-0.05 for the alpha = 0.5 pressure part (hollow circle under solid squares);
nan = that marker is hidden (alpha = 0.5 pressure part, Re <= 10).
Check: Cdp + Cdf at alpha = 5, Re = 5 is 0.747; the total-drag line of FIG. 9(a) reads 0.74 there.""",
      ["Re", "Cdp_a0.5", "Cdf_a0.5", "Cdp_a3", "Cdf_a3", "Cdp_a5", "Cdf_a5"],
      [(float(r), P9["a0.5_p"][i], P9["a0.5_f"][i], P9["a3_p"][i], P9["a3_f"][i], P9["a5_p"][i], P9["a5_f"][i]) for i, r in enumerate(RE9)])

A0 = {15: 2.295, 20: 1.99, 25: 1.82, 30: 1.695, 35: 1.60, 40: 1.525, 45: 1.46}
A05 = {15: 2.26, 20: 1.95, 25: 1.78, 30: 1.65, 35: 1.54, 40: 1.46, 45: 1.39}
A1 = {10: 2.60, 15: 2.10, 20: 1.82, 25: 1.65, 30: 1.51, 35: 1.39, 40: 1.31, 45: 1.25}
A2 = {7.5: 2.58, 10: 2.14, 15: 1.62, 20: 1.34, 25: 1.15, 30: 1.02, 35: 0.92, 40: 0.84, 45: 0.78}
write("stojkovic2002_fig9a_drag_lowRe.dat", """
Stojkovic, Breuer & Durst, Phys. Fluids 14, 3160 (2002), FIG. 9(a): total drag coefficient vs Re, steady flow,
for alpha = 0, 0.5, 1, 2, 3, 5.  The figure draws these as LINES without markers.
alpha = 0, 0.5, 1, 2: read from the lines at the listed Re, +-0.03 (+-0.05 for alpha = 0 and 0.5, which nearly
coincide); nan where the lines and the literature symbols cannot be told apart (low Re).
alpha = 3 and 5: sum of the pressure and friction parts digitized from FIG. 9(b) (+-0.03; +-0.08 at Re <= 5
for alpha = 3); they agree with the lines of FIG. 9(a) where those can be read (alpha = 5: -0.36 / -0.43 /
-0.41 / -0.34 at Re = 15 / 20 / 30 / 45 from the line).  Note the NEGATIVE drag at alpha = 5 for Re > 8.""",
      ["Re", "Cd_a0", "Cd_a0.5", "Cd_a1", "Cd_a2", "Cd_a3", "Cd_a5"],
      [(float(r), A0.get(r, nan), A05.get(r, nan), A1.get(r, nan), A2.get(r, nan), round(P9["a3_p"][i] + P9["a3_f"][i], 3), round(P9["a5_p"][i] + P9["a5_f"][i], 3))
       for i, r in enumerate(RE9)])

REL = ["0.01", "0.1", "1", "5", "10", "20", "30"]
F10A = {0: [0, 0, 0, 0, 0, 0, 0], 1: [2.85] * 7, 2: [5.95] * 7, 3: [9.1, 9.07, 9.15, nan, 9.60, 9.85, 9.85],
        4: [12.2, 12.2, 12.45, 13.61, 14.48, 15.29, 15.91], 5: [15.15, 15.3, 16.09, 18.99, 20.71, 22.54, 23.59],
        6: [18.05, 18.3, 19.93, 25.36, 28.51, 31.69, 33.75]}
write("stojkovic2002_fig10a_lift_lowRe.dat", """
Stojkovic, Breuer & Durst, Phys. Fluids 14, 3160 (2002), FIG. 10(a): lift coefficient vs alpha, steady flow,
Re = 0.01, 0.1, 1, 5, 10, 20, 30, 0 <= alpha <= 6.  Lift negated to the Kang / Hydro2 sign convention.
alpha = 4, 5, 6: read from enlarged crops of the figure, +-0.1 (+-0.2 for Re = 0.01 and 0.1, whose markers
overlap); they agree with the sums of the parts of FIG. 10(b) to 0.1.
alpha = 3: sums of the parts of FIG. 10(b), +-0.3; nan for Re = 5 (no parts given).
alpha = 1 and 2: the seven markers lie on top of each other; EVERY column holds the centre of the cluster
(marker centres span 2.65 .. 3.0 at alpha = 1 and 5.75 .. 6.35 at alpha = 2).
Potential flow: -2 pi alpha.""",
      ["alpha"] + ["Cl_Re" + r for r in REL], [tuple([float(a)] + [-float(v) if v == v else nan for v in F10A[a]]) for a in sorted(F10A)])

F10B = {  # series: [(value, score) for alpha = 1..6]
    "Re0.01_p": [(2.33, .62), (3.82, .76), (5.66, .93), (7.52, .86), (9.35, .74), (11.22, .64)], "Re0.01_f": [(1.26, .58), (2.42, .64), (3.60, .62), (4.73, .64), (5.88, .64), (7.15, .60)],
    "Re0.1_p": [(2.07, .59), (3.76, .71), (5.63, .83), (7.54, .78), (9.47, .74), (11.34, .67)], "Re0.1_f": [(1.12, .68), (2.31, .81), (3.44, .80), (4.66, .87), (5.81, .87), (6.96, .83)],
    "Re1_p": [(2.28, .65), (3.97, .71), (6.42, .89), (8.76, .87), (11.30, .89), (14.07, .60)], "Re1_f": [(1.23, .60), (2.00, .61), (2.73, .72), (3.72, .84), (4.79, .85), (5.88, .78)],
    "Re10_p": [(2.33, .65), (5.00, .67), (8.10, .79), (12.31, .84), (17.80, .86), (24.83, .70)], "Re10_f": [(0.30, .62), (0.85, .64), (1.50, .73), (2.13, .74), (2.87, .78), (3.70, .77)],
    "Re20_p": [(2.39, .64), (5.06, .65), (8.66, .74), (13.62, .89), (20.28, .93), (28.80, .73)], "Re20_f": [(0.35, .63), (0.80, .61), (1.2, 0.0), (1.67, 0.0), (2.26, 0.0), (2.87, .68)],
    "Re30_p": [(2.34, .72), (5.16, .71), (8.90, .80), (14.32, .81), (21.58, .88), (31.22, .59)], "Re30_f": [(0.14, .59), (0.73, .60), (0.95, .61), (1.44, .65), (1.91, .69), (2.50, .65)]}
rows = []
for k, a in enumerate((1, 2, 3, 4, 5, 6)):
    r = [float(a)]
    for re in ("0.01", "0.1", "1", "10", "20", "30"):
        for part in ("p", "f"):
            v, q = F10B[f"Re{re}_{part}"][k]; r += [-v, q]
    rows.append(tuple(r))
write("stojkovic2002_fig10b_lift_parts_lowRe.dat", """
Stojkovic, Breuer & Durst, Phys. Fluids 14, 3160 (2002), FIG. 10(b): pressure (p) and friction (f) parts of the
lift coefficient, steady flow, Re = 0.01, 0.1, 1, 10, 20, 30, alpha = 1 .. 6 (alpha = 0: both zero).
Digitized (legend-template matching); lift negated to the Kang / Hydro2 sign convention.
Each value is followed by its template score q: q >= 0.78 clean, +-0.1 (1 pixel = 0.044);
q < 0.78: the marker is partly hidden by neighbours, +-0.3 (all of alpha = 1 and 2, and the friction parts of
Re = 10, 20, 30, which lie within 1 of each other);
q = 0: not readable, value DERIVED as (total of FIG. 10(a)) - (pressure part) or interpolated (Re = 20 friction).
Check: p + f at alpha = 6 gives 33.72 / 31.67 / 28.53 / 19.95 / 18.30 for Re = 30 / 20 / 10 / 1 / 0.1 against
33.75 / 31.69 / 28.51 / 19.93 / 18.3 read independently from FIG. 10(a).""",
      ["alpha"] + [f"Cl{p}_Re{re}  q" for re in ("0.01", "0.1", "1", "10", "20", "30") for p in ("p", "f")], rows)

# --------------------------------------------------------------------------- Kang Fig. 8(b): friction / pressure parts
CDF = {40: {0: .515, .1: .521, .2: .520, .5: .524, 1: .543, 1.5: .575, 2: .626, 2.5: .706},
       60: {0: .426, .1: .426, .2: .426, .5: .430, 1: .438, 1.1: .439, 1.2: .442, 1.3: .446, 1.4: .451, 1.5: .455, 2: .511, 2.5: .587},
       100: {0: .340, .1: .340, .2: .340, .5: .340, 1: .348, 1.5: .364, 1.6: .370, 1.7: .372, 1.8: .374, 1.9: .384, 2: .396, 2.5: .471},
       160: {0: .275, .1: .275, .2: .276, .5: .278, 1: .284, 1.5: .298, 1.6: .300, 1.7: .304, 1.8: .306, 1.9: .307, 2: .316, 2.5: .386}}
write("kang1999_fig8b_drag_friction.dat", """
Kang, Choi & Lee, Phys. Fluids 11, 3312 (1999), FIG. 8(b), FRICTION part of the mean drag (dotted lines),
digitized.  Uncertainty +-0.006 for isolated markers, +-0.012 where the marker touches another one
(Re = 100 at alpha = 1.6 .. 1.8, Re = 60 at 1.4 .. 1.5, everything at alpha = 2).  Re = 100 and 160 at alpha = 0
are hidden by the axis: the alpha = 0.1 value is repeated there (the curves are flat).""",
      ["alpha", "Cdf_Re40", "Cdf_Re60", "Cdf_Re100", "Cdf_Re160"], table(CDF))
CDP = {r: {a: round(CD[r][a] - CDF[r][a], 3) for a in CD[r] if a in CDF[r]} for r in CD}
write("kang1999_fig8b_drag_pressure.dat", """
Kang, Choi & Lee, Phys. Fluids 11, 3312 (1999), FIG. 8(b), PRESSURE part of the mean drag.
NOT digitized directly (the dash-dot curves of the four Re overlap): DERIVED as total - friction from
kang1999_fig8b_mean_drag.dat and kang1999_fig8b_drag_friction.dat, so the uncertainty is +-0.012 (+-0.02 where
Re = 100 and 160 share one total, alpha <= 1.6).  The pressure drag changes sign near alpha = 2.2 - 2.4.
The lift parts of FIG. 8(a) are not digitized either: use kang1999_eq11_mean_lift_fit.dat (pressure and friction
slopes, alpha <= 1).""",
      ["alpha", "Cdp_Re40", "Cdp_Re60", "Cdp_Re100", "Cdp_Re160"], table(CDP))

# ------------------------------------------------------------------------------------------------ Munir et al. 2021
write("munir2021_table2_Re100_alpha5.dat", """
Munir, Zhao, Wu & Tong, Ocean Engineering 238, 109562 (2021), TABLE 2 (copied; exact): non-vibrating rotating
cylinder, Re = 100, alpha = 5 (second shedding mode).  Row 1: Bourguet (2020) as quoted there; row 2: Munir et al.
The table's Stojkovic et al. (2002) row is blank in the PDF text; Stojkovic FIG. 13 digitizes to Cl = -26.5 at
alpha = 5 (stojkovic2002_fig13_Re100.dat).  Frequency is f D / U of the mode-II shedding.""",
      ["row", "frequency", "Cd_mean", "Cl_mean"], [(1, 0.022, 0.32, -26.60), (2, 0.022, 0.329, -26.5)])
