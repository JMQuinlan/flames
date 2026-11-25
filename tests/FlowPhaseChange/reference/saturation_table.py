#!/usr/bin/env python3
"""
Saturation curve of the material pairs in data/hydro2_materials.dat against steam-table values.

  python3 saturation_table.py [path/to/hydro2_materials.dat]

p_sat(T) from g_liquid(p, T) = g_vapour(p, T) with the stiffened-gas Gibbs energy used by
Hydro2_PhaseChange.H; latent heat h_v - h_l = (gamma_v cv_v - gamma_l cv_l) T + q_v - q_l.
Table: IAPWS-IF97, rounded.  No solver run: this checks the MATERIAL data, not the code.
"""
import sys, os
import numpy as np
from scipy.optimize import brentq
f = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data/hydro2_materials.dat")
M = {}
for l in open(f):
    w = l.split("#")[0].split()
    if len(w) >= 7: M[w[0]] = dict(zip(("g", "pi", "cv", "q", "qp"), map(float, w[2:7])))
gib = lambda m, p, T: (m["g"] * m["cv"] - m["qp"]) * T - m["cv"] * T * (m["g"] * np.log(T) - (m["g"] - 1) * np.log(p + m["pi"])) + m["q"]
def psat(L, V, T):
    try: return np.exp(brentq(lambda lp: gib(L, np.exp(lp), T) - gib(V, np.exp(lp), T), 0.0, np.log(5e8)))
    except ValueError: return np.nan
lat = lambda L, V, T: (V["g"] * V["cv"] - L["g"] * L["cv"]) * T + V["q"] - L["q"]
TAB = [(300, 3537, 2.437e6), (325, 13531, 2.378e6), (350, 41682, 2.316e6), (373.15, 101325, 2.257e6), (400, 245770, 2.183e6), (450, 932200, 2.024e6), (500, 2639200, 1.827e6)]
for a, b in [("water_liquid", "water_vapor"), ("water_liquid", "steam_as_air"), ("water_tammann", "steam_as_air_tammann")]:
    print(f"{a} + {b}")
    for T, ps, h in TAB:
        p = psat(M[a], M[b], T); L = lat(M[a], M[b], T)
        print(f"   T = {T:6.2f} K: p_sat {p:12.1f} Pa   table {ps:9.0f}  ({100 * (p / ps - 1):+6.1f} %)    latent {L:.3e}   table {h:.3e}  ({100 * (L / h - 1):+5.1f} %)")
