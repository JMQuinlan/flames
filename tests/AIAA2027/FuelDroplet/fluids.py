# Liquid properties for make_fuel_decks.py (exec'd there; defines FLUIDS).
#
# Data: NIST Chemistry WebBook saturation tables (data/NIST_*_sat_*.tsv; n-dodecane from the Lemmon & Huber
# reference EOS, n-pentane from Span & Wagner).
#
# Stiffened gas (SG) for the liquid, fitted at the initial state (T0, 1 atm) so that density rho, sound speed c,
# cv and T are all exact there:
#     (p + pinf) = (gamma - 1) rho cv T     and     rho c^2 = gamma (p + pinf)
#  => gamma (gamma - 1) = c^2 / (cv T),   pinf = (gamma - 1) rho cv T - p.
# No published SG fit exists for n-pentane (Le Metayer & Saurel 2016 and Chiapolino & Saurel 2018 cover only
# dodecane, water and oxygen).  For n-dodecane the fit gives gamma 2.29, pinf 5.26e8 Pa, close to the widely used
# set gamma 2.35, pinf 4e8 Pa (Pelanti & Shyue 2014, JCP 259, Table 2, attributed to Le Metayer et al. 2004).
# SG cannot match cp as well (cp_SG = gamma cv); cp only enters through eos1.cp, which the film model does not use.
#
# Saturation pressure: Antoine fit log10(p/bar) = A - B/(T + C) by least squares on the NIST table (T0 .. 0.95 Tc).
# Lv: NIST value at T0 (the film model takes a constant Lv; it falls ~10-30 % by the boiling point).
# cp_vapor: NIST saturated-vapour cp at the stated temperature (near the film temperature at moderate Mach).
# sigma: NIST at T0.  V_fuller: Fuller diffusion volume, C 15.9 + H 2.31 per atom (Poling et al. 2001, Table 11-1).
import csv as _csv, math as _m


def _load(fn):
    rows = []
    with open(os.path.join(HERE, "data", fn)) as fh:
        r = _csv.reader(fh, delimiter="\t"); hdr = next(r)
        for row in r:
            try: rows.append([float(x) for x in row[:25]])
            except ValueError: pass
    return hdr, rows


def _col(hdr, key): return [i for i, h in enumerate(hdr) if h.startswith(key)][0]


def _fit_antoine(T, p):
    """Least-squares Antoine fit, log10(p/bar) = A - B/(T + C): scan C, linear fit for A, B."""
    best = None
    y = [_m.log10(x / 1e5) for x in p]
    for Ci in range(-200, 101):
        C = float(Ci)
        X = [-1.0 / (t + C) for t in T]
        n = len(T); sx = sum(X); sy = sum(y); sxx = sum(a * a for a in X); sxy = sum(a * b for a, b in zip(X, y))
        B = (n * sxy - sx * sy) / (n * sxx - sx * sx); A = (sy - B * sx) / n
        err = max(abs(A - B / (t + C) - yy) for t, yy in zip(T, y))
        if best is None or err < best[3]: best = (A, B, C, err)
    return best


def _fluid(name, short, fn, W, nC, nH, T0, Tc, T_cpv):
    hdr, rows = _load(fn)
    iT, ip, ir, icv, ic = _col(hdr, "Temperature"), _col(hdr, "Pressure"), _col(hdr, "Density (l"), \
        _col(hdr, "Cv (l"), _col(hdr, "Sound Spd. (l")
    ih_l, ih_v, isig, icpv = _col(hdr, "Enthalpy (l"), _col(hdr, "Enthalpy (v"), _col(hdr, "Surf. Tension"), _col(hdr, "Cp (v")
    imu = _col(hdr, "Viscosity (l")
    r0 = min(rows, key=lambda r: abs(r[iT] - T0))
    assert abs(r0[iT] - T0) < 1e-6, f"{name}: no NIST row at T0 = {T0}"
    rho, c, cv = r0[ir], r0[ic], r0[icv] * 1e3
    gamma = 0.5 * (1.0 + _m.sqrt(1.0 + 4.0 * c * c / (cv * T0)))
    pinf = (gamma - 1.0) * rho * cv * T0 - 101325.0
    fit = [r for r in rows if T0 <= r[iT] <= 0.95 * Tc]
    A, B, C, err = _fit_antoine([r[iT] for r in fit], [r[ip] for r in fit])
    rv = min(rows, key=lambda r: abs(r[iT] - T_cpv))
    return dict(name=name, short=short, W=W, V_fuller=15.9 * nC + 2.31 * nH, T0=T0,
                rho=rho, c=c, cv=cv, mu=r0[imu], gamma=gamma, pinf=pinf,
                Lv=(r0[ih_v] - r0[ih_l]) * 1e3, sigma=r0[isig], cp_vapor=rv[icpv] * 1e3,
                antoine=(A, B, C), antoine_maxerr_log10=err, antoine_range=(fit[0][iT], fit[-1][iT]),
                src=f"fit to NIST at {T0:g} K, 1 atm (rho, c, cv, T exact): gamma {gamma:.4f}, pinf {pinf:.4e} Pa, "
                    f"cv {cv:.1f}")


FLUIDS["dodecane"] = _fluid("n-dodecane", "C12", "NIST_dodecane_sat_300-650K.tsv", 170.3348, 12, 26, 300.0, 658.1, 400.0)
# n-pentane critical temperature 469.7 K (Span & Wagner EOS, NIST WebBook)
FLUIDS["pentane"] = _fluid("n-pentane", "C5", "NIST_pentane_sat_200-460K.tsv", 72.1488, 5, 12, 300.0, 469.7, 320.0)
