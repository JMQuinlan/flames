#ifndef INTEGRATOR_HYDRO2_PHASECHANGE_H
#define INTEGRATOR_HYDRO2_PHASECHANGE_H
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include "Set/Base.H"
#include "IO/ParmParse.H"
#include "Util/Util.H"
namespace Integrator {
// Hydro2_PhaseChange.H -- LIQUID <-> VAPOUR PHASE CHANGE for Hydro2, default OFF (phasechange.on = 0).
//
// Thermodynamic-equilibrium (p-T-g) relaxation, the standard closure for phase transition in the
// six-equation model: R. Saurel, F. Petitpas, R. Abgrall, J. Fluid Mech. 607 (2008) 313;
// M. Pelanti, K.-M. Shyue, J. Comput. Phys. 259 (2014) 331.  After the hydrodynamic step and the
// pressure relaxation, every cell that holds liquid is brought to the state in which liquid and
// vapour have the same pressure, temperature and Gibbs free energy, at fixed mass, volume and
// total energy.  Nothing is tuned: there is no rate constant and no mass-diffusion coefficient;
// the transfer is driven by temperature, energy and the pressure field the flow produces
// (expansion -> cavitation / flashing, compression or cooling -> condensation).
//
// Equation of state of each phase: stiffened gas with energy and entropy constants
// (O. Le Metayer, J. Massoni, R. Saurel, Int. J. Therm. Sci. 43 (2004) 265):
//     e(p, rho) = (p + gamma pi) / ((gamma - 1) rho) + q
//     T(p, rho) = (p + pi) / ((gamma - 1) cv rho)
//     g(p, T)   = (gamma cv - q') T - cv T ln( T^gamma / (p + pi)^(gamma - 1) ) + q
// gamma, pi are the phase's Hydro2 EOS (eos<k>.gamma, eos<k>.p0), cv is eos<k>.cv; q and q' come
// from the material data file.  NOTE: Hydro2's plotted temperature uses the Tammann convention
// T = (p + gamma pi)/((gamma - 1) rho cv); for a gas (pi = 0) the two agree, for a liquid they
// do not.  Phase change uses the form above throughout (field `pc_T`).
//
// NON-CONDENSABLE GAS.  The gas phase is one fluid (air and vapour share one EOS -- "steam has
// the properties of air"), but only its VAPOUR part can condense, and it is the vapour PARTIAL
// pressure that enters the equilibrium.  The partial density of vapour in the gas phase,
//     m_vap = (alpha rho)_gas Y_v,
// is transported as an extra conserved field (flux = gas mass flux x upwind Y_v), and the
// saturation condition is
//     g_liquid(p, T) = g_vapour(x_v p, T),     x_v = Y_v     (equal molar masses assumed).
// So air at room temperature does not condense, a droplet in air only humidifies its interface
// cells, and a liquid whose pressure falls below p_sat(T) cavitates into (nearly) pure vapour.
//
// Cell solve (all algebraic, nested bisection; unknowns p, T and the vapour mass m_vap):
//     mass      m_liq + m_gas = rho                       (m_air fixed)
//     volume    m_liq v_liq(p, T) + m_gas v_gas(p, T) = 1
//     energy    sum_k m_k e_k(p, T) = rho e               (includes the q_k: latent heat)
//     Gibbs     g_liq(p, T) = g_vap(x_v p, T)             (or m_vap = 0 / m_liq = trace at a bound)
// The hydrodynamic phase energies of Hydro2 exclude q, so the conserved rho E changes by
// -(m_vap_new - m_vap_old)(q_vap - q_liq): that is the latent heat.
//
// Material data file (phasechange.material_file, default data/hydro2_materials.dat), one
// material per line:    name  phase  gamma  pi[Pa]  cv[J/kg/K]  q[J/kg]  q'[J/kg/K]
// phasechange.liquid_material / phasechange.vapor_material pick the two entries; gamma, pi and cv
// in the file must match the deck's EOS (checked, 1 %) unless phasechange.use_file_eos = 1 is
// used only to read q and q'.  Alternatively phasechange.q_liquid etc. can be given directly.
class PhaseChange {
public:
    struct Phase { Set::Scalar g = 1.4, pi = 0.0, cv = 1.0, q = 0.0, qp = 0.0; };

    int          on = 0;
    int          liquid = 1;               // Hydro2 phase index of the liquid (0: eta = 1 phase, 1: eta = 0 phase)
    Phase        L, V;                     // liquid, vapour (= gas) thermodynamic constants
    Set::Scalar  alpha_min = 1.0e-9;       // cells with less liquid OR less gas volume than this are left alone
    Set::Scalar  T_lo = 50.0, T_hi = 5000.0; // bracket for temperatures in the solves
    int          evaporate = 1, condense = 1;  // allow each direction
    Set::Scalar  Yv0 = 0.0;                // initial vapour mass fraction of the gas phase (0 = dry gas, 1 = pure vapour)
    Set::Scalar  trace = 1.0e-9;           // liquid mass fraction kept when a cell evaporates completely
    std::string  material_file = "data/hydro2_materials.dat", liquid_material = "", vapor_material = "";
    Set::Field<Set::Scalar> vapor_mf, vapor_old_mf;   // m_vap = (alpha rho)_gas Y_v
    Set::Field<Set::Scalar> diag_mf;                  // plot: pc_T, pc_Tsat, pc_Yv, pc_mdot (mass evaporated in the last relaxation)

    // ------------------------------------------------------------------ thermodynamics
    AMREX_GPU_HOST_DEVICE static Set::Scalar vol(const Phase &m, Set::Scalar p, Set::Scalar T)       // specific volume
    { return (m.g - 1.0) * m.cv * T / (p + m.pi); }
    AMREX_GPU_HOST_DEVICE static Set::Scalar ener(const Phase &m, Set::Scalar p, Set::Scalar T)      // specific internal energy incl. q
    { return m.cv * T * (p + m.g * m.pi) / (p + m.pi) + m.q; }
    AMREX_GPU_HOST_DEVICE static Set::Scalar temp(const Phase &m, Set::Scalar p, Set::Scalar rho)
    { return (p + m.pi) / ((m.g - 1.0) * m.cv * rho); }
    AMREX_GPU_HOST_DEVICE static Set::Scalar gibbs(const Phase &m, Set::Scalar p, Set::Scalar T)
    { return (m.g * m.cv - m.qp) * T - m.cv * T * (m.g * std::log(T) - (m.g - 1.0) * std::log(p + m.pi)) + m.q; }
    // saturation temperature of the pure substance at pressure p (g_liq = g_vap); T_hi if none below it
    AMREX_GPU_HOST_DEVICE Set::Scalar Tsat(Set::Scalar p) const
    {
        Set::Scalar a = T_lo, b = T_hi;
        auto f = [&](Set::Scalar T) { return gibbs(L, p, T) - gibbs(V, p, T); };   // < 0: liquid stable
        Set::Scalar fa = f(a), fb = f(b);
        if (fa > 0.0) return a;
        if (fb < 0.0) return b;
        // bracketed regula falsi (Illinois); this is called in every cell for the pc_Tsat diagnostic, and 80
        // bisections there cost 2.5x the whole flow solver (atomization deck: 3.6x -> see changelog II.22)
        for (int it = 0; it < 80 && (b - a) > 1.0e-13 * b; ++it)
        {
            Set::Scalar c = (fb != fa) ? b - fb * (b - a) / (fb - fa) : 0.5 * (a + b);
            if (!(c > a && c < b)) c = 0.5 * (a + b);
            const Set::Scalar fc = f(c);
            if (fc == 0.0) return c;
            if (fc < 0.0) { a = c; fa = fc; fb *= 0.5; }
            else          { b = c; fb = fc; fa *= 0.5; }
        }
        return 0.5 * (a + b);
    }
    // latent heat at temperature T:  h_vap - h_liq,  h = gamma cv T + q
    AMREX_GPU_HOST_DEVICE Set::Scalar latent(Set::Scalar T) const { return (V.g * V.cv - L.g * L.cv) * T + V.q - L.q; }

    // Pressure-temperature equilibrium of ml of liquid and mg of gas (per unit volume) with total
    // internal energy rhoe (incl. q).  T(p) follows from the energy; the volume residual
    // ml v_l + mg v_g - 1 decreases with p.  Returns false if no root in the bracket.
    AMREX_GPU_HOST_DEVICE bool ptEquilibrium(Set::Scalar ml, Set::Scalar mg, Set::Scalar rhoe, Set::Scalar pguess,
                                             Set::Scalar &p, Set::Scalar &T) const
    {
        auto Tof = [&](Set::Scalar pp) {
            const Set::Scalar den = ml * L.cv * (pp + L.g * L.pi) / (pp + L.pi) + mg * V.cv * (pp + V.g * V.pi) / (pp + V.pi);
            return (rhoe - ml * L.q - mg * V.q) / den;
        };
        auto res = [&](Set::Scalar pp) { const Set::Scalar TT = Tof(pp); return ml * vol(L, pp, TT) + mg * vol(V, pp, TT) - 1.0; };
        // fast path: secant from the present pressure (a relaxed cell is close to the answer)
        {
            Set::Scalar p0 = pguess, p1 = pguess * (1.0 + 1.0e-4) + 1.0e-8, f0 = res(p0), f1 = res(p1);
            bool conv = false;
            for (int it = 0; it < 25 && f1 == f1 && f0 == f0; ++it)
            {
                if (std::abs(f1) < 1.0e-13) { conv = true; break; }
                const Set::Scalar d = f1 - f0;
                if (d == 0.0) break;
                const Set::Scalar p2 = p1 - f1 * (p1 - p0) / d;
                if (!(p2 > -std::min(L.pi, V.pi)) || !(p2 == p2)) break;
                p0 = p1; f0 = f1; p1 = p2; f1 = res(p1);
            }
            if (conv && Tof(p1) > 0.0) { p = p1; T = Tof(p1); return true; }
        }
        const Set::Scalar pfloor = -std::min(L.pi, V.pi) * (1.0 - 1.0e-9) + ((std::min(L.pi, V.pi) > 0.0) ? 0.0 : 1.0e-9 * std::abs(pguess) + 1.0e-300);
        Set::Scalar a = std::max(pfloor, 1.0e-6 * std::abs(pguess)), b = std::max(2.0 * std::abs(pguess), 2.0 * a);
        int n = 0;
        while (res(b) > 0.0 && n < 200) { a = b; b *= 4.0; ++n; }
        if (res(b) > 0.0 || res(a) < 0.0 || !(Tof(a) > 0.0)) { a = pfloor; if (res(a) < 0.0 || res(b) > 0.0) return false; }
        for (int it = 0; it < 100; ++it) { const Set::Scalar c = 0.5 * (a + b); if (res(c) > 0.0) a = c; else b = c; }
        p = 0.5 * (a + b); T = Tof(p);
        return T > 0.0;
    }

    // Full relaxation of one cell.  In:  ml, mg (phase masses per volume), mvap (vapour part of mg),
    // rhoe_h = E_l + E_g (Hydro2's hydrodynamic phase energies, WITHOUT q), p_in (current pressure).
    // Out: the equilibrium ml, mg, mvap, alpha_liquid, p, T and dE = change of the hydrodynamic
    // energy sum (= -latent heat absorbed).  Returns false (nothing changed) if no solution.
    AMREX_GPU_HOST_DEVICE bool relax(Set::Scalar &ml, Set::Scalar &mg, Set::Scalar &mvap, Set::Scalar rhoe_h, Set::Scalar p_in,
                                     Set::Scalar &alpha_l, Set::Scalar &p, Set::Scalar &T, Set::Scalar &dE) const
    {
        const Set::Scalar rho = ml + mg, mair = std::max(mg - mvap, 0.0), mvap0 = std::min(std::max(mvap, 0.0), mg);
        const Set::Scalar rhoe = rhoe_h + ml * L.q + mg * V.q;
        // r(x) for x = vapour mass: > 0 means the liquid has the higher potential -> evaporate more
        auto state = [&](Set::Scalar x, Set::Scalar &pp, Set::Scalar &TT) -> bool {
            return ptEquilibrium(rho - mair - x, mair + x, rhoe, p_in, pp, TT);
        };
        auto r = [&](Set::Scalar x, bool &ok) -> Set::Scalar {
            Set::Scalar pp, TT; ok = state(x, pp, TT);
            if (!ok) return 0.0;
            const Set::Scalar xv = x / std::max(mair + x, 1.0e-300);
            return gibbs(L, pp, TT) - gibbs(V, std::max(xv * (pp + V.pi) - V.pi, -V.pi + 1.0e-300 + 1.0e-12 * std::abs(pp)), TT);
        };
        const Set::Scalar xlo = (condense ? 1.0e-14 * rho : mvap0), xhi_all = rho - mair - trace * rho;
        const Set::Scalar xhi = (evaporate ? std::max(xhi_all, mvap0) : mvap0);
        // Start from the present composition (its p-T state exists: it is the mechanically relaxed
        // cell) and walk in the direction the Gibbs difference asks for, doubling the step, until
        // r changes sign, the bound is reached, or no p-T state exists (the cell cannot hold that
        // split at its energy: treated as "past the root").  Then bisect.  r decreases with x.
        bool ok;
        Set::Scalar x0 = std::min(std::max(mvap0, xlo), xhi);
        Set::Scalar r0 = r(x0, ok);
        if (!ok) return false;
        // already at equilibrium (to 1e-10 of the thermal scale of g): nothing to do
        if (std::abs(r0) < 1.0e-10 * (L.cv + V.cv) * std::max(T_lo, 1.0)) return false;
        const Set::Scalar dir = (r0 > 0.0) ? 1.0 : -1.0, bound = (r0 > 0.0) ? xhi : xlo;
        if (!((bound - x0) * dir > 0.0)) return false;
        Set::Scalar good = x0, rgood = r0, bad = bound, rbad = 0.0; bool bracket = false, bad_ok = false;
        // first step: Newton-like estimate from a small probe, else a fraction of the way to the bound
        Set::Scalar step = std::max(1.0e-9 * rho, 1.0e-3 * std::abs(bound - x0));
        {
            const Set::Scalar xp = x0 + dir * std::min(step, 0.5 * std::abs(bound - x0));
            const Set::Scalar rp = r(xp, ok);
            if (ok && rp != r0) { const Set::Scalar est = std::abs(r0 * (xp - x0) / (rp - r0)); if (est == est && est > 0.0) step = std::min(std::max(1.2 * est, step), std::abs(bound - x0)); }
        }
        for (int it = 0; it < 80; ++it)
        {
            Set::Scalar xt = good + dir * step;
            const bool at_bound = ((bound - xt) * dir <= 0.0);
            if (at_bound) xt = bound;
            const Set::Scalar rt = r(xt, ok);
            if (!ok || rt * dir <= 0.0) { bad = xt; rbad = rt; bad_ok = ok; bracket = true; break; }
            good = xt; rgood = rt;
            if (at_bound) break;
            step *= 2.0;
        }
        Set::Scalar x = good;
        if (bracket)
        {
            // Illinois regula falsi while both ends have a p-T state, plain bisection otherwise
            int side = 0;
            for (int it = 0; it < 90; ++it)
            {
                if (std::abs(bad - good) <= 1.0e-14 * rho + 1.0e-13 * std::abs(good)) break;
                Set::Scalar c = 0.5 * (good + bad);
                if (bad_ok && rgood != rbad) { const Set::Scalar cf = (good * rbad - bad * rgood) / (rbad - rgood); if ((cf - good) * (cf - bad) < 0.0) c = cf; }
                const Set::Scalar rc = r(c, ok);
                if (ok && rc * dir > 0.0) { good = c; rgood = rc; if (side == 1 && bad_ok) rbad *= 0.5; side = 1; }
                else { bad = c; rbad = rc; bad_ok = ok; if (side == -1) rgood *= 0.5; side = -1; }
            }
            x = good;
        }
        Set::Scalar pp, TT;
        if (!state(x, pp, TT)) return false;
        const Set::Scalar ml_n = rho - mair - x, mg_n = mair + x;
        dE = -(mg_n - mg) * (V.q - L.q);
        ml = ml_n; mg = mg_n; mvap = x; p = pp; T = TT; alpha_l = std::min(std::max(ml_n * vol(L, pp, TT), 0.0), 1.0);
        return true;
    }

    // ------------------------------------------------------------------ input
    static bool readMaterial(const std::string &file, const std::string &name, Phase &m)
    {
        std::ifstream in(file);
        if (!in.good()) return false;
        std::string line;
        while (std::getline(in, line))
        {
            const auto h = line.find('#'); if (h != std::string::npos) line = line.substr(0, h);
            std::istringstream ss(line); std::string nm, ph;
            if (!(ss >> nm >> ph)) continue;
            if (nm != name) continue;
            if (ss >> m.g >> m.pi >> m.cv >> m.q >> m.qp) return true;
        }
        return false;
    }
    // gamma0/pi0/cv0, gamma1/pi1/cv1: the deck's EOS of phases 0 and 1
    static void Parse(PhaseChange &value, IO::ParmParse &pp, Set::Scalar g0, Set::Scalar pi0, Set::Scalar cv0,
                      Set::Scalar g1, Set::Scalar pi1, Set::Scalar cv1)
    {
        pp_query_default("phasechange.on", value.on, 0);
        if (!value.on) return;
        pp_query_default("phasechange.liquid", value.liquid, 1);                 // Hydro2 phase index of the liquid
        pp_query_default("phasechange.material_file", value.material_file, std::string("data/hydro2_materials.dat"));
        pp_query_default("phasechange.liquid_material", value.liquid_material, std::string(""));
        pp_query_default("phasechange.vapor_material", value.vapor_material, std::string(""));
        pp_query_default("phasechange.alpha_min", value.alpha_min, 1.0e-9);      // cells with less liquid or gas volume than this are skipped
        pp_query_default("phasechange.T_lo", value.T_lo, 50.0);                  // temperature bracket of the solves
        pp_query_default("phasechange.T_hi", value.T_hi, 5000.0);
        pp_query_default("phasechange.Yv0", value.Yv0, 0.0);                     // initial vapour mass fraction of the gas (0 dry, 1 pure vapour)
        pp_query_default("phasechange.evaporate", value.evaporate, 1);           // 0 disables liquid -> vapour
        pp_query_default("phasechange.condense", value.condense, 1);             // 0 disables vapour -> liquid
        Phase &Lq = value.L, &Vp = value.V;
        if (value.liquid == 1) { Lq.g = g1; Lq.pi = pi1; Lq.cv = cv1; Vp.g = g0; Vp.pi = pi0; Vp.cv = cv0; }
        else                   { Lq.g = g0; Lq.pi = pi0; Lq.cv = cv0; Vp.g = g1; Vp.pi = pi1; Vp.cv = cv1; }
        auto fromFile = [&](const std::string &nm, Phase &m, const char *what) {
            Phase f;
            if (!readMaterial(value.material_file, nm, f))
                Util::Abort(INFO, "phasechange: material '", nm, "' not found in ", value.material_file);
            auto off = [](Set::Scalar a, Set::Scalar b) { return std::abs(a - b) > 0.01 * std::max(std::abs(a), std::abs(b)) + 1.0e-300; };
            if (off(f.g, m.g) || off(f.pi, m.pi) || off(f.cv, m.cv))
                Util::Abort(INFO, "phasechange: ", what, " material '", nm, "' (gamma ", f.g, ", pi ", f.pi, ", cv ", f.cv,
                            ") does not match the deck's EOS (gamma ", m.g, ", pi ", m.pi, ", cv ", m.cv, ")");
            m.q = f.q; m.qp = f.qp;
        };
        if (!value.liquid_material.empty()) fromFile(value.liquid_material, Lq, "liquid");
        if (!value.vapor_material.empty())  fromFile(value.vapor_material,  Vp, "vapour");
        pp_query_default("phasechange.q_liquid",  Lq.q,  Lq.q);                  // direct override of the energy / entropy constants
        pp_query_default("phasechange.qp_liquid", Lq.qp, Lq.qp);
        pp_query_default("phasechange.q_vapor",   Vp.q,  Vp.q);
        pp_query_default("phasechange.qp_vapor",  Vp.qp, Vp.qp);
        Util::Message(INFO, "phasechange: liquid = phase ", value.liquid, " (gamma ", Lq.g, ", pi ", Lq.pi, ", cv ", Lq.cv, ", q ", Lq.q, ", q' ", Lq.qp,
                      "); vapour (gamma ", Vp.g, ", pi ", Vp.pi, ", cv ", Vp.cv, ", q ", Vp.q, ", q' ", Vp.qp, ")");
        Util::Message(INFO, "phasechange: T_sat(101325) = ", value.Tsat(101325.0), ", latent heat there = ", value.latent(value.Tsat(101325.0)));
    }
};
}
#endif
