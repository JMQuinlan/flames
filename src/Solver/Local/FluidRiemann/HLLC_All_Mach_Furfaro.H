// ============================================================================
// HLLC_Furfaro_TwoPhase.H
//
// 5-wave HLLC-type Riemann solver for two-phase non-equilibrium flows
// following D. Furfaro & R. Saurel, JCP 2015.
//
// Wave structure:
//   S_L | U_L* | S_I | U_R* | S_R
//
// Key features:
//   • Per-phase thermodynamics (no pressure/velocity equilibrium)
//   • Stiffened gas EOS (Tammann) compatible
//   • Two contact discontinuities (one per phase)
//   • Interface wave for phase interaction
//   • All-Mach capable
//
// ============================================================================

#ifndef SOLVER_LOCAL_FLUIDRIEMANN_HLLC_ALL_MACH_FURFARO_H
#define SOLVER_LOCAL_FLUIDRIEMANN_HLLC_ALL_MACH_FURFARO_H

#include "IO/ParmParse.H"
#include "Solver/Local/FluidRiemann/Riemann.H"
// #include "Util/Util.H"
#include "Solver/EOS/EOS.H"

namespace Solver
{
namespace Local
{
namespace FluidRiemann
{

class HLLC_All_Mach_Furfaro : public FluidRiemann
{
public:
    static constexpr const char *name = "hllc_all_mach_furfaro";

    HLLC_All_Mach_Furfaro(IO::ParmParse &pp, std::string name) { pp_queryclass(name, *this); }
    HLLC_All_Mach_Furfaro(IO::ParmParse &pp) { pp_queryclass(*this); }
    HLLC_All_Mach_Furfaro()
    {
        IO::ParmParse pp;
        pp_queryclass(*this);
    }

    Set::Scalar interface_pressure_relaxation = 1.0; // β parameter (0=no relax, 1=full relax)

    static void Parse(HLLC_All_Mach_Furfaro &value, IO::ParmParse &pp)
    {
        pp.query_default("interface_pressure_relaxation", value.interface_pressure_relaxation, 1.0);
    }

    // -------------------------------------------------------------------------
    // Solve two-phase Riemann problem
    // -------------------------------------------------------------------------
    virtual Flux Solve(State lo, State hi, Set::Scalar p_ref, Set::Scalar small, int Spec_Vol = 1) override
    {
        // =====================================================================
        // 1. EXTRACT & SANITIZE STATES
        // =====================================================================
        Set::Scalar rho_L = std::max(lo.rho, small);
        Set::Scalar rho_R = std::max(hi.rho, small);
        Set::Scalar Mn_L = lo.M_normal;
        Set::Scalar Mn_R = hi.M_normal;
        Set::Scalar Mt_L = lo.M_tangent;
        Set::Scalar Mt_R = hi.M_tangent;
        Set::Scalar E_L = lo.E;
        Set::Scalar E_R = hi.E;
        Set::Scalar gamma_L = lo.gam;
        Set::Scalar gamma_R = hi.gam;
        Set::Scalar p0_L = lo.p0;
        Set::Scalar p0_R = hi.p0;
        Set::Scalar T_L = lo.T;
        Set::Scalar T_R = hi.T;

        // =====================================================================
        // 2. COMPUTE PRIMITIVES & THERMODYNAMICS
        // =====================================================================
        Set::Scalar ke_L, ke_R, ue_L, ue_R, p_L, p_R;

        if (Spec_Vol == 1)
        {
            // Energies per unit volume [J/m³]
            ke_L = 0.5 * (Mn_L * Mn_L + Mt_L * Mt_L) / (rho_L + small);
            ke_R = 0.5 * (Mn_R * Mn_R + Mt_R * Mt_R) / (rho_R + small);
            ue_L = std::max(E_L - ke_L, small); // IE per unit volume
            ue_R = std::max(E_R - ke_R, small);
            p_L = Solver::EOS::EOS::TammannPressure(rho_L, ue_L, gamma_L, p0_L, p_ref, small); // (gamma_L - 1.0) * ue_L - gamma_L * p0_L + p_ref;
            p_R = Solver::EOS::EOS::TammannPressure(rho_R, ue_R, gamma_R, p0_R, p_ref, small); // (gamma_R - 1.0) * ue_R - gamma_R * p0_R + p_ref;
        }
        else
        {
            // Energies per unit mass [J/kg]
            ke_L = 0.5 * (Mn_L * Mn_L + Mt_L * Mt_L) / (rho_L * rho_L + small);
            ke_R = 0.5 * (Mn_R * Mn_R + Mt_R * Mt_R) / (rho_R * rho_R + small);
            ue_L = E_L - ke_L;
            ue_R = E_R - ke_R;
            p_L = (gamma_L - 1.0) * rho_L * ue_L - gamma_L * p0_L + p_ref;
            p_R = (gamma_R - 1.0) * rho_R * ue_R - gamma_R * p0_R + p_ref;
        }

        // Acoustic Speeds
        Set::Scalar u_L = Mn_L / (rho_L + small), u_R = Mn_R / (rho_R + small);
        Set::Scalar v_L = Mt_L / (rho_L + small), v_R = Mt_R / (rho_R + small);

        // Limiting
        Set::Scalar u_limit = 1e8;
        u_L = (u_L < 0.0) ? std::max(u_L, -u_limit) : std::min(u_L, u_limit);
        v_L = (v_L < 0.0) ? std::max(v_L, -u_limit) : std::min(v_L, u_limit);
        u_R = (u_R < 0.0) ? std::max(u_R, -u_limit) : std::min(u_R, u_limit);
        v_R = (v_R < 0.0) ? std::max(v_R, -u_limit) : std::min(v_R, u_limit);

        // Sound Speeds
        Set::Scalar a_L = Solver::EOS::EOS::TammannSoundSpeed(rho_L, p_L, gamma_L, p0_L, small);
        Set::Scalar a_R = Solver::EOS::EOS::TammannSoundSpeed(rho_R, p_R, gamma_R, p0_R, small);

        // =====================================================================
        // 3. WAVE SPEED ESTIMATES (5-wave structure)
        // =====================================================================
        // Outer acoustic waves (same as standard HLLC)
        Set::Scalar S_L = std::min(u_L - a_L, u_R - a_R);
        Set::Scalar S_R = std::max(u_L + a_L, u_R + a_R);

        // Per-phase contact velocities (Furfaro Eq. 18-19)
        Set::Scalar S_star_L = u_L - (p_L - p_R) / (rho_L * (S_L - u_L) + small);
        Set::Scalar S_star_R = u_R + (p_L - p_R) / (rho_R * (S_R - u_R) + small);

        // Interface velocity (mass-weighted average with relaxation)
        Set::Scalar rho_bar_L = rho_L * (S_L - u_L) / (S_L - S_star_L + small);
        Set::Scalar rho_bar_R = rho_R * (S_R - u_R) / (S_R - S_star_R + small);
        Set::Scalar S_I = (rho_bar_L * S_star_L + rho_bar_R * S_star_R)
                          / (rho_bar_L + rho_bar_R + small);

        // =====================================================================
        // 4. STAR STATES (per phase, no equilibrium)
        // =====================================================================
        // Left star region
        Set::Scalar rho_star_L = rho_L * (S_L - u_L) / (S_L - S_star_L + small);
        Set::Scalar p_star_L = p_L + rho_L * (S_L - u_L) * (S_star_L - u_L);
        Set::Scalar E_star_L;

        if (Spec_Vol == 1)
        {
            E_star_L = rho_star_L * (E_L / rho_L + (S_star_L - u_L) * (S_star_L + p_L / (rho_L * (S_L - u_L) + small)));
        }
        else
        {
            E_star_L = E_L + (S_star_L - u_L) * (S_star_L + p_L / (rho_L * (S_L - u_L) + small));
        }

        // Right star region
        Set::Scalar rho_star_R = rho_R * (S_R - u_R) / (S_R - S_star_R + small);
        Set::Scalar p_star_R = p_R + rho_R * (S_R - u_R) * (S_star_R - u_R);
        Set::Scalar E_star_R;

        if (Spec_Vol == 1)
        {
            E_star_R = rho_star_R * (E_R / rho_R + (S_star_R - u_R) * (S_star_R + p_R / (rho_R * (S_R - u_R) + small)));
        }
        else
        {
            E_star_R = E_R + (S_star_R - u_R) * (S_star_R + p_R / (rho_R * (S_R - u_R) + small));
        }

        // Interface star states (with pressure relaxation)
        Set::Scalar p_I = interface_pressure_relaxation * 0.5 * (p_star_L + p_star_R)
                          + (1.0 - interface_pressure_relaxation) * p_star_L; // fallback to left

        Set::Scalar rho_star_I_L = rho_star_L; // density continuous across contact
        Set::Scalar rho_star_I_R = rho_star_R;
        Set::Scalar E_star_I_L, E_star_I_R;

        if (Spec_Vol == 1)
        {
            E_star_I_L = E_star_L + (p_I - p_star_L) / (gamma_L - 1.0);
            E_star_I_R = E_star_R + (p_I - p_star_R) / (gamma_R - 1.0);
        }
        else
        {
            E_star_I_L = E_star_L + (p_I - p_star_L) / (rho_star_I_L * (gamma_L - 1.0) + small);
            E_star_I_R = E_star_R + (p_I - p_star_R) / (rho_star_I_R * (gamma_R - 1.0) + small);
        }

        // =====================================================================
        // 5. COMPUTE FLUXES
        // =====================================================================
        Flux f_L, f_R, f_star_L, f_star_R, f_star_I_L, f_star_I_R;

        if (Spec_Vol == 1)
        {
            // ---- Left physical flux ----
            f_L.mass = rho_L * u_L;
            f_L.momentum_normal = rho_L * u_L * u_L + p_L;
            f_L.momentum_tangent = rho_L * u_L * v_L;
            f_L.energy = u_L * (E_L + p_L);

            // ---- Left star flux ----
            f_star_L.mass = f_L.mass + S_L * (rho_star_L - rho_L);
            f_star_L.momentum_normal = f_L.momentum_normal + S_L * (rho_star_L * S_star_L - rho_L * u_L);
            f_star_L.momentum_tangent = f_L.momentum_tangent + S_L * (rho_star_L * v_L - rho_L * v_L);
            f_star_L.energy = f_L.energy + S_L * (E_star_L - E_L);

            // ---- Left interface flux ----
            f_star_I_L.mass = f_star_L.mass + S_I * (rho_star_I_L - rho_star_L);
            f_star_I_L.momentum_normal = f_star_L.momentum_normal + S_I * (rho_star_I_L * S_I - rho_star_L * S_star_L);
            f_star_I_L.momentum_tangent = f_star_L.momentum_tangent + S_I * (rho_star_I_L * v_L - rho_star_L * v_L);
            f_star_I_L.energy = f_star_L.energy + S_I * (E_star_I_L - E_star_L);

            // ---- Right interface flux ----
            f_star_I_R.mass = rho_star_I_R * S_I;
            f_star_I_R.momentum_normal = rho_star_I_R * S_I * S_I + p_I;
            f_star_I_R.momentum_tangent = rho_star_I_R * S_I * v_R;
            f_star_I_R.energy = S_I * (E_star_I_R + p_I);

            // ---- Right star flux ----
            f_star_R.mass = rho_star_R * S_star_R;
            f_star_R.momentum_normal = rho_star_R * S_star_R * S_star_R + p_star_R;
            f_star_R.momentum_tangent = rho_star_R * S_star_R * v_R;
            f_star_R.energy = S_star_R * (E_star_R + p_star_R);

            // ---- Right physical flux ----
            f_R.mass = rho_R * u_R;
            f_R.momentum_normal = rho_R * u_R * u_R + p_R;
            f_R.momentum_tangent = rho_R * u_R * v_R;
            f_R.energy = u_R * (E_R + p_R);
        }
        else
        {
            // ---- Energies per unit mass [J/kg] ----
            // Left physical
            f_L.mass = rho_L * u_L;
            f_L.momentum_normal = rho_L * u_L * u_L + p_L;
            f_L.momentum_tangent = rho_L * u_L * v_L;
            f_L.energy = rho_L * u_L * (E_L + p_L / (rho_L + small));

            // Left star
            f_star_L.mass = f_L.mass + S_L * (rho_star_L - rho_L);
            f_star_L.momentum_normal = f_L.momentum_normal + S_L * (rho_star_L * S_star_L - rho_L * u_L);
            f_star_L.momentum_tangent = f_L.momentum_tangent + S_L * (rho_star_L * v_L - rho_L * v_L);
            f_star_L.energy = f_L.energy + S_L * rho_star_L * (E_star_L - E_L);

            // Left interface
            f_star_I_L.mass = f_star_L.mass + S_I * (rho_star_I_L - rho_star_L);
            f_star_I_L.momentum_normal = f_star_L.momentum_normal + S_I * (rho_star_I_L * S_I - rho_star_L * S_star_L);
            f_star_I_L.momentum_tangent = f_star_L.momentum_tangent + S_I * (rho_star_I_L * v_L - rho_star_L * v_L);
            f_star_I_L.energy = f_star_L.energy + S_I * rho_star_I_L * (E_star_I_L - E_star_L);

            // Right interface
            f_star_I_R.mass = rho_star_I_R * S_I;
            f_star_I_R.momentum_normal = rho_star_I_R * S_I * S_I + p_I;
            f_star_I_R.momentum_tangent = rho_star_I_R * S_I * v_R;
            f_star_I_R.energy = rho_star_I_R * S_I * (E_star_I_R + p_I / (rho_star_I_R + small));

            // Right star
            f_star_R.mass = rho_star_R * S_star_R;
            f_star_R.momentum_normal = rho_star_R * S_star_R * S_star_R + p_star_R;
            f_star_R.momentum_tangent = rho_star_R * S_star_R * v_R;
            f_star_R.energy = rho_star_R * S_star_R * (E_star_R + p_star_R / (rho_star_R + small));

            // Right physical
            f_R.mass = rho_R * u_R;
            f_R.momentum_normal = rho_R * u_R * u_R + p_R;
            f_R.momentum_tangent = rho_R * u_R * v_R;
            f_R.energy = rho_R * u_R * (E_R + p_R / (rho_R + small));
        }

        // =====================================================================
        // 6. SELECT FLUX BASED ON WAVE STRUCTURE
        // =====================================================================
        Flux fl;
        if (0.0 <= S_L)
        {
            fl = f_L;
            fl.u_interface = u_L;
        }
        else if (S_L <= 0.0 && 0.0 <= S_star_L)
        {
            fl = f_star_L;
            fl.u_interface = S_star_L;
        }
        else if (S_star_L <= 0.0 && 0.0 <= S_I)
        {
            fl = f_star_I_L;
            fl.u_interface = S_I;
        }
        else if (S_I <= 0.0 && 0.0 <= S_star_R)
        {
            fl = f_star_I_R;
            fl.u_interface = S_I;
        }
        else if (S_star_R <= 0.0 && 0.0 <= S_R)
        {
            fl = f_star_R;
            fl.u_interface = S_star_R;
        }
        else
        {
            fl = f_R;
            fl.u_interface = u_R;
        }

        // =====================================================================
        // 7. ERROR CHECKING
        // =====================================================================
        if ((fl.mass != fl.mass) || (fl.momentum_normal != fl.momentum_normal) || (fl.momentum_tangent != fl.momentum_tangent) || (fl.energy != fl.energy) || (a_L != a_L) || (a_R != a_R))
        {
            bool verbose = false;
            if (verbose)
            {
                Util::ParallelMessage(INFO, "-------------------------------");
                Util::ParallelMessage(INFO, "ERROR IN FURFARO TWO-PHASE HLLC:");
                Util::ParallelMessage(INFO, "hi:   ", hi);
                Util::ParallelMessage(INFO, "lo:   ", lo);
                Util::ParallelMessage(INFO, "flux: ", fl);
                Util::ParallelMessage(INFO, "S_L=", S_L, " S_star_L=", S_star_L, " S_I=", S_I);
                Util::ParallelMessage(INFO, "S_star_R=", S_star_R, " S_R=", S_R);
                Util::ParallelMessage(INFO, "p_L=", p_L, " p_star_L=", p_star_L, " p_I=", p_I);
                Util::ParallelMessage(INFO, "p_star_R=", p_star_R, " p_R=", p_R);
            }
            Util::Exception(INFO, "NaN detected in Furfaro two-phase HLLC solver");
        }

        return fl;
    }
};

} // namespace FluidRiemann
} // namespace Local
} // namespace Solver

#endif
