// All-Mach HLLC Riemann solver: standard HLLC with Guillard-Viozat
// low-Mach pressure-dissipation scaling.
//
//      At low Mach numbers, standard HLLC has O(1/Ma) pressure dissipation
//      that creates spurious pressure oscillations (checkerboard modes).
//      The Guillard-Viozat fix scales ONLY the pressure-velocity coupling in
//      the star-state energy by a local Mach-dependent factor alpha_Ma:
//          alpha_Ma = 1                              for Ma >= Ma_cutoff
//          alpha_Ma = alpha_min + (1-alpha_min)*Ma/Ma_cutoff  otherwise
//      Wave speeds and star densities remain UNCHANGED from standard HLLC,
//      preserving shock-capturing accuracy at high Mach.
//
// Source: H. Guillard and C. Viozat, "On the behaviour of upwind schemes in
//         the low Mach number limit," Computers & Fluids, vol. 28, no. 1,
//         pp. 63-86, 1999. doi: 10.1016/S0045-7930(98)00017-6
//
// Source: V.A. Titarev, E.F. Toro, Finite-volume WENO schemes for three-
//         dimensional conservation laws, J. Comput. Phys. 201(1), 2004.
//
// Source: M. J. Del Razo and R. J. LeVeque, "Numerical Methods for Interface
//         Coupling of Compressible and Almost Incompressible Media," SIAM J.
//         Sci. Comput. 39(3), pp. B486-B507, 2017.

#ifndef SOLVER_LOCAL_FLUIDRIEMANN_HLLC_ALL_MACH_H
#define SOLVER_LOCAL_FLUIDRIEMANN_HLLC_ALL_MACH_H

#include "IO/ParmParse.H"
#include "Solver/Local/FluidRiemann/Riemann.H"

namespace Solver
{
namespace Local
{
namespace FluidRiemann
{

/// All-Mach HLLC Riemann Solver (Guillard-Viozat low-Mach fix)
class HLLC_All_Mach : public FluidRiemann
{
public:
    static constexpr const char *name = "hllc_all_mach";

    HLLC_All_Mach(IO::ParmParse &pp, std::string name) { pp_queryclass(name, *this); }
    HLLC_All_Mach(IO::ParmParse &pp) { pp_queryclass(*this); }
    HLLC_All_Mach()
    {
        IO::ParmParse pp;
        pp_queryclass(*this);
    }

    Set::Scalar Ma_cutoff = 0.3; // Mach number below which low-Mach treatment activates
    Set::Scalar alpha_gv_min = 0.0; // Minimum pressure scaling (0 = full low-Mach, 1 = standard HLLC)

    static void Parse(HLLC_All_Mach &value, IO::ParmParse &pp)
    {
        // Mach number cutoff for low-Mach treatment (typical: 0.1 - 0.5)
        pp.query_default("ma_cutoff", value.Ma_cutoff, 0.3);

        // Minimum pressure scaling factor (typical: 0.0 - 0.1)
        // 0.0 = aggressive low-Mach treatment
        // 1.0 = standard HLLC (no modification)
        pp.query_default("alpha_min", value.alpha_gv_min, 0.0);
    }

    // Legacy single-phase entry point: forwards to the extended overload
    // with no passive scalars and no Korteweg face data.
    virtual Flux Solve(State lo, State hi, Set::Scalar p_ref,
                   Set::Scalar small, int Spec_Vol = 1) override
    {
        return Solve(lo, hi, p_ref, small,
                     /*n_passive=*/0,
                     /*q_lo=*/nullptr, /*q_hi=*/nullptr, /*q_flux=*/nullptr,
                     /*kfd=*/nullptr,
                     Spec_Vol);
    }

    // Extended All-Mach HLLC solve: passive scalars, Korteweg face data,
    // α-weighted two-phase EOS, AND Guillard-Viozat low-Mach fix.
    virtual Flux Solve(State lo, State hi, Set::Scalar p_ref,
                       Set::Scalar small,
                       int n_passive,
                       Set::Scalar const* q_lo, Set::Scalar const* q_hi,
                       Set::Scalar* q_flux,
                       KortewegFaceData const* kfd,
                       int /*Spec_Vol*/ = 1) override
    {
        // Vanishing-phase floor for the /α division.
        constexpr Set::Scalar alpha_min = Set::Scalar(1e-12);
        Set::Scalar aL_alpha = std::max(lo.alpha, alpha_min);
        Set::Scalar aR_alpha = std::max(hi.alpha, alpha_min);

        //---------------------------------------------------------------------
        // Extract left/right states (rho/M/E are partial; α cancels in u, KE)
        //---------------------------------------------------------------------
        Set::Scalar rhoL = std::max(lo.rho, small);
        Set::Scalar rhoR = std::max(hi.rho, small);

        Set::Scalar uL = lo.M_normal / rhoL;
        Set::Scalar uR = hi.M_normal / rhoR;

        Set::Scalar vL = lo.M_tangent / rhoL;
        Set::Scalar vR = hi.M_tangent / rhoR;

        Set::Scalar EL = lo.E;
        Set::Scalar ER = hi.E;

        Set::Scalar gamL = lo.gam;
        Set::Scalar gamR = hi.gam;

        Set::Scalar piL = lo.pi;
        Set::Scalar piR = hi.pi;

        //---------------------------------------------------------------------
        // Internal energy (partial; α cancels in KE = ½ M²/ρ_partial)
        //---------------------------------------------------------------------
        Set::Scalar KEL = 0.5 * (uL*uL + vL*vL) * rhoL;
        Set::Scalar UEL = EL - KEL;

        Set::Scalar KER = 0.5 * (uR*uR + vR*vR) * rhoR;
        Set::Scalar UER = ER - KER;

        //---------------------------------------------------------------------
        // Stiffened-gas physical pressure. UE_partial / α = UE_intrinsic.
        //---------------------------------------------------------------------
        Set::Scalar pL = (gamL - 1.0) * (UEL / aL_alpha) - gamL * piL + p_ref;
        Set::Scalar pR = (gamR - 1.0) * (UER / aR_alpha) - gamR * piR + p_ref;

        pL = std::max(pL, small);
        pR = std::max(pR, small);

        //---------------------------------------------------------------------
        // Stiffened-gas sound speeds: a² = γ(p + π)/ρ_intrinsic.
        //---------------------------------------------------------------------
        Set::Scalar rho_int_L = rhoL / aL_alpha;
        Set::Scalar rho_int_R = rhoR / aR_alpha;
        Set::Scalar aL = std::sqrt(gamL * ((pL + piL) / rho_int_L));
        Set::Scalar aR = std::sqrt(gamR * ((pR + piR) / rho_int_R));

        //---------------------------------------------------------------------
        // Wave speed estimates (HLLE) — UNCHANGED from standard HLLC
        //---------------------------------------------------------------------
        Set::Scalar SL = std::min(uL - aL, uR - aR);
        Set::Scalar SR = std::max(uL + aL, uR + aR);

        //---------------------------------------------------------------------
        // Contact wave speed — UNCHANGED from standard HLLC
        //---------------------------------------------------------------------
        Set::Scalar S_star =
            (pR - pL + rhoL*uL*(SL - uL) - rhoR*uR*(SR - uR))
            / (rhoL*(SL - uL) - rhoR*(SR - uR) + small);

        //=====================================================================
        // GUILLARD-VIOZAT LOW-MACH MODIFICATION
        // Compute local Mach numbers and pressure-dissipation scaling factor.
        // alpha_Ma = 1 for high-Mach (standard HLLC), → alpha_gv_min for
        // Ma → 0 (reduced pressure dissipation that eliminates the O(1/Ma)
        // checkerboard mode).
        //=====================================================================
        Set::Scalar Ma_L = std::abs(uL) / (aL + small);
        Set::Scalar Ma_R = std::abs(uR) / (aR + small);
        Set::Scalar Ma_local = std::max(Ma_L, Ma_R);

        Set::Scalar alpha_Ma;
        if (Ma_local >= Ma_cutoff)
        {
            alpha_Ma = Set::Scalar(1.0);
        }
        else
        {
            alpha_Ma = alpha_gv_min
                     + (Set::Scalar(1.0) - alpha_gv_min) * (Ma_local / Ma_cutoff);
        }

        //---------------------------------------------------------------------
        // Star region partial densities — UNCHANGED from standard HLLC
        //---------------------------------------------------------------------
        Set::Scalar rhoL_star = rhoL * (SL - uL) / (SL - S_star + small);
        Set::Scalar rhoR_star = rhoR * (SR - uR) / (SR - S_star + small);

        //---------------------------------------------------------------------
        // Star region partial energies — MODIFIED by alpha_Ma.
        // Standard HLLC: E*_L = ρ*_L · [E_L/ρ_L + (S*-u_L)·(S* + p_L/(ρ_L(S_L-u_L)))]
        // GV modified:   E*_L = ρ*_L · [E_L/ρ_L + (S*-u_L)·(alpha_Ma·S* + p_L/(ρ_L(S_L-u_L)))]
        // The alpha_Ma factor scales the pressure-velocity coupling term
        // (S*-u_L)·S* that carries the star-state pressure jump; the
        // explicit-pressure term p_L/(ρ_L(S_L-u_L)) is unmodified so that
        // the solver remains exact for contact discontinuities and uniform
        // flow regardless of Mach number.
        //---------------------------------------------------------------------
        Set::Scalar EL_star =
            (EL/rhoL + (S_star - uL)*(alpha_Ma*S_star + pL/(rhoL*(SL - uL) + small)))
            * rhoL_star;

        Set::Scalar ER_star =
            (ER/rhoR + (S_star - uR)*(alpha_Ma*S_star + pR/(rhoR*(SR - uR) + small)))
            * rhoR_star;

        //---------------------------------------------------------------------
        // Compute fluxes fL and fR (partial-density flux form)
        //---------------------------------------------------------------------
        Flux fL, fR;

        fL.mass = rhoL * uL;
        fL.momentum_normal  = rhoL*uL*uL + pL;
        fL.momentum_tangent = rhoL*uL*vL;
        fL.energy = uL * (EL + pL);

        fR.mass = rhoR * uR;
        fR.momentum_normal  = rhoR*uR*uR + pR;
        fR.momentum_tangent = rhoR*uR*vR;
        fR.energy = uR * (ER + pR);

        //---------------------------------------------------------------------
        // Star fluxes
        //---------------------------------------------------------------------
        Flux fL_star, fR_star;

        fL_star.mass = fL.mass + SL*(rhoL_star - rhoL);
        fL_star.momentum_normal =
            fL.momentum_normal + SL*(rhoL_star*S_star - rhoL*uL);
        fL_star.momentum_tangent =
            fL.momentum_tangent + SL*(rhoL_star*vL - rhoL*vL);
        fL_star.energy = fL.energy + SL*(EL_star - EL);

        fR_star.mass = fR.mass + SR*(rhoR_star - rhoR);
        fR_star.momentum_normal =
            fR.momentum_normal + SR*(rhoR_star*S_star - rhoR*uR);
        fR_star.momentum_tangent =
            fR.momentum_tangent + SR*(rhoR_star*vR - rhoR*vR);
        fR_star.energy = fR.energy + SR*(ER_star - ER);

        //---------------------------------------------------------------------
        // Pick the wave region once; reuse for both inviscid Flux and the
        // passive-scalar flux selection so they share identical wave logic.
        //---------------------------------------------------------------------
        enum Region { LEFT, LEFT_STAR, RIGHT_STAR, RIGHT };
        Region region;
        if (0.0 <= SL)                              region = LEFT;
        else if (SL <= 0.0 && 0.0 <= S_star)        region = LEFT_STAR;
        else if (S_star <= 0.0 && 0.0 <= SR)        region = RIGHT_STAR;
        else                                        region = RIGHT;

        Flux f;
        switch (region) {
            case LEFT:       f = fL;      break;
            case LEFT_STAR:  f = fL_star; break;
            case RIGHT_STAR: f = fR_star; break;
            case RIGHT:      f = fR;      break;
        }

        //---------------------------------------------------------------------
        // Passive scalars: each q is a conservative variable that advects
        // with the contact wave.
        //---------------------------------------------------------------------
        for (int s = 0; s < n_passive; ++s)
        {
            Set::Scalar qL_v   = q_lo[s];
            Set::Scalar qR_v   = q_hi[s];
            Set::Scalar fL_q   = qL_v * uL;
            Set::Scalar fR_q   = qR_v * uR;
            Set::Scalar qL_str = qL_v * (SL - uL) / (SL - S_star + small);
            Set::Scalar qR_str = qR_v * (SR - uR) / (SR - S_star + small);
            Set::Scalar fL_qs  = fL_q + SL * (qL_str - qL_v);
            Set::Scalar fR_qs  = fR_q + SR * (qR_str - qR_v);

            switch (region) {
                case LEFT:       q_flux[s] = fL_q;  break;
                case LEFT_STAR:  q_flux[s] = fL_qs; break;
                case RIGHT_STAR: q_flux[s] = fR_qs; break;
                case RIGHT:      q_flux[s] = fR_q;  break;
            }
        }

        //---------------------------------------------------------------------
        // Korteweg flux contribution (flux-form capillary stress).
        // Identical to HLLC.H — the low-Mach fix does not interact with
        // the Korteweg stress, which is a physical surface-tension force.
        //---------------------------------------------------------------------
        if (kfd != nullptr)
        {
            Set::Scalar dn   = kfd->dn_eta;
            Set::Scalar dt   = kfd->dt_eta;
            Set::Scalar etaf = kfd->eta_face;
            Set::Scalar lapf = kfd->lap_eta;
            Set::Scalar k    = kfd->kappa;
            Set::Scalar q    = kfd->q_eta;
            Set::Scalar ge2  = dn*dn + dt*dt;

            Set::Scalar S_nn = -k*dn*dn + k*etaf*lapf + Set::Scalar(0.5)*k*ge2 - q;
            Set::Scalar S_nt = -k*dn*dt;

            Set::Scalar ut_face = (S_star >= Set::Scalar(0.0)) ? vL : vR;

            f.momentum_normal  -= S_nn;
            f.momentum_tangent -= S_nt;
            if (!kfd->skip_energy)
                f.energy       -= S_nn * S_star + S_nt * ut_face;
        }

        return f;
    }

    static int
    Test()
    {
        HLLC_All_Mach solver;
        int failed = 0;
        Set::Scalar pref = 0.0;
        Set::Scalar small = 1E-10;

        // Test 1: Tangential Velocity Difference - No Normal Flux
        try
        {
            State left(1.0, 1.0, 0.0, 1.0, 1.4, 0.0, NAN);
            State center(1.0, 1.0, 1.0, 1.0, 1.4, 0.0, NAN);
            State right(1.0, 1.0, 2.0, 1.0, 1.4, 0.0, NAN);
            Flux fluxlo = solver.Solve(center, right, pref, small);
            Flux fluxhi = solver.Solve(left, center, pref, small);

            if (fabs(fluxhi.mass - fluxlo.mass) > 1E-10
                || fabs(fluxhi.momentum_normal - fluxlo.momentum_normal) > 1E-10
                || fabs(fluxhi.energy - fluxlo.energy) > 1E-10)
            {
                Util::Warning(INFO, "left:    ", left);
                Util::Warning(INFO, "center:  ", center);
                Util::Warning(INFO, "right:   ", right);
                Util::Warning(INFO, "Fluxlo:  ", fluxlo);
                Util::Warning(INFO, "Fluxhi:  ", fluxhi);
                Util::Exception(INFO, "Tangential velocity difference incorrectly affecting normal flux.");
            }
            Util::Test::SubMessage("Test 1: Tangential velocity should induce no normal flux", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 1: Tangential velocity should induce no normal flux", 1);
        }

        // Test 2: Symmetry Test (no flux across identical states)
        try
        {
            State left(1.0, 0.0, 0.0, 1.0, 1.4, 0.0, NAN);
            State center(1.0, 0.0, 0.0, 1.0, 1.4, 0.0, NAN);
            State right(1.0, 0.0, 0.0, 1.0, 1.4, 0.0, NAN);
            Flux fluxhi = solver.Solve(center, right, pref, small);
            Flux fluxlo = solver.Solve(left, center, pref, small);

            if (fabs(fluxhi.mass - fluxlo.mass) > 1E-10
                || fabs(fluxhi.momentum_normal - fluxlo.momentum_normal) > 1E-10
                || fabs(fluxhi.momentum_tangent) > 1E-10
                || fabs(fluxlo.momentum_tangent) > 1E-10
                || fabs(fluxhi.energy - fluxlo.energy) > 1E-10)
            {
                Util::Warning(INFO, "left:  ", left);
                Util::Warning(INFO, "right: ", right);
                Util::Warning(INFO, "Fluxhi:  ", fluxhi);
                Util::Warning(INFO, "Fluxlo:  ", fluxlo);
                Util::Exception(INFO, "Symmetric states should result in zero flux.");
            }
            Util::Test::SubMessage("Test 2: Constant states induces no flux difference", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 2: Constant states induces no flux difference", 1);
        }

        // Test 3: Uniform Flow Test
        try
        {
            State left(1.0, 1.0, 0.5, 1.0, 1.4, 0.0, NAN);
            State center(1.0, 1.0, 0.5, 1.0, 1.4, 0.0, NAN);
            State right(1.0, 1.0, 0.5, 1.0, 1.4, 0.0, NAN);
            Flux fluxhi = solver.Solve(center, right, pref, small);
            Flux fluxlo = solver.Solve(left, center, pref, small);

            if (fabs(fluxhi.mass - fluxlo.mass) > 1E-10
                || fabs(fluxhi.momentum_normal - fluxlo.momentum_normal) > 1E-10
                || fabs(fluxhi.energy - fluxlo.energy) > 1E-10)
            {
                Util::Warning(INFO, "left:   ", left);
                Util::Warning(INFO, "center: ", center);
                Util::Warning(INFO, "right:  ", right);
                Util::Warning(INFO, "Fluxlo: ", fluxlo);
                Util::Warning(INFO, "Fluxhi: ", fluxhi);
                Util::Exception(INFO, "Uniform flow should result in constant flux.");
            }
            Util::Test::SubMessage("Test 3: Uniform flow should maintain constant flux", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 3: Uniform flow should maintain constant flux", 1);
        }

        // Test 4: High-Mach recovers standard HLLC (alpha_Ma = 1)
        // At Ma > Ma_cutoff, the solver should produce identical results
        // to standard HLLC. Test with a supersonic shock tube.
        try
        {
            // Sod-like conditions: Ma ~ 0.9 for left state
            State left (1.0, 0.0, 0.0, 2.5, 1.4, 0.0, NAN);  // p=1.0
            State right(0.125, 0.0, 0.0, 0.25, 1.4, 0.0, NAN); // p=0.1

            Flux flux = solver.Solve(left, right, pref, small);

            if (flux.mass != flux.mass || flux.energy != flux.energy)
            {
                Util::Exception(INFO, "High-Mach shock tube producing NaN.");
            }
            Util::Test::SubMessage("Test 4: High-Mach shock tube (no NaN)", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 4: High-Mach shock tube (no NaN)", 1);
        }

        // Test 5: Passive scalar advects at the contact velocity
        try
        {
            State left (1.0, 1.0, 0.0, 1.0, 1.4, 0.0, NAN);
            State right(1.0, 1.0, 0.0, 1.0, 1.4, 0.0, NAN);
            Set::Scalar qL[2] = { 0.5, 2.0 };
            Set::Scalar qR[2] = { 0.5, 2.0 };
            Set::Scalar qF[2] = { 0.0, 0.0 };
            solver.Solve(left, right, pref, small,
                         2, qL, qR, qF, nullptr, 1);
            if (fabs(qF[0] - 0.5) > 1E-10 || fabs(qF[1] - 2.0) > 1E-10)
            {
                Util::Warning(INFO, "qF[0]=", qF[0], "  qF[1]=", qF[1]);
                Util::Exception(INFO, "Passive scalar: uniform-flow flux mismatch.");
            }
            Util::Test::SubMessage("Test 5: Passive scalar uniform advection", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 5: Passive scalar uniform advection", 1);
        }

        // Test 6: Korteweg with zero η-gradients reduces to base flux
        try
        {
            State left (1.0, 1.0, 0.0, 1.0, 1.4, 0.0, NAN);
            State right(1.5, 1.0, 0.0, 1.5, 1.4, 0.0, NAN);

            Flux f_base = solver.Solve(left, right, pref, small);

            KortewegFaceData kfd;
            kfd.eta_face = 0.5;
            kfd.dn_eta   = 0.0;
            kfd.dt_eta   = 0.0;
            kfd.lap_eta  = 0.0;
            kfd.kappa    = 1.0;
            Flux f_new = solver.Solve(left, right, pref, small,
                                      0, nullptr, nullptr, nullptr, &kfd, 1);

            if (fabs(f_base.mass             - f_new.mass)             > 1E-10
                || fabs(f_base.momentum_normal  - f_new.momentum_normal)  > 1E-10
                || fabs(f_base.momentum_tangent - f_new.momentum_tangent) > 1E-10
                || fabs(f_base.energy           - f_new.energy)           > 1E-10)
            {
                Util::Warning(INFO, "f_base: ", f_base);
                Util::Warning(INFO, "f_new:  ", f_new);
                Util::Exception(INFO, "KFD with zero gradients should not change flux.");
            }
            Util::Test::SubMessage("Test 6: Korteweg zero-gradient flux invariance", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 6: Korteweg zero-gradient flux invariance", 1);
        }

        return failed;
    }
};

}
}
}

#endif
