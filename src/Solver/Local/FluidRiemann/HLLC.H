// This implements a modified Riemann HLLC solver for weakly incompressible flow using the Tammann EOS.
//      HLLC solver uses the fastest left and right wave speeds, along with the middle contact wave.
//      Restores some accuracy when compared to HLLE with a slightly increased computational time, while providing same robustness.
//
// Source: V.A. Titarev, E.F. Toro, Finite-volume WENO schemes for three-dimensional conservation laws, Journal of Computational Physics, Volume 201, Issue 1, 2004, Pages 238-260, ISSN 0021-9991, https://doi.org/10.1016/j.jcp.2004.05.015.
//          URL: https://www.sciencedirect.com/science/article/pii/S0021999104002281
//
// Source: M. J. Del Razo and R. J. LeVeque, "Numerical Methods for Interface Coupling of Compressible and Almost Incompressible Media," SIAM Journal on Scientific Computing, vol. 39, no. 3, pp. B486�B507, Jan. 2017, doi: 10.1137/16m1067834.
//          URL: https://arxiv.org/abs/1604.08544
//          If you wish to use the regular gas HLLC sovler and not the stiffened gas modeled by the Tammann EOS, set pi = 0

#ifndef SOLVER_LOCAL_FLUIDRIEMANN_HLLC_H
#define SOLVER_LOCAL_FLUIDRIEMANN_HLLC_H

#include "IO/ParmParse.H"
#include "Solver/Local/FluidRiemann/Riemann.H"
// #include "Util/Util.H"

namespace Solver
{
namespace Local
{
namespace FluidRiemann
{

/// HLLC Riemann Solver
class HLLC : public FluidRiemann
{
public:
    static constexpr const char *name = "hllc";

    HLLC(IO::ParmParse &pp, std::string name) { pp_queryclass(name, *this); }
    HLLC(IO::ParmParse &pp) { pp_queryclass(*this); }
    HLLC()
    {
        IO::ParmParse pp;
        pp_queryclass(*this);
    }

    static void Parse(HLLC & /*value*/, IO::ParmParse & /*pp*/)
    {
        // No parameters to parse
    }

    // Legacy single-phase entry point: forwards to the extended overload
    // with no passive scalars and no Korteweg face data. State.alpha
    // defaults to 1.0, so the EOS calls below collapse to the original
    // single-phase formulas — legacy callers (Hydro.cpp solid-fluid
    // path, the unit tests below) need no source change.
    virtual Flux Solve(State lo, State hi, Set::Scalar p_ref,
                   Set::Scalar small, int Spec_Vol = 1) override
    {
        return Solve(lo, hi, p_ref, small,
                     /*n_passive=*/0,
                     /*q_lo=*/nullptr, /*q_hi=*/nullptr, /*q_flux=*/nullptr,
                     /*kfd=*/nullptr,
                     Spec_Vol);
    }

    // Extended HLLC solve: passive scalars (advected through the contact
    // wave) and an optional flux-form Korteweg capillary stress.
    //
    // State.rho / .M_normal / .M_tangent / .E are interpreted as PARTIAL
    // quantities (α_k · intrinsic) when State.alpha < 1; the EOS recovers
    // intrinsic ρ via /alpha. Velocity (M/ρ) and KE (M²/2ρ) are α-invariant
    // so those formulas are unchanged. Only the EOS pressure / sound-speed
    // line carries an explicit /α (collapses to identity for legacy α=1).
    //
    // The two phases' Korteweg contributions are intended to be summed at
    // the call site by passing kfd.kappa = Y_k_face · κ_global per phase.
    virtual Flux Solve(State lo, State hi, Set::Scalar p_ref,
                       Set::Scalar small,
                       int n_passive,
                       Set::Scalar const* q_lo, Set::Scalar const* q_hi,
                       Set::Scalar* q_flux,
                       KortewegFaceData const* kfd,
                       int /*Spec_Vol*/ = 1) override
    {
        // Vanishing-phase floor for the /α division. Caller is responsible
        // for setting State.alpha >= alpha_min when the phase is present;
        // this floor only guards against a phase that has truly disappeared.
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
        // Legacy callers (α = 1) get the original (γ−1) UE − γπ + p_ref.
        //---------------------------------------------------------------------
        Set::Scalar pL = (gamL - 1.0) * (UEL / aL_alpha) - gamL * piL + p_ref;
        Set::Scalar pR = (gamR - 1.0) * (UER / aR_alpha) - gamR * piR + p_ref;

        pL = std::max(pL, small);
        pR = std::max(pR, small);

        //---------------------------------------------------------------------
        // Stiffened-gas sound speeds: a² = γ(p + π)/ρ_intrinsic.
        // ρ_intrinsic = ρ_partial / α; legacy callers (α = 1) collapse to
        // a² = γ(p + π)/ρ as before.
        //---------------------------------------------------------------------
        Set::Scalar rho_int_L = rhoL / aL_alpha;
        Set::Scalar rho_int_R = rhoR / aR_alpha;
        Set::Scalar aL = std::sqrt(gamL * ((pL + piL) / rho_int_L));
        Set::Scalar aR = std::sqrt(gamR * ((pR + piR) / rho_int_R));

        //---------------------------------------------------------------------
        // Wave speed estimates (HLLE)
        //---------------------------------------------------------------------
        Set::Scalar SL = std::min(uL - aL, uR - aR);
        Set::Scalar SR = std::max(uL + aL, uR + aR);

        //---------------------------------------------------------------------
        // Contact wave speed. ρ here is the conserved-mass-density appearing
        // in the mass-flux f.mass = ρ u — for the two-phase formulation that
        // is the partial density (α ρ_int), so the existing rhoL/rhoR usage
        // stays correct without an α factor.
        //---------------------------------------------------------------------
        Set::Scalar S_star =
            (pR - pL + rhoL*uL*(SL - uL) - rhoR*uR*(SR - uR))
            / (rhoL*(SL - uL) - rhoR*(SR - uR) + small);

        //---------------------------------------------------------------------
        // Star region partial densities
        //---------------------------------------------------------------------
        Set::Scalar rhoL_star = rhoL * (SL - uL) / (SL - S_star + small);
        Set::Scalar rhoR_star = rhoR * (SR - uR) / (SR - S_star + small);

        //---------------------------------------------------------------------
        // Star region partial energies
        //---------------------------------------------------------------------
        Set::Scalar EL_star =
            (EL/rhoL + (S_star - uL)*(S_star + pL/(rhoL*(SL - uL) + small)))
            * rhoL_star;

        Set::Scalar ER_star =
            (ER/rhoR + (S_star - uR)*(S_star + pR/(rhoR*(SR - uR) + small)))
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
        // Passive scalars: each q is a conservative variable (e.g. α_k ρ_k η)
        // that advects with the contact wave.
        //   q_X_star = q_X · (S_X − u_X) / (S_X − S_star)
        //   f_q_X*   = q_X u_X + S_X (q_X_star − q_X)
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
        //
        //   Σ_K = -κ ∇η⊗∇η + (κ η ∇²η + κ/2 |∇η|²) I
        //
        // Momentum flux: F_M = ρu⊗u + p I − Σ_K  ⇒  subtract Σ_K · n̂
        // Energy flux:   F_E = u(E+p) − Σ_K · u   ⇒  subtract Σ_K · u_face
        //
        // For the energy work term, u_n at the face = S_star (HLLC contact),
        // u_t at the face is the upwind-of-S_star tangential. This is the
        // textbook HLLC convention (Toro §10).
        //
        // kfd.kappa is expected to be the per-phase-weighted κ (caller
        // sets kappa = Y_k_face · κ_global) so the two phases' Korteweg
        // flux contributions sum to the full mixture Σ_K flux.
        //---------------------------------------------------------------------
        if (kfd != nullptr)
        {
            Set::Scalar dn   = kfd->dn_eta;
            Set::Scalar dt   = kfd->dt_eta;
            Set::Scalar etaf = kfd->eta_face;
            Set::Scalar lapf = kfd->lap_eta;
            Set::Scalar k    = kfd->kappa;
            Set::Scalar q    = kfd->q_eta;       // bulk-thermo trace correction
            Set::Scalar ge2  = dn*dn + dt*dt;

            // Σ_K_wb_nn = (gradient-energy stress)_nn − q_eta
            // Σ_K_wb_nt = (gradient-energy stress)_nt          (no trace term)
            // q_eta is on the diagonal only — does not enter Σ_nt.
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
        HLLC solver;
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

        // Test 2: Pure Transverse Velocity Difference
        try
        {
            State left(1.0, 0.0, 0.0, 1.0, 1.4, 0.0, NAN);
            State center(1.0, 0.0, 1.0, 1.0, 1.4, 0.0, NAN);
            State right(1.0, 0.0, 2.0, 1.0, 1.4, 0.0, NAN);
            Flux fluxlo = solver.Solve(left, center, pref, small);
            Flux fluxhi = solver.Solve(center, right, pref, small);

            if (fabs(fluxhi.mass - fluxlo.mass) > 1E-10
                || fabs(fluxhi.momentum_normal - fluxlo.momentum_normal) > 1E-10
                || fabs(fluxhi.energy - fluxlo.energy) > 1E-10)
            {
                Util::Warning(INFO, "left:  ", left);
                Util::Warning(INFO, "center: ", center);
                Util::Warning(INFO, "right: ", right);
                Util::Warning(INFO, "Fluxhi:  ", fluxhi);
                Util::Warning(INFO, "Fluxlo:  ", fluxlo);
                Util::Exception(INFO, "Pure transverse velocity difference affecting normal flux.");
            }
            Util::Test::SubMessage("Test 2: Pure transverse velocity difference", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 2: Pure transverse velocity difference", 1);
        }

        // Test 3: Symmetry Test (no flux across identical states)
        try
        {
            State left(1.0, 0.0, 0.0, 1.0, 1.4, 0.0, NAN);
            State center(1.0, 0.0, 0.0, 1.0, 1.4, 0.0, NAN);
            State right(1.0, 0.0, 0.0, 1.0, 1.4, 0.0, NAN);
            Flux fluxhi = solver.Solve(center, right, pref, small);
            Flux fluxlo = solver.Solve(left, center, pref, small);

            if (fabs(fluxhi.mass - fluxlo.mass) > 1E-10                          // no change in mass flux
                || fabs(fluxhi.momentum_normal - fluxlo.momentum_normal) > 1E-10 // no change in momentum flux
                || fabs(fluxhi.momentum_tangent) > 1E-10                         // zero tangent flux
                || fabs(fluxlo.momentum_tangent) > 1E-10                         // zero tangent flux
                || fabs(fluxhi.energy - fluxlo.energy) > 1E-10                   // no change in energy flux
            )
            {
                Util::Warning(INFO, "left:  ", left);
                Util::Warning(INFO, "right: ", right);
                Util::Warning(INFO, "Fluxhi:  ", fluxhi);
                Util::Warning(INFO, "Fluxlo:  ", fluxlo);
                Util::Exception(INFO, "Symmetric states should result in zero flux.");
            }
            Util::Test::SubMessage("Test 3: Constant states induces no flux difference", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 3: Constant states induces no flux difference", 1);
        }

        // Test 4: Uniform Flow Test (no flux across uniform flow)
        try
        {
            State left(1.0, 1.0, 0.5, 1.0, 1.4, 0.0, NAN);
            State center(1.0, 1.0, 0.5, 1.0, 1.4, 0.0, NAN);
            State right(1.0, 1.0, 0.5, 1.0, 1.4, 0.0, NAN);
            Flux fluxhi = solver.Solve(center, right, pref, small);
            Flux fluxlo = solver.Solve(left, center, pref, small);

            if (fabs(fluxhi.mass - fluxlo.mass) > 1E-10 || fabs(fluxhi.momentum_normal - fluxlo.momentum_normal) > 1E-10 || fabs(fluxhi.energy - fluxlo.energy) > 1E-10)
            {
                Util::Warning(INFO, "left:   ", left);
                Util::Warning(INFO, "center: ", center);
                Util::Warning(INFO, "right:  ", right);
                Util::Warning(INFO, "Fluxlo: ", fluxlo);
                Util::Warning(INFO, "Fluxhi: ", fluxhi);
                Util::Exception(INFO, "Uniform flow should result in no flux.");
            }
            Util::Test::SubMessage("Test 4: Uniform flow should maintain constant flux", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 4: Uniform flow should maintain constant flux", 1);
        }

        // Test 5: Passive scalar advects at the contact velocity in
        // uniform flow. With L = R, S_star = u = 1.0 and we land in
        // LEFT_STAR, so q_flux = q_L * u_L = q_L.
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

        // Test 6: Korteweg face data with zero η-gradients (and zero ∇²η)
        // should reduce the new path to the legacy single-phase flux.
        try
        {
            State left (1.0, 1.0, 0.0, 1.0, 1.4, 0.0, NAN);
            State right(1.5, 1.0, 0.0, 1.5, 1.4, 0.0, NAN);

            Flux f_legacy = solver.Solve(left, right, pref, small);

            KortewegFaceData kfd;
            kfd.eta_face = 0.5;
            kfd.dn_eta   = 0.0;
            kfd.dt_eta   = 0.0;
            kfd.lap_eta  = 0.0;
            kfd.kappa    = 1.0;
            Flux f_new = solver.Solve(left, right, pref, small,
                                      0, nullptr, nullptr, nullptr, &kfd, 1);

            if (fabs(f_legacy.mass             - f_new.mass)             > 1E-10
                || fabs(f_legacy.momentum_normal  - f_new.momentum_normal)  > 1E-10
                || fabs(f_legacy.momentum_tangent - f_new.momentum_tangent) > 1E-10
                || fabs(f_legacy.energy           - f_new.energy)           > 1E-10)
            {
                Util::Warning(INFO, "f_legacy: ", f_legacy);
                Util::Warning(INFO, "f_new:    ", f_new);
                Util::Exception(INFO, "KFD with zero gradients should not change flux.");
            }
            Util::Test::SubMessage("Test 6: Korteweg zero-gradient flux invariance", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 6: Korteweg zero-gradient flux invariance", 1);
        }

        // Test 7: Korteweg face data with non-trivial η-derivatives shifts
        // the momentum/energy flux by exactly -Σ_K · n̂ (and -Σ_K · u for
        // energy). Uniform L=R state: S_star = u = 1.0, ut_face = vL = 0.
        try
        {
            State left (1.0, 1.0, 0.0, 1.0, 1.4, 0.0, NAN);
            State right(1.0, 1.0, 0.0, 1.0, 1.4, 0.0, NAN);

            Flux f_legacy = solver.Solve(left, right, pref, small);

            KortewegFaceData kfd;
            kfd.eta_face = 0.5;
            kfd.dn_eta   = 1.0;
            kfd.dt_eta   = 0.5;
            kfd.lap_eta  = -0.2;
            kfd.kappa    = 0.1;
            Flux f_new = solver.Solve(left, right, pref, small,
                                      0, nullptr, nullptr, nullptr, &kfd, 1);

            Set::Scalar dn = kfd.dn_eta, dt = kfd.dt_eta;
            Set::Scalar etaf = kfd.eta_face, lapf = kfd.lap_eta, k = kfd.kappa;
            Set::Scalar ge2  = dn*dn + dt*dt;
            Set::Scalar S_nn = -k*dn*dn + k*etaf*lapf + 0.5*k*ge2;
            Set::Scalar S_nt = -k*dn*dt;
            Set::Scalar exp_dM_n = -S_nn;
            Set::Scalar exp_dM_t = -S_nt;
            Set::Scalar exp_dE   = -S_nn * 1.0;  // S_star=1, vL=0

            if (fabs((f_new.momentum_normal  - f_legacy.momentum_normal)  - exp_dM_n) > 1E-10
                || fabs((f_new.momentum_tangent - f_legacy.momentum_tangent) - exp_dM_t) > 1E-10
                || fabs((f_new.energy           - f_legacy.energy)           - exp_dE)   > 1E-10
                || fabs(f_legacy.mass             - f_new.mass)                          > 1E-10)
            {
                Util::Warning(INFO, "f_legacy: ", f_legacy);
                Util::Warning(INFO, "f_new:    ", f_new);
                Util::Warning(INFO, "expected ΔM_n=", exp_dM_n, "  ΔM_t=", exp_dM_t, "  ΔE=", exp_dE);
                Util::Exception(INFO, "Korteweg flux shift does not match -Σ_K.");
            }
            Util::Test::SubMessage("Test 7: Korteweg nonzero ⇒ -Σ_K flux shift", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 7: Korteweg nonzero ⇒ -Σ_K flux shift", 1);
        }

        // Test 8: q_eta (bulk-thermo trace correction) shifts only the
        // momentum_normal flux (and energy via S_star), not momentum_tangent.
        // With pure q_eta (zero ∇η, zero ∇²η) the gradient stress is zero
        // and only the −q_eta term remains, so the sign of the momentum
        // shift is +q_eta (since Σ_nn = −q ⇒ f.M_n -= Σ_nn = +q).
        try
        {
            State left (1.0, 1.0, 0.0, 1.0, 1.4, 0.0, NAN);
            State right(1.0, 1.0, 0.0, 1.0, 1.4, 0.0, NAN);

            Flux f_legacy = solver.Solve(left, right, pref, small);

            KortewegFaceData kfd;
            kfd.eta_face = 0.5;
            kfd.dn_eta   = 0.0;
            kfd.dt_eta   = 0.0;
            kfd.lap_eta  = 0.0;
            kfd.kappa    = 0.0;
            kfd.q_eta    = 0.07;   // arbitrary nonzero
            Flux f_new = solver.Solve(left, right, pref, small,
                                      0, nullptr, nullptr, nullptr, &kfd, 1);

            // S_nn = -q_eta = -0.07 ⇒ ΔM_n = -S_nn = +0.07,
            // ΔM_t = 0, ΔE = -S_nn · S_star = +0.07 · 1.0 = +0.07
            Set::Scalar exp_dM_n = +0.07;
            Set::Scalar exp_dE   = +0.07;
            if (fabs((f_new.momentum_normal  - f_legacy.momentum_normal)  - exp_dM_n) > 1E-10
                || fabs( f_new.momentum_tangent - f_legacy.momentum_tangent) > 1E-10
                || fabs((f_new.energy - f_legacy.energy) - exp_dE) > 1E-10
                || fabs( f_new.mass   - f_legacy.mass)             > 1E-10)
            {
                Util::Warning(INFO, "f_legacy: ", f_legacy);
                Util::Warning(INFO, "f_new:    ", f_new);
                Util::Warning(INFO, "expected ΔM_n=", exp_dM_n, "  ΔE=", exp_dE);
                Util::Exception(INFO, "q_eta-only flux shift does not match +q_eta on M_n / E.");
            }
            Util::Test::SubMessage("Test 8: q_eta trace correction ⇒ +q_eta on M_n/E only", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 8: q_eta trace correction ⇒ +q_eta on M_n/E only", 1);
        }

        return failed;
    }
};

}
}
}

#endif
