#ifndef BC_NSCBC_H
#define BC_NSCBC_H

#include "BC/BC.H"
#include "IO/ParmParse.H"
#include "Set/Set.H"
#include "Solver/EOS/EOS.H"
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

namespace BC
{

/// Navier-Stokes Characteristic Boundary Conditions (NSCBC)
///
/// Implementation of Ghost-Cell NSCBC method from:
/// Motheau et al. "A mixed formulation for direct numerical simulation of
/// turbulent low-Mach number flames" AIAA J. Vol. 55, No. 10, pp. 3399-3408, 2017
///
/// CORRECTED MATH (matching professor's stable implementation):
///   - LODI waves include proper 1/(2a^2) and 1/a^2 eigenvalue normalization
///   - Relaxation coefficient K = sigma*(1-M^2)/(a*L) has correct units [1/(m*s)]
///   - Uses 1st-order FD to eliminate coefficient-3 amplification
///   - Transverse terms include full Lodato (2008) T-tilde corrections at corners
///   - Writes gamma, pi, pressure to ghost cells for Riemann solver
class NSCBC
{
public:
    enum class Type
    {
        None,
        Inflow,
        Outflow,
        SlipWall,
        NoSlipWall
    };

    struct BoundaryParams
    {
        Type type = Type::Outflow;
        Set::Scalar target_u = 0.0;
        Set::Scalar target_v = 0.0;
        Set::Scalar target_w = 0.0;
        Set::Scalar target_T = 300.0;
        Set::Scalar target_p = 101325.0;
        Set::Scalar relax_u = 0.3;
        Set::Scalar relax_v = 0.3;
        Set::Scalar relax_w = 0.3;
        Set::Scalar relax_T = 0.3;
        Set::Scalar beta = 0.6;
        Set::Scalar sigma = 0.25;
        Set::Scalar L_ref = 1.0;
    };

private:
    BoundaryParams params_xlo, params_xhi;
    BoundaryParams params_ylo, params_yhi;
    BoundaryParams params_zlo, params_zhi;
    Set::Scalar small = 1.0e-10;

    static BoundaryParams ParseFace(IO::ParmParse &pp, std::string face_name);

public:
    NSCBC() = default;
    NSCBC(IO::ParmParse &pp, std::string prefix = "nscbc");
    static void Parse(NSCBC &value, IO::ParmParse &pp);

    /// Fill ghost cells for all conservative variables
    ///
    /// CORRECTED FORMULATION:
    ///   - L-waves: L1 = (u-a)/(2*a^2) * (dp/dn - rho*a*du/dn)  [proper scaling]
    ///   - K coefficient: sigma*(1-M^2)/(a*L)  [units: 1/(m*s)]
    ///   - Corners: Full Lodato T-tilde transverse corrections
    ///
    /// @param density_mf    Density field [kg/m^3]
    /// @param momentum_mf   Momentum field [kg*m/s]
    /// @param energy_mf     Total energy field [J/m^3]
    /// @param eta_mf        Volume fraction field (for mixed EOS)
    /// @param gamma_mf      Gamma field (written to ghost cells)
    /// @param pi_mf         Pi field (written to ghost cells)
    /// @param pressure_mf   Pressure field (written to ghost cells)
    /// @param eos0          EOS for fluid 0
    /// @param eos1          EOS for fluid 1
    /// @param geom          Geometry object
    /// @param time          Current simulation time
    /// @param pref          Reference pressure [Pa]
    template <typename EOS0, typename EOS1>
    void FillBoundary(amrex::MultiFab &density_mf,
                      amrex::MultiFab &momentum_mf,
                      amrex::MultiFab &energy_mf,
                      amrex::MultiFab &eta_mf,
                      amrex::MultiFab &gamma_mf,
                      amrex::MultiFab &pi_mf,
                      amrex::MultiFab &pressure_mf,
                      const EOS0 &eos0,
                      const EOS1 &eos1,
                      const amrex::Geometry &geom,
                      Set::Scalar time,
                      Set::Scalar pref)
    {
        const amrex::Box &domain = geom.Domain();
        const auto dx = geom.CellSizeArray();

        // Capture EOS parameters for GPU
        Set::Scalar g0_ = eos0.Gamma();
        Set::Scalar g1_ = eos1.Gamma();
        Set::Scalar pi0_ = eos0.P0();
        Set::Scalar pi1_ = eos1.P0();
        Set::Scalar cv0_ = eos0.Cv();
        Set::Scalar cv1_ = eos1.Cv();
        Set::Scalar pref_local = pref;
        Set::Scalar small_local = small;

        // Capture boundary parameters for GPU
        BoundaryParams params_xlo_local = params_xlo;
        BoundaryParams params_xhi_local = params_xhi;
        BoundaryParams params_ylo_local = params_ylo;
        BoundaryParams params_yhi_local = params_yhi;
#if AMREX_SPACEDIM == 3
        BoundaryParams params_zlo_local = params_zlo;
        BoundaryParams params_zhi_local = params_zhi;
#endif

        // =========================================================================
        // LAMBDA 1: COMPUTE PRIMITIVES FROM CONSERVED VARIABLES
        //
        // Computes: rho, un (normal velocity), ut (tangential velocity),
        //           p, a (sound speed), gamma, pi
        //
        // STABILITY FIX: Added velocity limiter to prevent Inf from near-zero rho
        // =========================================================================
        auto compute_prim = [=] AMREX_GPU_HOST_DEVICE(
                                const amrex::Array4<const amrex::Real> &eta_arr,
                                const amrex::Array4<const amrex::Real> &rho_arr,
                                const amrex::Array4<const amrex::Real> &M_arr,
                                const amrex::Array4<const amrex::Real> &E_arr,
                                int ii,
                                int jj,
                                int kk,
                                int dir,
                                Set::Scalar &rho_out,
                                Set::Scalar &un_out,
                                Set::Scalar &ut_out,
                                Set::Scalar &p_out,
                                Set::Scalar &a_out,
                                Set::Scalar &gamma_out,
                                Set::Scalar &pi_out) -> void {
            Set::Scalar eta_val = eta_arr(ii, jj, kk);
            Set::Scalar rho_val = amrex::max(rho_arr(ii, jj, kk), small_local);
            Set::Scalar mx = M_arr(ii, jj, kk, 0);
            Set::Scalar my = M_arr(ii, jj, kk, 1);
#if AMREX_SPACEDIM == 3
            Set::Scalar mz = M_arr(ii, jj, kk, 2);
#endif
            Set::Scalar E_val = E_arr(ii, jj, kk);

            // Velocities
            Set::Scalar vx = mx / rho_val;
            Set::Scalar vy = my / rho_val;
#if AMREX_SPACEDIM == 3
            Set::Scalar vz = mz / rho_val;
#endif

            // STABILITY FIX: Cap velocities to prevent Inf from near-zero rho
            // Without this, vx = mx/1e-14 can be enormous, making transverse terms
            // Inf, then dp_dn_new = Inf, ghost pressure = Inf, Riemann solver NaN
            const Set::Scalar vel_lim = Set::Scalar(1.0e7);
            if (!std::isfinite(vx) || std::abs(vx) > vel_lim)
                vx = Set::Scalar(0.0);
            if (!std::isfinite(vy) || std::abs(vy) > vel_lim)
                vy = Set::Scalar(0.0);
#if AMREX_SPACEDIM == 3
            if (!std::isfinite(vz) || std::abs(vz) > vel_lim)
                vz = Set::Scalar(0.0);
            Set::Scalar KE = 0.5 * rho_val * (vx * vx + vy * vy + vz * vz);
#else
            Set::Scalar KE = 0.5 * rho_val * (vx * vx + vy * vy);
#endif

            // Internal energy
            Set::Scalar UE = E_val - KE;
            if (!(UE > 0.0))
                UE = 0.0;

            // Mixed EOS properties (Motheau Eq. 4-6)
            Set::Scalar A = eta_val / (g0_ - 1.0) + (1.0 - eta_val) / (g1_ - 1.0);
            Set::Scalar B = (eta_val * g0_ * pi0_) / (g0_ - 1.0)
                            + ((1.0 - eta_val) * g1_ * pi1_) / (g1_ - 1.0);
            Set::Scalar gamma_mix = 1.0 + 1.0 / A;
            Set::Scalar pi_mix = (B / A) / gamma_mix;

            // Pressure (Motheau Eq. 5)
            Set::Scalar p_val = (gamma_mix - 1.0) * UE - gamma_mix * pi_mix + pref_local;
            if (!(p_val > 0.0))
                p_val = 1.0e-6;

            // Sound speed (Motheau Eq. 6)
            Set::Scalar a_val = std::sqrt(gamma_mix * (p_val + pi_mix) / rho_val);

            // Output based on direction
            rho_out = rho_val;
            if (dir == 0)
            {
                un_out = vx; // normal velocity
                ut_out = vy; // tangential velocity
            }
            else if (dir == 1)
            {
                un_out = vy;
                ut_out = vx;
#if AMREX_SPACEDIM == 3
            }
            else
            {
                un_out = vz;
                ut_out = vx;
#endif
            }
            p_out = p_val;
            a_out = a_val;
            gamma_out = gamma_mix;
            pi_out = pi_mix;
        };
        // =========================================================================
        // LAMBDA 2: WRITE CONSERVATIVE VARIABLES FROM PRIMITIVES
        //
        // Converts primitive ghost values (rho, un, ut, p) back to conservative
        // form (rho, M, E) and writes gamma, pi, pressure for Riemann solver.
        //
        // STABILITY FIXES:
        //   - Added gamma_arr, pi_arr, p_arr parameters to write to ghost cells
        //   - Added eta guard to prevent NaN gamma from bad eta values
        // =========================================================================
        auto set_ghost_cons = [=] AMREX_GPU_HOST_DEVICE(
                                  const amrex::Array4<const amrex::Real> &eta_arr,
                                  const amrex::Array4<amrex::Real> &rho_arr,
                                  const amrex::Array4<amrex::Real> &M_arr,
                                  const amrex::Array4<amrex::Real> &E_arr,
                                  const amrex::Array4<amrex::Real> &gamma_arr,
                                  const amrex::Array4<amrex::Real> &pi_arr,
                                  const amrex::Array4<amrex::Real> &p_arr,
                                  int ii,
                                  int jj,
                                  int kk,
                                  int dir,
                                  Set::Scalar rho_val,
                                  Set::Scalar un_val,
                                  Set::Scalar ut_val,
                                  Set::Scalar p_val) -> void {
            Set::Scalar eta_val = eta_arr(ii, jj, kk);

            // STABILITY FIX: Guard eta to prevent NaN gamma
            // If eta at ghost cell is NaN or out of range, clamp to [0,1]
            // A NaN eta flows through EOS and makes gamma/pi NaN, which then makes
            // UE = (p + NaN - pref)/(NaN-1) = NaN even when p and rho are finite
            if (!(eta_val >= 0.0 && eta_val <= 1.0))
                eta_val = amrex::max(Set::Scalar(0.0), amrex::min(Set::Scalar(1.0), eta_val));
            if (!std::isfinite(eta_val))
                eta_val = Set::Scalar(0.0);

            // Mixed EOS (Motheau Eq. 4-6)
            Set::Scalar A = eta_val / (g0_ - 1.0) + (1.0 - eta_val) / (g1_ - 1.0);
            Set::Scalar B = (eta_val * g0_ * pi0_) / (g0_ - 1.0)
                            + ((1.0 - eta_val) * g1_ * pi1_) / (g1_ - 1.0);
            Set::Scalar gamma_mix = 1.0 + 1.0 / A;
            Set::Scalar pi_mix = (B / A) / gamma_mix;

            // Inverse EOS: compute UE from pressure
            Set::Scalar UE = (p_val + gamma_mix * pi_mix - pref_local) / (gamma_mix - 1.0);
            if (UE < 0.0)
                UE = 0.0;

            // Velocities (convert from normal/tangential back to x/y)
            Set::Scalar vx, vy, vz;
            if (dir == 0)
            {
                vx = un_val; // normal = x
                vy = ut_val; // tangential = y
                vz = 0.0;
            }
            else if (dir == 1)
            {
                vx = ut_val; // tangential = x
                vy = un_val; // normal = y
                vz = 0.0;
#if AMREX_SPACEDIM == 3
            }
            else
            {
                vx = ut_val;
                vy = 0.0;
                vz = un_val;
#endif
            }

            // Kinetic energy
#if AMREX_SPACEDIM == 3
            Set::Scalar KE = 0.5 * rho_val * (vx * vx + vy * vy + vz * vz);
#else
            Set::Scalar KE = 0.5 * rho_val * (vx * vx + vy * vy);
#endif

            // Write conservative variables
            rho_arr(ii, jj, kk) = rho_val;
            M_arr(ii, jj, kk, 0) = rho_val * vx;
            M_arr(ii, jj, kk, 1) = rho_val * vy;
#if AMREX_SPACEDIM == 3
            M_arr(ii, jj, kk, 2) = rho_val * vz;
#endif
            E_arr(ii, jj, kk) = UE + KE;

            // STABILITY FIX: Write gamma, pi, pressure to ghost cells for Riemann solver
            gamma_arr(ii, jj, kk) = gamma_mix;
            pi_arr(ii, jj, kk) = pi_mix;
            p_arr(ii, jj, kk) = p_val;
        };
        // =========================================================================
        // MAIN LOOP OVER MULTIFAB BOXES
        // =========================================================================
        for (amrex::MFIter mfi(density_mf); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();

            // Get Array4 views (including gamma, pi, pressure for ghost cell writes)
            amrex::Array4<const amrex::Real> const &eta_arr = eta_mf.const_array(mfi);
            amrex::Array4<amrex::Real> const &rho_arr = density_mf.array(mfi);
            amrex::Array4<amrex::Real> const &M_arr = momentum_mf.array(mfi);
            amrex::Array4<amrex::Real> const &E_arr = energy_mf.array(mfi);
            amrex::Array4<amrex::Real> const &gamma_arr = gamma_mf.array(mfi);
            amrex::Array4<amrex::Real> const &pi_arr = pi_mf.array(mfi);
            amrex::Array4<amrex::Real> const &p_arr = pressure_mf.array(mfi);
            // =====================================================================
            // FACE LOOP: Process each of the 4 boundary faces (xlo, xhi, ylo, yhi)
            // =====================================================================
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
            {
                Set::Scalar delta = dx[dir]; // cell size in normal direction

                for (int side = 0; side < 2; ++side) // side=0 -> lo, side=1 -> hi
                {
                    // Determine which face we're processing
                    // face_idx: 0=xlo, 1=xhi, 2=ylo, 3=yhi
                    int face_idx = dir * 2 + side;

                    // Get boundary parameters for this face
                    BoundaryParams params_face;
                    if (face_idx == 0)
                        params_face = params_xlo_local;
                    else if (face_idx == 1)
                        params_face = params_xhi_local;
                    else if (face_idx == 2)
                        params_face = params_ylo_local;
                    else if (face_idx == 3)
                        params_face = params_yhi_local;

                    if (params_face.type == Type::None)
                        continue;

                    // Check if this tile touches the physical domain boundary
                    int ibdry;
                    if (side == 0)
                    {
                        if (bx.smallEnd(dir) != domain.smallEnd(dir))
                            continue;
                        ibdry = domain.smallEnd(dir);
                    }
                    else
                    {
                        if (bx.bigEnd(dir) != domain.bigEnd(dir))
                            continue;
                        ibdry = domain.bigEnd(dir);
                    }

                    // Build edge box (1-cell-thick slice along the boundary)
                    // CORRECTED: Include ALL transverse indices (including corners)
                    int ng = density_mf.nGrow();
                    amrex::Box edge_box = bx;
                    edge_box.setSmall(dir, ibdry);
                    edge_box.setBig(dir, ibdry);

                    // Capture parameters for GPU kernel
                    Set::Scalar p_t = params_face.target_p;
                    Set::Scalar un_t = params_face.target_u;
                    Set::Scalar ut_t = params_face.target_v;
                    Set::Scalar T_t = params_face.target_T;
                    Set::Scalar eta_r = params_face.relax_u;
                    Set::Scalar sigma_face = params_face.sigma;
                    Set::Scalar beta_face = params_face.beta;
                    Set::Scalar L_ref = params_face.L_ref;
                    Type ftype = params_face.type;
                    int side_local = side;
                    int dir_local = dir;
                    int ibdry_local = ibdry;
                    int ng_total = ng;

                    amrex::ParallelFor(edge_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        // =========================================================
                        // STEP 1: Compute primitives at boundary and interior neighbors
                        // =========================================================

                        // Neighbor offsets in normal direction
                        int di = (dir_local == 0) ? 1 : 0;
                        int dj = (dir_local == 1) ? 1 : 0;
                        int sign_in = (side_local == 1) ? -1 : 1; // -1 for hi, +1 for lo

                        int im1 = i + sign_in * di;
                        int jm1 = j + sign_in * dj;

                        // Primitives at boundary
                        Set::Scalar rho_b, un_b, ut_b, p_b, a_b, gm_b, pi_b;
                        compute_prim(eta_arr, rho_arr, M_arr, E_arr, i, j, k, dir_local, rho_b, un_b, ut_b, p_b, a_b, gm_b, pi_b);

                        // Early exit if boundary cell has bad energy
                        if (!std::isfinite(E_arr(i, j, k)) || !std::isfinite(M_arr(i, j, k, 0)) || !std::isfinite(M_arr(i, j, k, 1)))
                            return;

                        // Primitives at first interior neighbor
                        Set::Scalar rho_m1, un_m1, ut_m1, p_m1, a_m1, gm_m1, pi_m1;
                        compute_prim(eta_arr, rho_arr, M_arr, E_arr, im1, jm1, k, dir_local, rho_m1, un_m1, ut_m1, p_m1, a_m1, gm_m1, pi_m1);

                        Set::Scalar a2_b = a_b * a_b;
                        Set::Scalar M_loc = std::abs(un_b) / (a_b + small_local);

                        // =========================================================
                        // STEP 2: 1st-order one-sided FD in normal direction
                        //
                        // CORRECTED: True 1st-order (not the broken 2nd-order stencil)
                        //   hi side: df/dn = (f_b - f_m1) / delta
                        //   lo side: df/dn = (f_m1 - f_b) / delta
                        // =========================================================
                        Set::Scalar fd_sign = (side_local == 1) ? 1.0 : -1.0;
                        Set::Scalar drho_dn = fd_sign * (rho_b - rho_m1) / delta;
                        Set::Scalar dun_dn = fd_sign * (un_b - un_m1) / delta;
                        Set::Scalar dut_dn = fd_sign * (ut_b - ut_m1) / delta;
                        Set::Scalar dp_dn = fd_sign * (p_b - p_m1) / delta;

                        // =========================================================
                        // STEP 3: Central 2nd-order FD in transverse direction
                        // =========================================================
                        int ti = (dir_local == 1) ? 1 : 0;
                        int tj = (dir_local == 0) ? 1 : 0;
                        int itp = amrex::min(i + ti, domain.bigEnd(0));
                        int itm = amrex::max(i - ti, domain.smallEnd(0));
                        int jtp = amrex::min(j + tj, domain.bigEnd(1));
                        int jtm = amrex::max(j - tj, domain.smallEnd(1));

                        int nsteps_t = (itp - itm) + (jtp - jtm);
                        Set::Scalar dt_denom = dx[1 - dir_local] * amrex::max(nsteps_t, 1);

                        Set::Scalar rho_tp, un_tp, ut_tp, p_tp, a_tp, gm_tp, pi_tp;
                        compute_prim(eta_arr, rho_arr, M_arr, E_arr, itp, jtp, k, dir_local, rho_tp, un_tp, ut_tp, p_tp, a_tp, gm_tp, pi_tp);

                        Set::Scalar rho_tm, un_tm, ut_tm, p_tm, a_tm, gm_tm, pi_tm;
                        compute_prim(eta_arr, rho_arr, M_arr, E_arr, itm, jtm, k, dir_local, rho_tm, un_tm, ut_tm, p_tm, a_tm, gm_tm, pi_tm);

                        Set::Scalar drho_dt = (rho_tp - rho_tm) / dt_denom;
                        Set::Scalar dun_dt = (un_tp - un_tm) / dt_denom;
                        Set::Scalar dut_dt = (ut_tp - ut_tm) / dt_denom;
                        Set::Scalar dp_dt = (p_tp - p_tm) / dt_denom;
                        // =========================================================
                        // STEP 4: LODI wave amplitudes from interior values
                        //
                        // CORRECTED FORMULATION (Motheau Eq. 23):
                        //   L1 = (un-a)/(2*a^2) * (dp/dn - rho*a*dun/dn)  [backward acoustic]
                        //   L2 = un/a^2 * (a^2*drho/dn - dp/dn)           [entropy]
                        //   L3 = un * dut/dn                              [shear]
                        //   L4 = (un+a)/(2*a^2) * (dp/dn + rho*a*dun/dn)  [forward acoustic]
                        //
                        // KEY FIX: Include 1/(2*a^2) and 1/a^2 eigenvalue normalization
                        // Without these, L-waves are amplified by ~2*a^2 (4.5 million for water!)
                        // =========================================================
                        Set::Scalar L1 = (un_b - a_b) / (2.0 * a2_b) * (dp_dn - rho_b * a_b * dun_dn);
                        Set::Scalar L2 = (std::abs(un_b) > 1.0e-14) ? un_b / a2_b * (a2_b * drho_dn - dp_dn) : 0.0;
                        Set::Scalar L3 = (std::abs(un_b) > 1.0e-14) ? un_b * dut_dn : 0.0;
                        Set::Scalar L4 = (un_b + a_b) / (2.0 * a2_b) * (dp_dn + rho_b * a_b * dun_dn);

                        // =========================================================
                        // STEP 5: Transverse source terms (Motheau Eqs. 36-37)
                        //
                        // These are the "T" terms included in the incoming wave modification
                        // T1 involves the backward acoustic direction (L1 wave)
                        // T4 involves the forward acoustic direction (L4 wave)
                        // T2 and T3 are for entropy and shear
                        // =========================================================
                        Set::Scalar T1 = (un_b - a_b) / (2.0 * a2_b) * (-rho_b * a_b * ut_b * dun_dt + a2_b * rho_b * dut_dt + ut_b * dp_dt);
                        Set::Scalar T4 = (un_b + a_b) / (2.0 * a2_b) * (rho_b * a_b * ut_b * dun_dt + a2_b * rho_b * dut_dt + ut_b * dp_dt);
                        Set::Scalar T2 = (std::abs(un_b) > 1.0e-14) ? un_b * ut_b / a2_b * (a2_b * drho_dt - dp_dt) : 0.0;
                        Set::Scalar T3 = (std::abs(un_b) > 1.0e-14) ? un_b * (ut_b * dut_dt + (1.0 / rho_b) * dp_dt) : 0.0;
                        // =========================================================
                        // STEP 6: Apply BC-specific modification to incoming wave(s)
                        //
                        // Convention (subsonic, standard orientation):
                        //   hi side (xhi/yhi): L1 is the incoming acoustic (speed un-a < 0)
                        //   lo side (xlo/ylo): L4 is the incoming acoustic (speed un+a > 0)
                        //
                        // CORRECTED K coefficient (Motheau Eq. 35):
                        //   K = sigma * (1 - M^2) / (a * L_ref)  [units: 1/(m*s)]
                        //
                        // For OUTFLOW: damp the single incoming acoustic wave
                        //   L_in = K*(p - p_t) - (1-beta)*T_in
                        //
                        // For INFLOW: specify all three incoming waves (Motheau Eqs. 42-44)
                        //   Acoustic: L_in = eta_relax*(rho*(1-M^2)/L)*(un - un_t) - T_in
                        //   Entropy:  L2   = eta_relax*(rho*a*(gm-1)/L)*(T - T_t) - T2
                        //   Shear:    L3   = eta_relax*(a/L)*(ut - ut_t) - T3
                        // =========================================================

                        // CORRECTED: K has units [1/(m*s)] for dimensional consistency
                        Set::Scalar K = sigma_face * (1.0 - M_loc * M_loc) / (a_b * L_ref);

                        // Local temperature (needed for inflow entropy relaxation)
                        Set::Scalar eta_b = eta_arr(i, j, k);
                        Set::Scalar cv_mix = eta_b * cv0_ + (1.0 - eta_b) * cv1_;
                        Set::Scalar T_b = (p_b + pi_b) / (rho_b * cv_mix * (gm_b - 1.0) + small_local);

                        if (ftype == Type::Outflow)
                        {
                            // hi side: incoming = L1; lo side: incoming = L4
                            if (side_local == 1)
                            {
                                // hi boundary (xhi or yhi)
                                L1 = K * (p_b - p_t) - (1.0 - beta_face) * T1;
                            }
                            else
                            {
                                // lo boundary (xlo or ylo)
                                L4 = K * (p_b - p_t) - (1.0 - beta_face) * T4;
                            }
                        }
                        else if (ftype == Type::Inflow)
                        {
                            Set::Scalar K_acoustic = eta_r * rho_b * (1.0 - M_loc * M_loc) / L_ref;
                            Set::Scalar K_entropy = eta_r * rho_b * a_b * (gm_b - 1.0) / L_ref;
                            Set::Scalar K_shear = eta_r * a_b / L_ref;

                            if (side_local == 1)
                            {
                                // hi boundary: L1 is incoming
                                L1 = -K_acoustic * (un_b - un_t) - T1;
                                L2 = -K_entropy * (T_b - T_t) - T2;
                                L3 = -K_shear * (ut_b - ut_t) - T3;
                            }
                            else
                            {
                                // lo boundary: L4 is incoming
                                L4 = K_acoustic * (un_b - un_t) - T4;
                                L2 = K_entropy * (T_b - T_t) - T2;
                                L3 = K_shear * (ut_b - ut_t) - T3;
                            }
                        }

                        // =========================================================
                        // STEP 7: Reconstruct modified primitive gradients
                        //
                        // Motheau Eq. 24-26:
                        //   xi1 = L1/(un-a),  xi2 = L2/un,  xi3 = L3/un,  xi4 = L4/(un+a)
                        //
                        //   drho/dn = xi1 + xi2 + xi4
                        //   dun/dn  = (-a/rho)*xi1 + (a/rho)*xi4
                        //   dut/dn  = xi3
                        //   dp/dn   = a^2*(xi1 + xi4)
                        // =========================================================
                        Set::Scalar xi1 = L1 / (un_b - a_b + small_local);
                        Set::Scalar xi2 = (std::abs(un_b) > 1.0e-14) ? L2 / un_b : 0.0;
                        Set::Scalar xi3 = (std::abs(un_b) > 1.0e-14) ? L3 / un_b : 0.0;
                        Set::Scalar xi4 = L4 / (un_b + a_b + small_local);

                        Set::Scalar drho_dn_new = xi1 + xi2 + xi4;
                        Set::Scalar dun_dn_new = (-a_b / rho_b) * xi1 + (a_b / rho_b) * xi4;
                        Set::Scalar dut_dn_new = xi3;
                        Set::Scalar dp_dn_new = a2_b * (xi1 + xi4);

                        // Guard: if any gradient is NaN or Inf, zero it for flat extrapolation
                        if (!std::isfinite(drho_dn_new))
                            drho_dn_new = 0.0;
                        if (!std::isfinite(dun_dn_new))
                            dun_dn_new = 0.0;
                        if (!std::isfinite(dut_dn_new))
                            dut_dn_new = 0.0;
                        if (!std::isfinite(dp_dn_new))
                            dp_dn_new = 0.0;
                        // =========================================================
                        // STEP 8: Fill ghost cells from modified gradients
                        //
                        // hi side (right/top): ghost at N+1, N+2, ...
                        //   Q_{N+1} = Q_{N-1} + 2*delta * dQ/dn  (Motheau Eq. 31)
                        //   Q_{N+2} = 2*Q_{N+1} - Q_N            (simple linear)
                        //
                        // lo side (left/bottom): ghost at -1, -2, ...
                        //   Q_{-1} = Q_{+1} - 2*delta * dQ/dn    (Motheau Eq. 27)
                        //   Q_{-2} = 2*Q_{-1} - Q_0              (simple linear)
                        //
                        // Layers 3+: constant extrapolation (copy layer 2)
                        // =========================================================

                        // Ghost cell positions in normal direction
                        int ig1_i = i + (-sign_in) * di;
                        int jg1_j = j + (-sign_in) * dj;
                        int ig2_i = i + (-sign_in) * 2 * di;
                        int jg2_j = j + (-sign_in) * 2 * dj;

                        // Layer 1: use Q_{m1} (1 cell interior) as reference
                        Set::Scalar sign_gh = (side_local == 1) ? 1.0 : -1.0;
                        Set::Scalar rho_g1 = rho_m1 + sign_gh * 2.0 * delta * drho_dn_new;
                        Set::Scalar un_g1 = un_m1 + sign_gh * 2.0 * delta * dun_dn_new;
                        Set::Scalar ut_g1 = ut_m1 + sign_gh * 2.0 * delta * dut_dn_new;
                        Set::Scalar p_g1 = p_m1 + sign_gh * 2.0 * delta * dp_dn_new;

                        // Fallback: if NSCBC produces bad pressure, use zero-gradient
                        if (!std::isfinite(p_g1) || p_g1 < Set::Scalar(1.0e-3) * p_b)
                        {
                            // Copy boundary state to all ghost layers
                            for (int ig = 1; ig <= ng_total; ++ig)
                            {
                                set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, gamma_arr, pi_arr, p_arr, i + (-sign_in) * ig * di, j + (-sign_in) * ig * dj, k, dir_local, rho_b, un_b, ut_b, p_b);
                            }
                            return;
                        }

                        // Guards for layer 1
                        if (!(rho_g1 > 0.0))
                            rho_g1 = rho_b;
                        if (!std::isfinite(un_g1))
                            un_g1 = 0.0;
                        if (!std::isfinite(ut_g1))
                            ut_g1 = 0.0;
                        rho_g1 = amrex::max(rho_g1, small_local);
                        p_g1 = amrex::max(p_g1, Set::Scalar(1.0e-6));

                        // Write layer 1
                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, gamma_arr, pi_arr, p_arr, ig1_i, jg1_j, k, dir_local, rho_g1, un_g1, ut_g1, p_g1);

                        // Layer 2: simple linear extrapolation g2 = 2*g1 - b
                        // (reduces amplification from 6x to 2x compared to Motheau Eq. 32)
                        Set::Scalar rho_g2 = 2.0 * rho_g1 - rho_b;
                        Set::Scalar un_g2 = 2.0 * un_g1 - un_b;
                        Set::Scalar ut_g2 = 2.0 * ut_g1 - ut_b;
                        Set::Scalar p_g2 = 2.0 * p_g1 - p_b;

                        // Guards for layer 2
                        if (!(rho_g2 > 0.0))
                            rho_g2 = rho_b;
                        if (!std::isfinite(p_g2) || !(p_g2 > 0.0))
                            p_g2 = p_b;
                        if (!std::isfinite(un_g2))
                            un_g2 = 0.0;
                        if (!std::isfinite(ut_g2))
                            ut_g2 = 0.0;
                        rho_g2 = amrex::max(rho_g2, small_local);
                        p_g2 = amrex::max(p_g2, Set::Scalar(1.0e-6));

                        // Write layer 2
                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, gamma_arr, pi_arr, p_arr, ig2_i, jg2_j, k, dir_local, rho_g2, un_g2, ut_g2, p_g2);

                        // Layers 3+: constant extrapolation (copy layer 2 state)
                        for (int ig = 3; ig <= ng_total; ++ig)
                        {
                            set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, gamma_arr, pi_arr, p_arr, i + (-sign_in) * ig * di, j + (-sign_in) * ig * dj, k, dir_local, rho_g2, un_g2, ut_g2, p_g2);
                        }
                    }); // end ParallelFor edge_box

                } // end side loop
            } // end dir loop
            // =====================================================================
            // CORNER TREATMENT: Lodato (2008) coupled system with T-tilde corrections
            //
            // For cells at the intersection of two NSCBC faces, solve a 2x2 coupled
            // system for the two incoming acoustic wave amplitudes and fill corner
            // ghost cells from both normal-direction gradients.
            // =====================================================================
            for (int cx = 0; cx < 2; ++cx)     // cx=0 -> xlo, cx=1 -> xhi
                for (int cy = 0; cy < 2; ++cy) // cy=0 -> ylo, cy=1 -> yhi
                {
                    int fx = cx;     // face index for x: 0=xlo, 1=xhi
                    int fy = 2 + cy; // face index for y: 2=ylo, 3=yhi

                    BoundaryParams params_x = (cx == 0) ? params_xlo_local : params_xhi_local;
                    BoundaryParams params_y = (cy == 0) ? params_ylo_local : params_yhi_local;

                    if (params_x.type == Type::None || params_y.type == Type::None)
                        continue;

                    // Check if this tile touches both corner-contributing boundaries
                    int ibdry_x = (cx == 0) ? domain.smallEnd(0) : domain.bigEnd(0);
                    int jbdry_y = (cy == 0) ? domain.smallEnd(1) : domain.bigEnd(1);

                    bool touch_x = (cx == 0) ? (bx.smallEnd(0) == domain.smallEnd(0))
                                             : (bx.bigEnd(0) == domain.bigEnd(0));
                    bool touch_y = (cy == 0) ? (bx.smallEnd(1) == domain.smallEnd(1))
                                             : (bx.bigEnd(1) == domain.bigEnd(1));

                    if (!touch_x || !touch_y)
                        continue;

                    // Build corner ghost box
                    int ng = density_mf.nGrow();
                    int sx = (cx == 1) ? 1 : -1;
                    int sy = (cy == 1) ? 1 : -1;

                    amrex::Box corner_box;
                    corner_box.setSmall(0, (cx == 1) ? ibdry_x + 1 : ibdry_x - ng);
                    corner_box.setBig(0, (cx == 1) ? ibdry_x + ng : ibdry_x - 1);
                    corner_box.setSmall(1, (cy == 1) ? jbdry_y + 1 : jbdry_y - ng);
                    corner_box.setBig(1, (cy == 1) ? jbdry_y + ng : jbdry_y - 1);
#if AMREX_SPACEDIM == 3
                    corner_box.setSmall(2, domain.smallEnd(2));
                    corner_box.setBig(2, domain.bigEnd(2));
#endif

                    // Capture per-face scalars for corner kernel
                    Set::Scalar px_t = params_x.target_p;
                    Set::Scalar py_t = params_y.target_p;
                    Set::Scalar sigma_x = params_x.sigma;
                    Set::Scalar sigma_y = params_y.sigma;
                    Set::Scalar beta_x = params_x.beta;
                    Set::Scalar beta_y = params_y.beta;
                    Set::Scalar Lx_ref = params_x.L_ref;
                    Set::Scalar Ly_ref = params_y.L_ref;
                    Type ftx = params_x.type;
                    Type fty = params_y.type;
                    Set::Scalar etar_x = params_x.relax_u;
                    Set::Scalar etar_y = params_y.relax_u;
                    Set::Scalar un_t_x = params_x.target_u;
                    Set::Scalar un_t_y = params_y.target_u;
                    int sx_local = sx;
                    int sy_local = sy;
                    int ibdry_x_local = ibdry_x;
                    int jbdry_y_local = jbdry_y;

                    amrex::ParallelFor(corner_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        int ic = ibdry_x_local;
                        int jc = jbdry_y_local;

                        // Primitives at corner boundary cell
                        Set::Scalar rho_c, vx_c, vy_c, p_c, a_c, gm_c, pi_c;
                        {
                            Set::Scalar un_tmp, ut_tmp;
                            compute_prim(eta_arr, rho_arr, M_arr, E_arr, ic, jc, k, 0, rho_c, un_tmp, ut_tmp, p_c, a_c, gm_c, pi_c);
                            vx_c = un_tmp;
                            vy_c = ut_tmp;
                        }

                        Set::Scalar a2_c = a_c * a_c;
                        Set::Scalar Mx_c = std::abs(vx_c) / (a_c + small_local);
                        Set::Scalar My_c = std::abs(vy_c) / (a_c + small_local);

                        // X-direction: 1st-order one-sided FD at corner
                        int im1 = ic - sx_local;
                        Set::Scalar rho_xm1, vx_xm1, vy_xm1, p_xm1, a_xm1, gm_xm1, pi_xm1;
                        {
                            Set::Scalar un_tmp, ut_tmp;
                            compute_prim(eta_arr, rho_arr, M_arr, E_arr, im1, jc, k, 0, rho_xm1, un_tmp, ut_tmp, p_xm1, a_xm1, gm_xm1, pi_xm1);
                            vx_xm1 = un_tmp;
                            vy_xm1 = ut_tmp;
                        }

                        Set::Scalar fdx = (Set::Scalar)sx_local;
                        Set::Scalar drho_dx_c = fdx * (rho_c - rho_xm1) / dx[0];
                        Set::Scalar dvx_dx_c = fdx * (vx_c - vx_xm1) / dx[0];
                        Set::Scalar dvy_dx_c = fdx * (vy_c - vy_xm1) / dx[0];
                        Set::Scalar dp_dx_c = fdx * (p_c - p_xm1) / dx[0];

                        // Y-direction: 1st-order one-sided FD at corner
                        int jm1 = jc - sy_local;
                        Set::Scalar rho_ym1, vy_ym1, vx_ym1, p_ym1, a_ym1, gm_ym1, pi_ym1;
                        {
                            Set::Scalar un_tmp, ut_tmp;
                            compute_prim(eta_arr, rho_arr, M_arr, E_arr, ic, jm1, k, 1, rho_ym1, un_tmp, ut_tmp, p_ym1, a_ym1, gm_ym1, pi_ym1);
                            vy_ym1 = un_tmp;
                            vx_ym1 = ut_tmp;
                        }

                        Set::Scalar fdy = (Set::Scalar)sy_local;
                        Set::Scalar drho_dy_c = fdy * (rho_c - rho_ym1) / dx[1];
                        Set::Scalar dvy_dy_c = fdy * (vy_c - vy_ym1) / dx[1];
                        Set::Scalar dvx_dy_c = fdy * (vx_c - vx_ym1) / dx[1];
                        Set::Scalar dp_dy_c = fdy * (p_c - p_ym1) / dx[1];

                        // LODI waves from x-direction at corner (CORRECTED with 1/(2*a^2) scaling)
                        Set::Scalar L1_x = (vx_c - a_c) / (2.0 * a2_c) * (dp_dx_c - rho_c * a_c * dvx_dx_c);
                        Set::Scalar L2_x = (std::abs(vx_c) > 1.0e-14) ? vx_c / a2_c * (a2_c * drho_dx_c - dp_dx_c) : 0.0;
                        Set::Scalar L3_x = (std::abs(vx_c) > 1.0e-14) ? vx_c * dvy_dx_c : 0.0;
                        Set::Scalar L4_x = (vx_c + a_c) / (2.0 * a2_c) * (dp_dx_c + rho_c * a_c * dvx_dx_c);

                        // LODI waves from y-direction at corner (CORRECTED with 1/(2*a^2) scaling)
                        Set::Scalar L1_y = (vy_c - a_c) / (2.0 * a2_c) * (dp_dy_c - rho_c * a_c * dvy_dy_c);
                        Set::Scalar L2_y = (std::abs(vy_c) > 1.0e-14) ? vy_c / a2_c * (a2_c * drho_dy_c - dp_dy_c) : 0.0;
                        Set::Scalar L3_y = (std::abs(vy_c) > 1.0e-14) ? vy_c * dvx_dy_c : 0.0;
                        Set::Scalar L4_y = (vy_c + a_c) / (2.0 * a2_c) * (dp_dy_c + rho_c * a_c * dvy_dy_c);

                        // Identify incoming acoustic waves for this corner
                        Set::Scalar Lx_in = (cx == 1) ? L1_x : L4_x;
                        Set::Scalar Ly_in = (cy == 1) ? L1_y : L4_y;

                        // CORRECTED K coefficients with proper units [1/(m*s)]
                        Set::Scalar Kx = sigma_x * (1.0 - Mx_c * Mx_c) / (a_c * Lx_ref);
                        Set::Scalar Ky = sigma_y * (1.0 - My_c * My_c) / (a_c * Ly_ref);

                        Set::Scalar ab = (1.0 - 0.5 * (beta_x + beta_y)) / 2.0;

                        // Lodato (2008) T-tilde transverse corrections for corners
                        Set::Scalar T5_x = a2_c * dvy_dy_c;
                        Set::Scalar T5_y = a2_c * dvx_dx_c;

                        Set::Scalar Lx_out = (cx == 1) ? L4_x : L1_x;
                        Set::Scalar Ly_out = (cy == 1) ? L4_y : L1_y;

                        Set::Scalar T2_xface = (std::abs(vx_c) > 1.0e-14) ? vx_c * vy_c / a2_c * (a2_c * drho_dy_c - dp_dy_c) : 0.0;
                        Set::Scalar T3_yface = (std::abs(vy_c) > 1.0e-14) ? vy_c * (vx_c * dvx_dx_c + (1.0 / rho_c) * dp_dx_c) : 0.0;

                        Set::Scalar zeta_x = (cx == 1) ? 1.0 : -1.0;
                        Set::Scalar zeta_y = (cy == 1) ? 1.0 : -1.0;

                        Set::Scalar Ttilde_x = T5_x - 0.5 * Ly_out
                                               - zeta_x * rho_c * a_c * (L2_y - T2_xface);
                        Set::Scalar Ttilde_y = T5_y - 0.5 * Lx_out
                                               - zeta_y * rho_c * a_c * (L3_x - T3_yface);

                        // Apply BC-specific corner coupling
                        if (ftx == Type::Outflow && fty == Type::Outflow)
                        {
                            // Lodato Eq. 42: coupled outflow/outflow with T-tilde corrections
                            Set::Scalar bx = Kx * (p_c - px_t) + (1.0 - beta_x) * Ttilde_x;
                            Set::Scalar by = Ky * (p_c - py_t) + (1.0 - beta_y) * Ttilde_y;
                            Set::Scalar det_inv = 1.0 / (1.0 - ab * ab + 1.0e-20);
                            Lx_in = det_inv * (bx - ab * by);
                            Ly_in = det_inv * (by - ab * bx);
                        }
                        else if (ftx == Type::Outflow && fty == Type::Inflow)
                        {
                            Ly_in = 0.0;
                            Lx_in = Kx * (p_c - px_t) + (1.0 - beta_x) * T5_x;
                        }
                        else if (ftx == Type::Inflow && fty == Type::Outflow)
                        {
                            Lx_in = 0.0;
                            Ly_in = Ky * (p_c - py_t) + (1.0 - beta_y) * T5_y;
                        }
                        else if (ftx == Type::Inflow && fty == Type::Inflow)
                        {
                            Set::Scalar Kax = etar_x * rho_c * (1.0 - Mx_c * Mx_c) / Lx_ref;
                            Set::Scalar Kay = etar_y * rho_c * (1.0 - My_c * My_c) / Ly_ref;
                            Lx_in = (cx == 1) ? (-Kax * (vx_c - un_t_x)) : (Kax * (vx_c - un_t_x));
                            Ly_in = (cy == 1) ? (-Kay * (vy_c - un_t_y)) : (Kay * (vy_c - un_t_y));
                        }

                        // Update incoming waves
                        if (cx == 1)
                            L1_x = Lx_in;
                        else
                            L4_x = Lx_in;
                        if (cy == 1)
                            L1_y = Ly_in;
                        else
                            L4_y = Ly_in;

                        // Reconstruct gradients from modified L-waves
                        Set::Scalar xi1_x = L1_x / (vx_c - a_c + small_local);
                        Set::Scalar xi2_x = (std::abs(vx_c) > 1.0e-14) ? L2_x / vx_c : 0.0;
                        Set::Scalar xi3_x = (std::abs(vx_c) > 1.0e-14) ? L3_x / vx_c : 0.0;
                        Set::Scalar xi4_x = L4_x / (vx_c + a_c + small_local);

                        Set::Scalar drho_dx_new = xi1_x + xi2_x + xi4_x;
                        Set::Scalar dvx_dx_new = (-a_c / rho_c) * xi1_x + (a_c / rho_c) * xi4_x;
                        Set::Scalar dvy_dx_new = xi3_x;
                        Set::Scalar dp_dx_new = a2_c * (xi1_x + xi4_x);

                        Set::Scalar xi1_y = L1_y / (vy_c - a_c + small_local);
                        Set::Scalar xi2_y = (std::abs(vy_c) > 1.0e-14) ? L2_y / vy_c : 0.0;
                        Set::Scalar xi3_y = (std::abs(vy_c) > 1.0e-14) ? L3_y / vy_c : 0.0;
                        Set::Scalar xi4_y = L4_y / (vy_c + a_c + small_local);

                        Set::Scalar drho_dy_new = xi1_y + xi2_y + xi4_y;
                        Set::Scalar dvy_dy_new = (-a_c / rho_c) * xi1_y + (a_c / rho_c) * xi4_y;
                        Set::Scalar dvx_dy_new = xi3_y;
                        Set::Scalar dp_dy_new = a2_c * (xi1_y + xi4_y);

                        // NaN guards
                        if (!std::isfinite(drho_dx_new))
                            drho_dx_new = 0.0;
                        if (!std::isfinite(dvx_dx_new))
                            dvx_dx_new = 0.0;
                        if (!std::isfinite(dvy_dx_new))
                            dvy_dx_new = 0.0;
                        if (!std::isfinite(dp_dx_new))
                            dp_dx_new = 0.0;
                        if (!std::isfinite(drho_dy_new))
                            drho_dy_new = 0.0;
                        if (!std::isfinite(dvy_dy_new))
                            dvy_dy_new = 0.0;
                        if (!std::isfinite(dvx_dy_new))
                            dvx_dy_new = 0.0;
                        if (!std::isfinite(dp_dy_new))
                            dp_dy_new = 0.0;

                        // Fill corner ghost cell from both direction gradients
                        int ox = i - ic;
                        int oy = j - jc;

                        Set::Scalar rho_g = rho_c + (Set::Scalar)ox * dx[0] * drho_dx_new
                                            + (Set::Scalar)oy * dx[1] * drho_dy_new;
                        Set::Scalar vx_g = vx_c + (Set::Scalar)ox * dx[0] * dvx_dx_new
                                           + (Set::Scalar)oy * dx[1] * dvx_dy_new;
                        Set::Scalar vy_g = vy_c + (Set::Scalar)ox * dx[0] * dvy_dx_new
                                           + (Set::Scalar)oy * dx[1] * dvy_dy_new;
                        Set::Scalar p_g = p_c + (Set::Scalar)ox * dx[0] * dp_dx_new
                                          + (Set::Scalar)oy * dx[1] * dp_dy_new;

                        // Guards and bounds
                        if (!(rho_g > 0.0))
                            rho_g = rho_c;
                        if (!std::isfinite(p_g) || !(p_g > 0.0))
                            p_g = p_c;
                        if (!std::isfinite(vx_g))
                            vx_g = 0.0;
                        if (!std::isfinite(vy_g))
                            vy_g = 0.0;

                        rho_g = amrex::min(amrex::max(rho_g, Set::Scalar(1.0e-8)), Set::Scalar(10000.0));
                        p_g = amrex::min(amrex::max(p_g, Set::Scalar(1.0)), Set::Scalar(1.0e8));
                        vx_g = amrex::min(amrex::max(vx_g, Set::Scalar(-1000.0)), Set::Scalar(1000.0));
                        vy_g = amrex::min(amrex::max(vy_g, Set::Scalar(-1000.0)), Set::Scalar(1000.0));

                        // Write to conserved arrays (use dir=0 since we have vx,vy separately)
                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, gamma_arr, pi_arr, p_arr, i, j, k, 0, rho_g, vx_g, vy_g, p_g);
                    }); // end ParallelFor corner_box

                } // end corner loops

        } // end MFIter loop
        // =========================================================================
        // DIAGNOSTIC: Scan ghost cells for NaN and print location
        // =========================================================================
        amrex::Gpu::synchronize();
        for (amrex::MFIter mfi(density_mf, false); mfi.isValid(); ++mfi)
        {
            const amrex::Box vbx = mfi.validbox();
            const amrex::Box gbx = amrex::grow(vbx, density_mf.nGrow());
            auto const &rho_scan = density_mf.array(mfi);
            auto const &E_scan = energy_mf.array(mfi);
            auto const &eta_scan = eta_mf.array(mfi);

            amrex::LoopOnCpu(gbx, [&](int i, int j, int k) {
                bool in_valid = vbx.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)));
                if (amrex::isnan(rho_scan(i, j, k)))
                    amrex::Print() << "[NSCBC-NaN] rho NaN at ("
                                   << i << "," << j << "," << k << ")"
                                   << (in_valid ? " [interior]" : " [ghost]") << "\n";
                if (amrex::isnan(E_scan(i, j, k)))
                    amrex::Print() << "[NSCBC-NaN] E NaN at ("
                                   << i << "," << j << "," << k << ")"
                                   << (in_valid ? " [interior]" : " [ghost]") << "\n";
                if (amrex::isnan(eta_scan(i, j, k)))
                    amrex::Print() << "[NSCBC-NaN] eta NaN at ("
                                   << i << "," << j << "," << k << ")"
                                   << (in_valid ? " [interior]" : " [ghost]") << "\n";
            });
        }

    } // end FillBoundary()

}; // end class NSCBC

} // namespace BC

#endif // BC_NSCBC_H

