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
/// This class fills ghost cells using characteristic wave decomposition
/// to provide non-reflecting boundary conditions for compressible flow.
///
/// CRITICAL: Ghost cells are filled in the following order (Motheau Figure 2):
///   1. CORNERS FIRST (both 2 layers)
///   2. EDGES SECOND (skip corners since already filled)
///
/// This ordering is essential to prevent NaN propagation from uninitialized corner data.
/// NOTE: Modified to use only 2 ghost cell layers instead of 4.
class NSCBC
{
public:
    enum class Type
    {
        None,
        Inflow,    ///< Subsonic inflow with relaxation to target state
        Outflow,   ///< Subsonic outflow with pressure relaxation
        SlipWall,  ///< Inviscid wall (tangential slip allowed)
        NoSlipWall ///< Viscous wall (no slip)
    };

    /// Parameters for each boundary face
    struct BoundaryParams
    {
        Type type = Type::Outflow;

        // Target values for relaxation
        Set::Scalar target_u = 0.0;      ///< Target normal velocity [m/s]
        Set::Scalar target_v = 0.0;      ///< Target tangential velocity [m/s]
        Set::Scalar target_w = 0.0;      ///< Target second tangential velocity [m/s] (3D only)
        Set::Scalar target_T = 300.0;    ///< Target temperature [K]
        Set::Scalar target_p = 101325.0; ///< Target pressure [Pa]

        // Relaxation coefficients (0.2-0.5 typical)
        Set::Scalar relax_u = 0.3; ///< Normal velocity relaxation
        Set::Scalar relax_v = 0.3; ///< Tangential velocity relaxation
        Set::Scalar relax_w = 0.3; ///< Second tangential velocity relaxation (3D)
        Set::Scalar relax_T = 0.3; ///< Temperature relaxation

        // Transverse term weight (0-1, typically 0.2-0.6)
        Set::Scalar beta = 0.6; ///< Transverse term coefficient

        // Outflow parameters
        Set::Scalar sigma = 0.25; ///< Outflow pressure relaxation coefficient (Motheau Eq. 38)

        // Reference length scale
        Set::Scalar L_ref = 1.0; ///< Reference length [m] (domain size)
    };

private:
    // Boundary parameters for each face
    BoundaryParams params_xlo, params_xhi;
    BoundaryParams params_ylo, params_yhi;
    BoundaryParams params_zlo, params_zhi;

    // Small number for regularization
    Set::Scalar small = 1.0e-10;

public:
    NSCBC() = default;

    /// Constructor with input file parsing
    NSCBC(IO::ParmParse &pp, std::string prefix = "nscbc");

    /// Parse boundary parameters from input file
    static void Parse(NSCBC &value, IO::ParmParse &pp);

    /// Fill ghost cells for all conservative variables
    ///
    /// ORDER OF OPERATIONS (CRITICAL - Motheau Figure 2):
    ///   1. Fill ALL 2 CORNERS first (both 2 ghost cell layers)
    ///   2. Fill EDGES second (skip corners since already done)
    ///
    /// @param density_mf    Density field [kg/m^3]
    /// @param momentum_mf   Momentum field [kg*m/s]
    /// @param energy_mf     Total energy field [J/m^3]
    /// @param eta_mf        Volume fraction field (for mixed EOS)
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
                      const EOS0 &eos0,
                      const EOS1 &eos1,
                      const amrex::Geometry &geom,
                      Set::Scalar time,
                      Set::Scalar pref)
    {
        const amrex::Box &domain = geom.Domain();
        const auto dx = geom.CellSizeArray();

        // =========================================================================
        // CAPTURE EOS PARAMETERS FOR GPU
        // =========================================================================
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
            Set::Scalar KE = 0.5 * rho_val * (vx * vx + vy * vy + vz * vz);
#else
            Set::Scalar KE = 0.5 * rho_val * (vx * vx + vy * vy);
#endif

            // Internal energy
            Set::Scalar UE = E_val - KE;
            if (!(UE > 0.0))
                UE = 0.0;

            // Mixed EOS properties
            Set::Scalar A = eta_val / (g0_ - 1.0) + (1.0 - eta_val) / (g1_ - 1.0);
            Set::Scalar B = (eta_val * g0_ * pi0_) / (g0_ - 1.0)
                            + ((1.0 - eta_val) * g1_ * pi1_) / (g1_ - 1.0);
            Set::Scalar gamma_mix = 1.0 + 1.0 / A;
            Set::Scalar pi_mix = (B / A) / gamma_mix;

            // Pressure
            Set::Scalar p_val = (gamma_mix - 1.0) * UE - gamma_mix * pi_mix + pref_local;
            if (!(p_val > 0.0))
                p_val = 1.0e-6;

            // Sound speed
            Set::Scalar a_val = std::sqrt(gamma_mix * (p_val + pi_mix) / rho_val);

            // Output based on direction
            rho_out = rho_val;
            if (dir == 0)
            {
                un_out = vx;
                ut_out = vy;
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
        // =========================================================================
        auto set_ghost_cons = [=] AMREX_GPU_HOST_DEVICE(
                                  const amrex::Array4<const amrex::Real> &eta_arr,
                                  const amrex::Array4<amrex::Real> &rho_arr,
                                  const amrex::Array4<amrex::Real> &M_arr,
                                  const amrex::Array4<amrex::Real> &E_arr,
                                  int ii,
                                  int jj,
                                  int kk,
                                  int dir,
                                  Set::Scalar rho_val,
                                  Set::Scalar un_val,
                                  Set::Scalar ut_val,
                                  Set::Scalar p_val) -> void {
            Set::Scalar eta_val = eta_arr(ii, jj, kk);

            // Mixed EOS
            Set::Scalar A = eta_val / (g0_ - 1.0) + (1.0 - eta_val) / (g1_ - 1.0);
            Set::Scalar B = (eta_val * g0_ * pi0_) / (g0_ - 1.0)
                            + ((1.0 - eta_val) * g1_ * pi1_) / (g1_ - 1.0);
            Set::Scalar gamma_mix = 1.0 + 1.0 / A;
            Set::Scalar pi_mix = (B / A) / gamma_mix;

            // Inverse EOS
            Set::Scalar UE = (p_val + gamma_mix * pi_mix - pref_local) / (gamma_mix - 1.0);
            if (UE < 0.0)
                UE = 0.0;

            // Velocities
            Set::Scalar vx, vy, vz;
            if (dir == 0)
            {
                vx = un_val;
                vy = ut_val;
                vz = 0.0;
            }
            else if (dir == 1)
            {
                vx = ut_val;
                vy = un_val;
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
        };

        // =========================================================================
        // MAIN LOOP OVER MULTIFAB BOXES
        // =========================================================================
        for (amrex::MFIter mfi(density_mf); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();

            // Get Array4 views
            amrex::Array4<const amrex::Real> const &eta_arr = eta_mf.const_array(mfi);
            amrex::Array4<amrex::Real> const &rho_arr = density_mf.array(mfi);
            amrex::Array4<amrex::Real> const &M_arr = momentum_mf.array(mfi);
            amrex::Array4<amrex::Real> const &E_arr = energy_mf.array(mfi);

            // =====================================================================
            // STEP 1: FILL ALL 4 CORNERS FIRST (Motheau Figure 2)
            // MODIFIED: Only 2 ghost cell layers
            // =====================================================================
            for (int cx = 0; cx < 2; ++cx)
                for (int cy = 0; cy < 2; ++cy)
                {
                    BoundaryParams params_x = (cx == 0) ? params_xlo_local : params_xhi_local;
                    BoundaryParams params_y = (cy == 0) ? params_ylo_local : params_yhi_local;

                    if (params_x.type == Type::None || params_y.type == Type::None)
                        continue;

                    int ibdry_x = (cx == 0) ? domain.smallEnd(0) : domain.bigEnd(0);
                    int jbdry_y = (cy == 0) ? domain.smallEnd(1) : domain.bigEnd(1);

                    bool touch_x = (cx == 0) ? (bx.smallEnd(0) == domain.smallEnd(0))
                                             : (bx.bigEnd(0) == domain.bigEnd(0));
                    bool touch_y = (cy == 0) ? (bx.smallEnd(1) == domain.smallEnd(1))
                                             : (bx.bigEnd(1) == domain.bigEnd(1));

                    if (!touch_x || !touch_y)
                        continue;

                    // Build corner ghost box - ONLY 2 LAYERS
                    int ng = 2;
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

                    // Capture parameters
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
                    int sx_local = sx;
                    int sy_local = sy;
                    int ibdry_x_local = ibdry_x;
                    int jbdry_y_local = jbdry_y;

                    amrex::ParallelFor(corner_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        int ic = ibdry_x_local;
                        int jc = jbdry_y_local;

                        int ox = std::abs(i - ic);
                        int oy = std::abs(j - jc);

                        // Compute primitives at corner
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

                        // X-direction derivatives
                        Set::Scalar rho_xm1, vx_xm1, vy_xm1, p_xm1, a_xm1, gm_xm1, pi_xm1;
                        Set::Scalar rho_xm2, vx_xm2, vy_xm2, p_xm2, a_xm2, gm_xm2, pi_xm2;
                        {
                            int im1 = ic - sx_local;
                            int im2 = ic - 2 * sx_local;
                            Set::Scalar un_tmp, ut_tmp;
                            compute_prim(eta_arr, rho_arr, M_arr, E_arr, im1, jc, k, 0, rho_xm1, un_tmp, ut_tmp, p_xm1, a_xm1, gm_xm1, pi_xm1);
                            vx_xm1 = un_tmp;
                            vy_xm1 = ut_tmp;
                            compute_prim(eta_arr, rho_arr, M_arr, E_arr, im2, jc, k, 0, rho_xm2, un_tmp, ut_tmp, p_xm2, a_xm2, gm_xm2, pi_xm2);
                            vx_xm2 = un_tmp;
                            vy_xm2 = ut_tmp;
                        }

                        Set::Scalar fdx = (Set::Scalar)sx_local;
                        Set::Scalar drho_dx = fdx * (3.0 * rho_c - 4.0 * rho_xm1 + rho_xm2) / (2.0 * dx[0]);
                        Set::Scalar dvx_dx = fdx * (3.0 * vx_c - 4.0 * vx_xm1 + vx_xm2) / (2.0 * dx[0]);
                        Set::Scalar dvy_dx = fdx * (3.0 * vy_c - 4.0 * vy_xm1 + vy_xm2) / (2.0 * dx[0]);
                        Set::Scalar dp_dx = fdx * (3.0 * p_c - 4.0 * p_xm1 + p_xm2) / (2.0 * dx[0]);

                        // Y-direction derivatives
                        Set::Scalar rho_ym1, vx_ym1, vy_ym1, p_ym1, a_ym1, gm_ym1, pi_ym1;
                        Set::Scalar rho_ym2, vx_ym2, vy_ym2, p_ym2, a_ym2, gm_ym2, pi_ym2;
                        {
                            int jm1 = jc - sy_local;
                            int jm2 = jc - 2 * sy_local;
                            Set::Scalar un_tmp, ut_tmp;
                            compute_prim(eta_arr, rho_arr, M_arr, E_arr, ic, jm1, k, 1, rho_ym1, un_tmp, ut_tmp, p_ym1, a_ym1, gm_ym1, pi_ym1);
                            vy_ym1 = un_tmp;
                            vx_ym1 = ut_tmp;
                            compute_prim(eta_arr, rho_arr, M_arr, E_arr, ic, jm2, k, 1, rho_ym2, un_tmp, ut_tmp, p_ym2, a_ym2, gm_ym2, pi_ym2);
                            vy_ym2 = un_tmp;
                            vx_ym2 = ut_tmp;
                        }

                        Set::Scalar fdy = (Set::Scalar)sy_local;
                        Set::Scalar drho_dy = fdy * (3.0 * rho_c - 4.0 * rho_ym1 + rho_ym2) / (2.0 * dx[1]);
                        Set::Scalar dvy_dy = fdy * (3.0 * vy_c - 4.0 * vy_ym1 + vy_ym2) / (2.0 * dx[1]);
                        Set::Scalar dvx_dy = fdy * (3.0 * vx_c - 4.0 * vx_ym1 + vx_ym2) / (2.0 * dx[1]);
                        Set::Scalar dp_dy = fdy * (3.0 * p_c - 4.0 * p_ym1 + p_ym2) / (2.0 * dx[1]);

                        // LODI waves
                        Set::Scalar L1_x = (vx_c - a_c) * (dp_dx - rho_c * a_c * dvx_dx);
                        Set::Scalar L2_x = (std::abs(vx_c) > 1.0e-14) ? vx_c * (a2_c * drho_dx - dp_dx) : 0.0;
                        Set::Scalar L3_x = (std::abs(vx_c) > 1.0e-14) ? vx_c * dvy_dx : 0.0;
                        Set::Scalar L4_x = (vx_c + a_c) * (dp_dx + rho_c * a_c * dvx_dx);

                        Set::Scalar L1_y = (vy_c - a_c) * (dp_dy - rho_c * a_c * dvy_dy);
                        Set::Scalar L2_y = (std::abs(vy_c) > 1.0e-14) ? vy_c * (a2_c * drho_dy - dp_dy) : 0.0;
                        Set::Scalar L3_y = (std::abs(vy_c) > 1.0e-14) ? vy_c * dvx_dy : 0.0;
                        Set::Scalar L4_y = (vy_c + a_c) * (dp_dy + rho_c * a_c * dvy_dy);

                        // Solve coupled system
                        Set::Scalar Lx_in = (cx == 1) ? L1_x : L4_x;
                        Set::Scalar Ly_in = (cy == 1) ? L1_y : L4_y;

                        Set::Scalar Kx = sigma_x * (1.0 - Mx_c * Mx_c) / (a_c * Lx_ref);
                        Set::Scalar Ky = sigma_y * (1.0 - My_c * My_c) / (a_c * Ly_ref);

                        Set::Scalar ab = (1.0 - 0.5 * (beta_x + beta_y)) / 2.0;

                        if (ftx == Type::Outflow && fty == Type::Outflow)
                        {
                            Set::Scalar bx = Kx * (p_c - px_t);
                            Set::Scalar by = Ky * (p_c - py_t);
                            Set::Scalar det_inv = 1.0 / (1.0 - ab * ab + 1.0e-20);
                            Lx_in = det_inv * (bx - ab * by);
                            Ly_in = det_inv * (by - ab * bx);
                        }
                        else if (ftx == Type::Outflow && fty == Type::Inflow)
                        {
                            Ly_in = 0.0;
                            Lx_in = Kx * (p_c - px_t);
                        }
                        else if (ftx == Type::Inflow && fty == Type::Outflow)
                        {
                            Lx_in = 0.0;
                            Ly_in = Ky * (p_c - py_t);
                        }
                        else if (ftx == Type::Inflow && fty == Type::Inflow)
                        {
                            Lx_in = 0.0;
                            Ly_in = 0.0;
                        }

                        if (cx == 1)
                            L1_x = Lx_in;
                        else
                            L4_x = Lx_in;
                        if (cy == 1)
                            L1_y = Ly_in;
                        else
                            L4_y = Ly_in;

                        // Reconstruct gradients
                        Set::Scalar xi1_x = L1_x / (vx_c - a_c + small_local);
                        Set::Scalar xi2_x = (std::abs(vx_c) > 1.0e-14) ? L2_x / vx_c : 0.0;
                        Set::Scalar xi3_x = (std::abs(vx_c) > 1.0e-14) ? L3_x / vx_c : 0.0;
                        Set::Scalar xi4_x = L4_x / (vx_c + a_c + small_local);

                        Set::Scalar drho_dx_new = xi1_x + xi2_x + xi4_x;
                        Set::Scalar dvx_dx_new = (-a_c / (rho_c + small_local)) * xi1_x + (a_c / (rho_c + small_local)) * xi4_x;
                        Set::Scalar dvy_dx_new = xi3_x;
                        Set::Scalar dp_dx_new = a2_c * (xi1_x + xi4_x);

                        Set::Scalar xi1_y = L1_y / (vy_c - a_c + small_local);
                        Set::Scalar xi2_y = (std::abs(vy_c) > 1.0e-14) ? L2_y / vy_c : 0.0;
                        Set::Scalar xi3_y = (std::abs(vy_c) > 1.0e-14) ? L3_y / vy_c : 0.0;
                        Set::Scalar xi4_y = L4_y / (vy_c + a_c + small_local);

                        Set::Scalar drho_dy_new = xi1_y + xi2_y + xi4_y;
                        Set::Scalar dvy_dy_new = (-a_c / (rho_c + small_local)) * xi1_y + (a_c / (rho_c + small_local)) * xi4_y;
                        Set::Scalar dvx_dy_new = xi3_y;
                        Set::Scalar dp_dy_new = a2_c * (xi1_y + xi4_y);

                        // Fill ghost cells by layer - ONLY 2 LAYERS
                        Set::Scalar rho_g, vx_g, vy_g, p_g;

                        int layer = std::max(ox, oy);

                        if (layer == 1)
                        {
                            // Layer 1: Motheau Eq. 27/31 - Linear extrapolation
                            Set::Scalar dx_off = (Set::Scalar)(i - ic) * dx[0];
                            Set::Scalar dy_off = (Set::Scalar)(j - jc) * dx[1];

                            rho_g = rho_c + dx_off * drho_dx_new + dy_off * drho_dy_new;
                            vx_g = vx_c + dx_off * dvx_dx_new + dy_off * dvx_dy_new;
                            vy_g = vy_c + dx_off * dvy_dx_new + dy_off * dvy_dy_new;
                            p_g = p_c + dx_off * dp_dx_new + dy_off * dp_dy_new;
                        }
                        else // layer == 2
                        {
                            // Layer 2: Motheau Eq. 28/32 - Quadratic extrapolation
                            int i1 = ic + sx_local * 1;
                            int j1 = jc + sy_local * 1;

                            Set::Scalar rho_1 = rho_arr(i1, j1, k);
                            Set::Scalar vx_1 = M_arr(i1, j1, k, 0) / (rho_1 + small_local);
                            Set::Scalar vy_1 = M_arr(i1, j1, k, 1) / (rho_1 + small_local);

                            Set::Scalar eta_1 = eta_arr(i1, j1, k);
                            Set::Scalar A_1 = eta_1 / (g0_ - 1.0) + (1.0 - eta_1) / (g1_ - 1.0);
                            Set::Scalar gm_1 = 1.0 + 1.0 / A_1;
                            Set::Scalar KE_1 = 0.5 * rho_1 * (vx_1 * vx_1 + vy_1 * vy_1);
                            Set::Scalar UE_1 = E_arr(i1, j1, k) - KE_1;
                            Set::Scalar B_1 = (eta_1 * g0_ * pi0_) / (g0_ - 1.0) + ((1.0 - eta_1) * g1_ * pi1_) / (g1_ - 1.0);
                            Set::Scalar pi_1 = (B_1 / A_1) / gm_1;
                            Set::Scalar p_1 = (gm_1 - 1.0) * UE_1 - gm_1 * pi_1 + pref_local;

                            Set::Scalar dx_grad = (ox == 2) ? 6.0 * dx[0] * drho_dx_new : 0.0;
                            Set::Scalar dy_grad = (oy == 2) ? 6.0 * dx[1] * drho_dy_new : 0.0;
                            rho_g = -2.0 * rho_xm1 - 3.0 * rho_c + 6.0 * rho_1 + dx_grad + dy_grad;

                            dx_grad = (ox == 2) ? 6.0 * dx[0] * dvx_dx_new : 0.0;
                            dy_grad = (oy == 2) ? 6.0 * dx[1] * dvx_dy_new : 0.0;
                            vx_g = -2.0 * vx_xm1 - 3.0 * vx_c + 6.0 * vx_1 + dx_grad + dy_grad;

                            dx_grad = (ox == 2) ? 6.0 * dx[0] * dvy_dx_new : 0.0;
                            dy_grad = (oy == 2) ? 6.0 * dx[1] * dvy_dy_new : 0.0;
                            vy_g = -2.0 * vy_xm1 - 3.0 * vy_c + 6.0 * vy_1 + dx_grad + dy_grad;

                            dx_grad = (ox == 2) ? 6.0 * dx[0] * dp_dx_new : 0.0;
                            dy_grad = (oy == 2) ? 6.0 * dx[1] * dp_dy_new : 0.0;
                            p_g = -2.0 * p_xm1 - 3.0 * p_c + 6.0 * p_1 + dx_grad + dy_grad;
                        }

                        // Enforce physical bounds
                        rho_g = amrex::min(amrex::max(rho_g, 1.0e-8), 10000.0);
                        p_g = amrex::min(amrex::max(p_g, 1.0), 1.0e8);
                        vx_g = amrex::min(amrex::max(vx_g, -1000.0), 1000.0);
                        vy_g = amrex::min(amrex::max(vy_g, -1000.0), 1000.0);

                        // Write to conserved arrays
                        Set::Scalar eta_g = eta_arr(i, j, k);
                        Set::Scalar A = eta_g / (g0_ - 1.0) + (1.0 - eta_g) / (g1_ - 1.0);
                        Set::Scalar B = (eta_g * g0_ * pi0_) / (g0_ - 1.0)
                                        + ((1.0 - eta_g) * g1_ * pi1_) / (g1_ - 1.0);
                        Set::Scalar gm_g = 1.0 + 1.0 / A;
                        Set::Scalar pi_g = (B / A) / gm_g;

                        Set::Scalar UE_g = (p_g + gm_g * pi_g - pref_local) / (gm_g - 1.0);
                        if (UE_g < 0.0)
                            UE_g = 0.0;

                        Set::Scalar KE_g = 0.5 * rho_g * (vx_g * vx_g + vy_g * vy_g);

                        rho_arr(i, j, k) = rho_g;
                        M_arr(i, j, k, 0) = rho_g * vx_g;
                        M_arr(i, j, k, 1) = rho_g * vy_g;
                        E_arr(i, j, k) = UE_g + KE_g;
                    }); // end ParallelFor corner_box

                } // end corner loops

            // =====================================================================
            // STEP 2: FILL EDGES (SKIP CORNERS - ALREADY FILLED)
            // MODIFIED: Only 2 ghost cell layers
            // =====================================================================

            // =====================================================================
            // X-LO BOUNDARY (LEFT EDGE - SKIP CORNERS)
            // =====================================================================
            if (bx.smallEnd(0) == domain.smallEnd(0) && params_xlo_local.type != Type::None)
            {
                int idir = 0;
                Set::Scalar delta = dx[0];
                int i_bnd = domain.smallEnd(0);

                int j_start = domain.smallEnd(1) + 1;
                int j_end = domain.bigEnd(1) - 1;

                for (int jj = j_start; jj <= j_end; ++jj)
                {
                    if (jj < bx.smallEnd(1) || jj > bx.bigEnd(1))
                        continue;

                    amrex::Box ghost_box;
                    ghost_box.setSmall(0, i_bnd - 2);
                    ghost_box.setBig(0, i_bnd - 1);
                    ghost_box.setSmall(1, jj);
                    ghost_box.setBig(1, jj);
#if AMREX_SPACEDIM == 3
                    ghost_box.setSmall(2, bx.smallEnd(2));
                    ghost_box.setBig(2, bx.bigEnd(2));
#endif

                    Set::Scalar px_t = params_xlo_local.target_p;
                    Set::Scalar sigma_x = params_xlo_local.sigma;
                    Set::Scalar beta_x = params_xlo_local.beta;
                    Set::Scalar Lx_ref = params_xlo_local.L_ref;
                    Type ftx = params_xlo_local.type;
                    Set::Scalar relax_u = params_xlo_local.relax_u;
                    Set::Scalar target_u = params_xlo_local.target_u;
                    Set::Scalar target_v = params_xlo_local.target_v;
                    Set::Scalar target_T = params_xlo_local.target_T;

                    amrex::ParallelFor(ghost_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        Set::Scalar rho_bnd, un_bnd, ut_bnd, p_bnd, a_bnd, gm_bnd, pi_bnd;
                        compute_prim(eta_arr, rho_arr, M_arr, E_arr, i_bnd, j, k, idir, rho_bnd, un_bnd, ut_bnd, p_bnd, a_bnd, gm_bnd, pi_bnd);

                        Set::Scalar rho_m1, un_m1, ut_m1, p_m1, a_m1, gm_m1, pi_m1;
                        compute_prim(eta_arr, rho_arr, M_arr, E_arr, i_bnd + 1, j, k, idir, rho_m1, un_m1, ut_m1, p_m1, a_m1, gm_m1, pi_m1);

                        Set::Scalar rho_m2, un_m2, ut_m2, p_m2, a_m2, gm_m2, pi_m2;
                        compute_prim(eta_arr, rho_arr, M_arr, E_arr, i_bnd + 2, j, k, idir, rho_m2, un_m2, ut_m2, p_m2, a_m2, gm_m2, pi_m2);

                        Set::Scalar drho_dx = (-3.0 * rho_bnd + 4.0 * rho_m1 - rho_m2) / (2.0 * delta);
                        Set::Scalar dun_dx = (-3.0 * un_bnd + 4.0 * un_m1 - un_m2) / (2.0 * delta);
                        Set::Scalar dut_dx = (-3.0 * ut_bnd + 4.0 * ut_m1 - ut_m2) / (2.0 * delta);
                        Set::Scalar dp_dx = (-3.0 * p_bnd + 4.0 * p_m1 - p_m2) / (2.0 * delta);

                        Set::Scalar drho_dy = 0.0, dun_dy = 0.0, dut_dy = 0.0, dp_dy = 0.0;
                        if (j > domain.smallEnd(1) && j < domain.bigEnd(1))
                        {
                            Set::Scalar rho_jp, un_jp, ut_jp, p_jp, a_jp, gm_jp, pi_jp;
                            compute_prim(eta_arr, rho_arr, M_arr, E_arr, i_bnd, j + 1, k, idir, rho_jp, un_jp, ut_jp, p_jp, a_jp, gm_jp, pi_jp);

                            Set::Scalar rho_jm, un_jm, ut_jm, p_jm, a_jm, gm_jm, pi_jm;
                            compute_prim(eta_arr, rho_arr, M_arr, E_arr, i_bnd, j - 1, k, idir, rho_jm, un_jm, ut_jm, p_jm, a_jm, gm_jm, pi_jm);

                            drho_dy = (rho_jp - rho_jm) / (2.0 * dx[1]);
                            dun_dy = (un_jp - un_jm) / (2.0 * dx[1]);
                            dut_dy = (ut_jp - ut_jm) / (2.0 * dx[1]);
                            dp_dy = (p_jp - p_jm) / (2.0 * dx[1]);
                        }

                        Set::Scalar a2_bnd = a_bnd * a_bnd;
                        Set::Scalar M_loc = std::abs(un_bnd) / (a_bnd + small_local);

                        Set::Scalar L1 = (un_bnd - a_bnd) * (dp_dx - rho_bnd * a_bnd * dun_dx);
                        Set::Scalar L2 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd * (a2_bnd * drho_dx - dp_dx) : 0.0;
                        Set::Scalar L3 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd * dut_dx : 0.0;
                        Set::Scalar L4 = (un_bnd + a_bnd) * (dp_dx + rho_bnd * a_bnd * dun_dx);

                        Set::Scalar T1 = ut_bnd * (dp_dy - rho_bnd * a_bnd * dun_dy) + gm_bnd * p_bnd * dut_dy;
                        Set::Scalar T4 = ut_bnd * (dp_dy + rho_bnd * a_bnd * dun_dy) + gm_bnd * p_bnd * dut_dy;
                        Set::Scalar T2 = (std::abs(un_bnd) > 1.0e-14) ? ut_bnd * (a2_bnd * drho_dy - dp_dy) + (a2_bnd * rho_bnd - gm_bnd * p_bnd) * dut_dy : 0.0;
                        Set::Scalar T3 = (std::abs(un_bnd) > 1.0e-14) ? ut_bnd * dut_dy + (1.0 / rho_bnd) * dp_dy : 0.0;

                        Set::Scalar K = sigma_x * (1.0 - M_loc * M_loc) * a_bnd / Lx_ref;

                        if (ftx == Type::Outflow)
                        {
                            L4 = K * (p_bnd - px_t) - (1.0 - beta_x) * T4;
                        }
                        else if (ftx == Type::Inflow)
                        {
                            Set::Scalar K_acoustic = relax_u * rho_bnd * (1.0 - M_loc * M_loc) / Lx_ref;
                            Set::Scalar K_entropy = relax_u * rho_bnd * a_bnd * (gm_bnd - 1.0) / Lx_ref;
                            Set::Scalar K_shear = relax_u * a_bnd / Lx_ref;

                            Set::Scalar eta_bnd = eta_arr(i_bnd, j, k);
                            Set::Scalar cv_mix = eta_bnd * cv0_ + (1.0 - eta_bnd) * cv1_;
                            Set::Scalar T_bnd = (p_bnd + pi_bnd) / (rho_bnd * cv_mix * (gm_bnd - 1.0) + small_local);

                            L4 = K_acoustic * (un_bnd - target_u) - T4;
                            L2 = K_entropy * (T_bnd - target_T) - T2;
                            L3 = K_shear * (ut_bnd - target_v) - T3;
                        }

                        Set::Scalar xi1 = L1 / (un_bnd - a_bnd + small_local);
                        Set::Scalar xi2 = (std::abs(un_bnd) > 1.0e-14) ? L2 / un_bnd : 0.0;
                        Set::Scalar xi3 = (std::abs(un_bnd) > 1.0e-14) ? L3 / un_bnd : 0.0;
                        Set::Scalar xi4 = L4 / (un_bnd + a_bnd + small_local);

                        Set::Scalar drho_dx_new = xi1 + xi2 + xi4;
                        Set::Scalar dun_dx_new = (-a_bnd / (rho_bnd + small_local)) * xi1 + (a_bnd / (rho_bnd + small_local)) * xi4;
                        Set::Scalar dut_dx_new = xi3;
                        Set::Scalar dp_dx_new = a2_bnd * (xi1 + xi4);

                        // Layer 1: Motheau Eq. 27
                        Set::Scalar rho_g1 = rho_m1 - 2.0 * delta * drho_dx_new;
                        Set::Scalar un_g1 = un_m1 - 2.0 * delta * dun_dx_new;
                        Set::Scalar ut_g1 = ut_m1 - 2.0 * delta * dut_dx_new;
                        Set::Scalar p_g1 = p_m1 - 2.0 * delta * dp_dx_new;

                        rho_g1 = amrex::max(rho_g1, small_local);
                        p_g1 = amrex::max(p_g1, 1.0e-6);

                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, i_bnd - 1, j, k, idir, rho_g1, un_g1, ut_g1, p_g1);

                        // Layer 2: Motheau Eq. 28
                        Set::Scalar rho_g2 = -2.0 * rho_m1 - 3.0 * rho_bnd + 6.0 * rho_g1 + 6.0 * delta * drho_dx_new;
                        Set::Scalar un_g2 = -2.0 * un_m1 - 3.0 * un_bnd + 6.0 * un_g1 + 6.0 * delta * dun_dx_new;
                        Set::Scalar ut_g2 = -2.0 * ut_m1 - 3.0 * ut_bnd + 6.0 * ut_g1 + 6.0 * delta * dut_dx_new;
                        Set::Scalar p_g2 = -2.0 * p_m1 - 3.0 * p_bnd + 6.0 * p_g1 + 6.0 * delta * dp_dx_new;

                        rho_g2 = amrex::max(rho_g2, small_local);
                        p_g2 = amrex::max(p_g2, 1.0e-6);

                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, i_bnd - 2, j, k, idir, rho_g2, un_g2, ut_g2, p_g2);
                    });
                }
            } // end X-LO boundary

            // =====================================================================
            // X-HI BOUNDARY (RIGHT EDGE - SKIP CORNERS)
            // =====================================================================
            if (bx.bigEnd(0) == domain.bigEnd(0) && params_xhi_local.type != Type::None)
            {
                int idir = 0;
                Set::Scalar delta = dx[0];
                int i_bnd = domain.bigEnd(0);

                int j_start = domain.smallEnd(1) + 1;
                int j_end = domain.bigEnd(1) - 1;

                for (int jj = j_start; jj <= j_end; ++jj)
                {
                    if (jj < bx.smallEnd(1) || jj > bx.bigEnd(1))
                        continue;

                    amrex::Box ghost_box;
                    ghost_box.setSmall(0, i_bnd + 1);
                    ghost_box.setBig(0, i_bnd + 2);
                    ghost_box.setSmall(1, jj);
                    ghost_box.setBig(1, jj);
#if AMREX_SPACEDIM == 3
                    ghost_box.setSmall(2, bx.smallEnd(2));
                    ghost_box.setBig(2, bx.bigEnd(2));
#endif

                    Set::Scalar px_t = params_xhi_local.target_p;
                    Set::Scalar sigma_x = params_xhi_local.sigma;
                    Set::Scalar beta_x = params_xhi_local.beta;
                    Set::Scalar Lx_ref = params_xhi_local.L_ref;
                    Type ftx = params_xhi_local.type;
                    Set::Scalar relax_u = params_xhi_local.relax_u;
                    Set::Scalar target_u = params_xhi_local.target_u;
                    Set::Scalar target_v = params_xhi_local.target_v;
                    Set::Scalar target_T = params_xhi_local.target_T;

                    amrex::ParallelFor(ghost_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        Set::Scalar rho_bnd, un_bnd, ut_bnd, p_bnd, a_bnd, gm_bnd, pi_bnd;
                        compute_prim(eta_arr, rho_arr, M_arr, E_arr, i_bnd, j, k, idir, rho_bnd, un_bnd, ut_bnd, p_bnd, a_bnd, gm_bnd, pi_bnd);

                        Set::Scalar rho_m1, un_m1, ut_m1, p_m1, a_m1, gm_m1, pi_m1;
                        compute_prim(eta_arr, rho_arr, M_arr, E_arr, i_bnd - 1, j, k, idir, rho_m1, un_m1, ut_m1, p_m1, a_m1, gm_m1, pi_m1);

                        Set::Scalar rho_m2, un_m2, ut_m2, p_m2, a_m2, gm_m2, pi_m2;
                        compute_prim(eta_arr, rho_arr, M_arr, E_arr, i_bnd - 2, j, k, idir, rho_m2, un_m2, ut_m2, p_m2, a_m2, gm_m2, pi_m2);

                        Set::Scalar drho_dx = (3.0 * rho_bnd - 4.0 * rho_m1 + rho_m2) / (2.0 * delta);
                        Set::Scalar dun_dx = (3.0 * un_bnd - 4.0 * un_m1 + un_m2) / (2.0 * delta);
                        Set::Scalar dut_dx = (3.0 * ut_bnd - 4.0 * ut_m1 + ut_m2) / (2.0 * delta);
                        Set::Scalar dp_dx = (3.0 * p_bnd - 4.0 * p_m1 + p_m2) / (2.0 * delta);

                        Set::Scalar drho_dy = 0.0, dun_dy = 0.0, dut_dy = 0.0, dp_dy = 0.0;
                        if (j > domain.smallEnd(1) && j < domain.bigEnd(1))
                        {
                            Set::Scalar rho_jp, un_jp, ut_jp, p_jp, a_jp, gm_jp, pi_jp;
                            compute_prim(eta_arr, rho_arr, M_arr, E_arr, i_bnd, j + 1, k, idir, rho_jp, un_jp, ut_jp, p_jp, a_jp, gm_jp, pi_jp);

                            Set::Scalar rho_jm, un_jm, ut_jm, p_jm, a_jm, gm_jm, pi_jm;
                            compute_prim(eta_arr, rho_arr, M_arr, E_arr, i_bnd, j - 1, k, idir, rho_jm, un_jm, ut_jm, p_jm, a_jm, gm_jm, pi_jm);

                            drho_dy = (rho_jp - rho_jm) / (2.0 * dx[1]);
                            dun_dy = (un_jp - un_jm) / (2.0 * dx[1]);
                            dut_dy = (ut_jp - ut_jm) / (2.0 * dx[1]);
                            dp_dy = (p_jp - p_jm) / (2.0 * dx[1]);
                        }

                        Set::Scalar a2_bnd = a_bnd * a_bnd;
                        Set::Scalar M_loc = std::abs(un_bnd) / (a_bnd + small_local);

                        Set::Scalar L1 = (un_bnd - a_bnd) * (dp_dx - rho_bnd * a_bnd * dun_dx);
                        Set::Scalar L2 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd * (a2_bnd * drho_dx - dp_dx) : 0.0;
                        Set::Scalar L3 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd * dut_dx : 0.0;
                        Set::Scalar L4 = (un_bnd + a_bnd) * (dp_dx + rho_bnd * a_bnd * dun_dx);

                        Set::Scalar T1 = ut_bnd * (dp_dy - rho_bnd * a_bnd * dun_dy) + gm_bnd * p_bnd * dut_dy;
                        Set::Scalar T4 = ut_bnd * (dp_dy + rho_bnd * a_bnd * dun_dy) + gm_bnd * p_bnd * dut_dy;
                        Set::Scalar T2 = (std::abs(un_bnd) > 1.0e-14) ? ut_bnd * (a2_bnd * drho_dy - dp_dy) + (a2_bnd * rho_bnd - gm_bnd * p_bnd) * dut_dy : 0.0;
                        Set::Scalar T3 = (std::abs(un_bnd) > 1.0e-14) ? ut_bnd * dut_dy + (1.0 / rho_bnd) * dp_dy : 0.0;

                        Set::Scalar K = sigma_x * (1.0 - M_loc * M_loc) * a_bnd / Lx_ref;

                        if (ftx == Type::Outflow)
                        {
                            L1 = K * (p_bnd - px_t) - (1.0 - beta_x) * T1;
                        }
                        else if (ftx == Type::Inflow)
                        {
                            Set::Scalar K_acoustic = relax_u * rho_bnd * (1.0 - M_loc * M_loc) / Lx_ref;
                            Set::Scalar K_entropy = relax_u * rho_bnd * a_bnd * (gm_bnd - 1.0) / Lx_ref;
                            Set::Scalar K_shear = relax_u * a_bnd / Lx_ref;

                            Set::Scalar eta_bnd = eta_arr(i_bnd, j, k);
                            Set::Scalar cv_mix = eta_bnd * cv0_ + (1.0 - eta_bnd) * cv1_;
                            Set::Scalar T_bnd = (p_bnd + pi_bnd) / (rho_bnd * cv_mix * (gm_bnd - 1.0) + small_local);

                            L1 = -K_acoustic * (un_bnd - target_u) - T1;
                            L2 = K_entropy * (T_bnd - target_T) - T2;
                            L3 = K_shear * (ut_bnd - target_v) - T3;
                        }

                        Set::Scalar xi1 = L1 / (un_bnd - a_bnd + small_local);
                        Set::Scalar xi2 = (std::abs(un_bnd) > 1.0e-14) ? L2 / un_bnd : 0.0;
                        Set::Scalar xi3 = (std::abs(un_bnd) > 1.0e-14) ? L3 / un_bnd : 0.0;
                        Set::Scalar xi4 = L4 / (un_bnd + a_bnd + small_local);

                        Set::Scalar drho_dx_new = xi1 + xi2 + xi4;
                        Set::Scalar dun_dx_new = (-a_bnd / (rho_bnd + small_local)) * xi1 + (a_bnd / (rho_bnd + small_local)) * xi4;
                        Set::Scalar dut_dx_new = xi3;
                        Set::Scalar dp_dx_new = a2_bnd * (xi1 + xi4);

                        // Layer 1: Motheau Eq. 31
                        Set::Scalar rho_g1 = rho_m1 + 2.0 * delta * drho_dx_new;
                        Set::Scalar un_g1 = un_m1 + 2.0 * delta * dun_dx_new;
                        Set::Scalar ut_g1 = ut_m1 + 2.0 * delta * dut_dx_new;
                        Set::Scalar p_g1 = p_m1 + 2.0 * delta * dp_dx_new;

                        rho_g1 = amrex::max(rho_g1, small_local);
                        p_g1 = amrex::max(p_g1, 1.0e-6);

                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, i_bnd + 1, j, k, idir, rho_g1, un_g1, ut_g1, p_g1);

                        // Layer 2: Motheau Eq. 32
                        Set::Scalar rho_g2 = -2.0 * rho_m1 - 3.0 * rho_bnd + 6.0 * rho_g1 - 6.0 * delta * drho_dx_new;
                        Set::Scalar un_g2 = -2.0 * un_m1 - 3.0 * un_bnd + 6.0 * un_g1 - 6.0 * delta * dun_dx_new;
                        Set::Scalar ut_g2 = -2.0 * ut_m1 - 3.0 * ut_bnd + 6.0 * ut_g1 - 6.0 * delta * dut_dx_new;
                        Set::Scalar p_g2 = -2.0 * p_m1 - 3.0 * p_bnd + 6.0 * p_g1 - 6.0 * delta * dp_dx_new;

                        rho_g2 = amrex::max(rho_g2, small_local);
                        p_g2 = amrex::max(p_g2, 1.0e-6);

                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, i_bnd + 2, j, k, idir, rho_g2, un_g2, ut_g2, p_g2);
                    });
                }
            } // end X-HI boundary

            // =====================================================================
            // Y-LO BOUNDARY (BOTTOM EDGE - SKIP CORNERS)
            // =====================================================================
            if (bx.smallEnd(1) == domain.smallEnd(1) && params_ylo_local.type != Type::None)
            {
                int idir = 1;
                Set::Scalar delta = dx[1];
                int j_bnd = domain.smallEnd(1);

                int i_start = domain.smallEnd(0) + 1;
                int i_end = domain.bigEnd(0) - 1;

                for (int ii = i_start; ii <= i_end; ++ii)
                {
                    if (ii < bx.smallEnd(0) || ii > bx.bigEnd(0))
                        continue;

                    amrex::Box ghost_box;
                    ghost_box.setSmall(0, ii);
                    ghost_box.setBig(0, ii);
                    ghost_box.setSmall(1, j_bnd - 2);
                    ghost_box.setBig(1, j_bnd - 1);
#if AMREX_SPACEDIM == 3
                    ghost_box.setSmall(2, bx.smallEnd(2));
                    ghost_box.setBig(2, bx.bigEnd(2));
#endif

                    Set::Scalar py_t = params_ylo_local.target_p;
                    Set::Scalar sigma_y = params_ylo_local.sigma;
                    Set::Scalar beta_y = params_ylo_local.beta;
                    Set::Scalar Ly_ref = params_ylo_local.L_ref;
                    Type fty = params_ylo_local.type;
                    Set::Scalar relax_u = params_ylo_local.relax_u;
                    Set::Scalar target_v = params_ylo_local.target_v;
                    Set::Scalar target_u = params_ylo_local.target_u;
                    Set::Scalar target_T = params_ylo_local.target_T;

                    amrex::ParallelFor(ghost_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        Set::Scalar rho_bnd, un_bnd, ut_bnd, p_bnd, a_bnd, gm_bnd, pi_bnd;
                        compute_prim(eta_arr, rho_arr, M_arr, E_arr, i, j_bnd, k, idir, rho_bnd, un_bnd, ut_bnd, p_bnd, a_bnd, gm_bnd, pi_bnd);

                        Set::Scalar rho_m1, un_m1, ut_m1, p_m1, a_m1, gm_m1, pi_m1;
                        compute_prim(eta_arr, rho_arr, M_arr, E_arr, i, j_bnd + 1, k, idir, rho_m1, un_m1, ut_m1, p_m1, a_m1, gm_m1, pi_m1);

                        Set::Scalar rho_m2, un_m2, ut_m2, p_m2, a_m2, gm_m2, pi_m2;
                        compute_prim(eta_arr, rho_arr, M_arr, E_arr, i, j_bnd + 2, k, idir, rho_m2, un_m2, ut_m2, p_m2, a_m2, gm_m2, pi_m2);

                        Set::Scalar drho_dy = (-3.0 * rho_bnd + 4.0 * rho_m1 - rho_m2) / (2.0 * delta);
                        Set::Scalar dun_dy = (-3.0 * un_bnd + 4.0 * un_m1 - un_m2) / (2.0 * delta);
                        Set::Scalar dut_dy = (-3.0 * ut_bnd + 4.0 * ut_m1 - ut_m2) / (2.0 * delta);
                        Set::Scalar dp_dy = (-3.0 * p_bnd + 4.0 * p_m1 - p_m2) / (2.0 * delta);

                        Set::Scalar drho_dx = 0.0, dun_dx = 0.0, dut_dx = 0.0, dp_dx = 0.0;
                        if (i > domain.smallEnd(0) && i < domain.bigEnd(0))
                        {
                            Set::Scalar rho_ip, un_ip, ut_ip, p_ip, a_ip, gm_ip, pi_ip;
                            compute_prim(eta_arr, rho_arr, M_arr, E_arr, i + 1, j_bnd, k, idir, rho_ip, un_ip, ut_ip, p_ip, a_ip, gm_ip, pi_ip);

                            Set::Scalar rho_im, un_im, ut_im, p_im, a_im, gm_im, pi_im;
                            compute_prim(eta_arr, rho_arr, M_arr, E_arr, i - 1, j_bnd, k, idir, rho_im, un_im, ut_im, p_im, a_im, gm_im, pi_im);

                            drho_dx = (rho_ip - rho_im) / (2.0 * dx[0]);
                            dun_dx = (un_ip - un_im) / (2.0 * dx[0]);
                            dut_dx = (ut_ip - ut_im) / (2.0 * dx[0]);
                            dp_dx = (p_ip - p_im) / (2.0 * dx[0]);
                        }

                        Set::Scalar a2_bnd = a_bnd * a_bnd;
                        Set::Scalar M_loc = std::abs(un_bnd) / (a_bnd + small_local);

                        Set::Scalar L1 = (un_bnd - a_bnd) * (dp_dy - rho_bnd * a_bnd * dun_dy);
                        Set::Scalar L2 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd * (a2_bnd * drho_dy - dp_dy) : 0.0;
                        Set::Scalar L3 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd * dut_dy : 0.0;
                        Set::Scalar L4 = (un_bnd + a_bnd) * (dp_dy + rho_bnd * a_bnd * dun_dy);

                        Set::Scalar T1 = ut_bnd * (dp_dx - rho_bnd * a_bnd * dun_dx) + gm_bnd * p_bnd * dut_dx;
                        Set::Scalar T4 = ut_bnd * (dp_dx + rho_bnd * a_bnd * dun_dx) + gm_bnd * p_bnd * dut_dx;
                        Set::Scalar T2 = (std::abs(un_bnd) > 1.0e-14) ? ut_bnd * (a2_bnd * drho_dx - dp_dx) + (a2_bnd * rho_bnd - gm_bnd * p_bnd) * dut_dx : 0.0;
                        Set::Scalar T3 = (std::abs(un_bnd) > 1.0e-14) ? ut_bnd * dut_dx + (1.0 / rho_bnd) * dp_dx : 0.0;

                        Set::Scalar K = sigma_y * (1.0 - M_loc * M_loc) * a_bnd / Ly_ref;

                        if (fty == Type::Outflow)
                        {
                            L4 = K * (p_bnd - py_t) - (1.0 - beta_y) * T4;
                        }
                        else if (fty == Type::Inflow)
                        {
                            Set::Scalar K_acoustic = relax_u * rho_bnd * (1.0 - M_loc * M_loc) / Ly_ref;
                            Set::Scalar K_entropy = relax_u * rho_bnd * a_bnd * (gm_bnd - 1.0) / Ly_ref;
                            Set::Scalar K_shear = relax_u * a_bnd / Ly_ref;

                            Set::Scalar eta_bnd = eta_arr(i, j_bnd, k);
                            Set::Scalar cv_mix = eta_bnd * cv0_ + (1.0 - eta_bnd) * cv1_;
                            Set::Scalar T_bnd = (p_bnd + pi_bnd) / (rho_bnd * cv_mix * (gm_bnd - 1.0) + small_local);

                            L4 = K_acoustic * (un_bnd - target_v) - T4;
                            L2 = K_entropy * (T_bnd - target_T) - T2;
                            L3 = K_shear * (ut_bnd - target_u) - T3;
                        }

                        Set::Scalar xi1 = L1 / (un_bnd - a_bnd + small_local);
                        Set::Scalar xi2 = (std::abs(un_bnd) > 1.0e-14) ? L2 / un_bnd : 0.0;
                        Set::Scalar xi3 = (std::abs(un_bnd) > 1.0e-14) ? L3 / un_bnd : 0.0;
                        Set::Scalar xi4 = L4 / (un_bnd + a_bnd + small_local);

                        Set::Scalar drho_dy_new = xi1 + xi2 + xi4;
                        Set::Scalar dun_dy_new = (-a_bnd / (rho_bnd + small_local)) * xi1 + (a_bnd / (rho_bnd + small_local)) * xi4;
                        Set::Scalar dut_dy_new = xi3;
                        Set::Scalar dp_dy_new = a2_bnd * (xi1 + xi4);

                        // Layer 1: Motheau Eq. 27
                        Set::Scalar rho_g1 = rho_m1 - 2.0 * delta * drho_dy_new;
                        Set::Scalar un_g1 = un_m1 - 2.0 * delta * dun_dy_new;
                        Set::Scalar ut_g1 = ut_m1 - 2.0 * delta * dut_dy_new;
                        Set::Scalar p_g1 = p_m1 - 2.0 * delta * dp_dy_new;

                        rho_g1 = amrex::max(rho_g1, small_local);
                        p_g1 = amrex::max(p_g1, 1.0e-6);

                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, i, j_bnd - 1, k, idir, rho_g1, un_g1, ut_g1, p_g1);

                        // Layer 2: Motheau Eq. 28
                        Set::Scalar rho_g2 = -2.0 * rho_m1 - 3.0 * rho_bnd + 6.0 * rho_g1 + 6.0 * delta * drho_dy_new;
                        Set::Scalar un_g2 = -2.0 * un_m1 - 3.0 * un_bnd + 6.0 * un_g1 + 6.0 * delta * dun_dy_new;
                        Set::Scalar ut_g2 = -2.0 * ut_m1 - 3.0 * ut_bnd + 6.0 * ut_g1 + 6.0 * delta * dut_dy_new;
                        Set::Scalar p_g2 = -2.0 * p_m1 - 3.0 * p_bnd + 6.0 * p_g1 + 6.0 * delta * dp_dy_new;

                        rho_g2 = amrex::max(rho_g2, small_local);
                        p_g2 = amrex::max(p_g2, 1.0e-6);

                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, i, j_bnd - 2, k, idir, rho_g2, un_g2, ut_g2, p_g2);
                    });
                }
            } // end Y-LO boundary

            // =====================================================================
            // Y-HI BOUNDARY (TOP EDGE - SKIP CORNERS)
            // =====================================================================
            if (bx.bigEnd(1) == domain.bigEnd(1) && params_yhi_local.type != Type::None)
            {
                int idir = 1;
                Set::Scalar delta = dx[1];
                int j_bnd = domain.bigEnd(1);

                int i_start = domain.smallEnd(0) + 1;
                int i_end = domain.bigEnd(0) - 1;

                for (int ii = i_start; ii <= i_end; ++ii)
                {
                    if (ii < bx.smallEnd(0) || ii > bx.bigEnd(0))
                        continue;

                    amrex::Box ghost_box;
                    ghost_box.setSmall(0, ii);
                    ghost_box.setBig(0, ii);
                    ghost_box.setSmall(1, j_bnd + 1);
                    ghost_box.setBig(1, j_bnd + 2);
#if AMREX_SPACEDIM == 3
                    ghost_box.setSmall(2, bx.smallEnd(2));
                    ghost_box.setBig(2, bx.bigEnd(2));
#endif

                    Set::Scalar py_t = params_yhi_local.target_p;
                    Set::Scalar sigma_y = params_yhi_local.sigma;
                    Set::Scalar beta_y = params_yhi_local.beta;
                    Set::Scalar Ly_ref = params_yhi_local.L_ref;
                    Type fty = params_yhi_local.type;
                    Set::Scalar relax_u = params_yhi_local.relax_u;
                    Set::Scalar target_v = params_yhi_local.target_v;
                    Set::Scalar target_u = params_yhi_local.target_u;
                    Set::Scalar target_T = params_yhi_local.target_T;

                    amrex::ParallelFor(ghost_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        Set::Scalar rho_bnd, un_bnd, ut_bnd, p_bnd, a_bnd, gm_bnd, pi_bnd;
                        compute_prim(eta_arr, rho_arr, M_arr, E_arr, i, j_bnd, k, idir, rho_bnd, un_bnd, ut_bnd, p_bnd, a_bnd, gm_bnd, pi_bnd);

                        Set::Scalar rho_m1, un_m1, ut_m1, p_m1, a_m1, gm_m1, pi_m1;
                        compute_prim(eta_arr, rho_arr, M_arr, E_arr, i, j_bnd - 1, k, idir, rho_m1, un_m1, ut_m1, p_m1, a_m1, gm_m1, pi_m1);

                        Set::Scalar rho_m2, un_m2, ut_m2, p_m2, a_m2, gm_m2, pi_m2;
                        compute_prim(eta_arr, rho_arr, M_arr, E_arr, i, j_bnd - 2, k, idir, rho_m2, un_m2, ut_m2, p_m2, a_m2, gm_m2, pi_m2);

                        Set::Scalar drho_dy = (3.0 * rho_bnd - 4.0 * rho_m1 + rho_m2) / (2.0 * delta);
                        Set::Scalar dun_dy = (3.0 * un_bnd - 4.0 * un_m1 + un_m2) / (2.0 * delta);
                        Set::Scalar dut_dy = (3.0 * ut_bnd - 4.0 * ut_m1 + ut_m2) / (2.0 * delta);
                        Set::Scalar dp_dy = (3.0 * p_bnd - 4.0 * p_m1 + p_m2) / (2.0 * delta);

                        Set::Scalar drho_dx = 0.0, dun_dx = 0.0, dut_dx = 0.0, dp_dx = 0.0;
                        if (i > domain.smallEnd(0) && i < domain.bigEnd(0))
                        {
                            Set::Scalar rho_ip, un_ip, ut_ip, p_ip, a_ip, gm_ip, pi_ip;
                            compute_prim(eta_arr, rho_arr, M_arr, E_arr, i + 1, j_bnd, k, idir, rho_ip, un_ip, ut_ip, p_ip, a_ip, gm_ip, pi_ip);

                            Set::Scalar rho_im, un_im, ut_im, p_im, a_im, gm_im, pi_im;
                            compute_prim(eta_arr, rho_arr, M_arr, E_arr, i - 1, j_bnd, k, idir, rho_im, un_im, ut_im, p_im, a_im, gm_im, pi_im);

                            drho_dx = (rho_ip - rho_im) / (2.0 * dx[0]);
                            dun_dx = (un_ip - un_im) / (2.0 * dx[0]);
                            dut_dx = (ut_ip - ut_im) / (2.0 * dx[0]);
                            dp_dx = (p_ip - p_im) / (2.0 * dx[0]);
                        }

                        Set::Scalar a2_bnd = a_bnd * a_bnd;
                        Set::Scalar M_loc = std::abs(un_bnd) / (a_bnd + small_local);

                        Set::Scalar L1 = (un_bnd - a_bnd) * (dp_dy - rho_bnd * a_bnd * dun_dy);
                        Set::Scalar L2 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd * (a2_bnd * drho_dy - dp_dy) : 0.0;
                        Set::Scalar L3 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd * dut_dy : 0.0;
                        Set::Scalar L4 = (un_bnd + a_bnd) * (dp_dy + rho_bnd * a_bnd * dun_dy);

                        Set::Scalar T1 = ut_bnd * (dp_dx - rho_bnd * a_bnd * dun_dx) + gm_bnd * p_bnd * dut_dx;
                        Set::Scalar T4 = ut_bnd * (dp_dx + rho_bnd * a_bnd * dun_dx) + gm_bnd * p_bnd * dut_dx;
                        Set::Scalar T2 = (std::abs(un_bnd) > 1.0e-14) ? ut_bnd * (a2_bnd * drho_dx - dp_dx) + (a2_bnd * rho_bnd - gm_bnd * p_bnd) * dut_dx : 0.0;
                        Set::Scalar T3 = (std::abs(un_bnd) > 1.0e-14) ? ut_bnd * dut_dx + (1.0 / rho_bnd) * dp_dx : 0.0;

                        Set::Scalar K = sigma_y * (1.0 - M_loc * M_loc) * a_bnd / Ly_ref;

                        if (fty == Type::Outflow)
                        {
                            L1 = K * (p_bnd - py_t) - (1.0 - beta_y) * T1;
                        }
                        else if (fty == Type::Inflow)
                        {
                            Set::Scalar K_acoustic = relax_u * rho_bnd * (1.0 - M_loc * M_loc) / Ly_ref;
                            Set::Scalar K_entropy = relax_u * rho_bnd * a_bnd * (gm_bnd - 1.0) / Ly_ref;
                            Set::Scalar K_shear = relax_u * a_bnd / Ly_ref;

                            Set::Scalar eta_bnd = eta_arr(i, j_bnd, k);
                            Set::Scalar cv_mix = eta_bnd * cv0_ + (1.0 - eta_bnd) * cv1_;
                            Set::Scalar T_bnd = (p_bnd + pi_bnd) / (rho_bnd * cv_mix * (gm_bnd - 1.0) + small_local);

                            L1 = -K_acoustic * (un_bnd - target_v) - T1;
                            L2 = K_entropy * (T_bnd - target_T) - T2;
                            L3 = K_shear * (ut_bnd - target_u) - T3;
                        }

                        Set::Scalar xi1 = L1 / (un_bnd - a_bnd + small_local);
                        Set::Scalar xi2 = (std::abs(un_bnd) > 1.0e-14) ? L2 / un_bnd : 0.0;
                        Set::Scalar xi3 = (std::abs(un_bnd) > 1.0e-14) ? L3 / un_bnd : 0.0;
                        Set::Scalar xi4 = L4 / (un_bnd + a_bnd + small_local);

                        Set::Scalar drho_dy_new = xi1 + xi2 + xi4;
                        Set::Scalar dun_dy_new = (-a_bnd / (rho_bnd + small_local)) * xi1 + (a_bnd / (rho_bnd + small_local)) * xi4;
                        Set::Scalar dut_dy_new = xi3;
                        Set::Scalar dp_dy_new = a2_bnd * (xi1 + xi4);

                        // Layer 1: Motheau Eq. 31
                        Set::Scalar rho_g1 = rho_m1 + 2.0 * delta * drho_dy_new;
                        Set::Scalar un_g1 = un_m1 + 2.0 * delta * dun_dy_new;
                        Set::Scalar ut_g1 = ut_m1 + 2.0 * delta * dut_dy_new;
                        Set::Scalar p_g1 = p_m1 + 2.0 * delta * dp_dy_new;

                        rho_g1 = amrex::max(rho_g1, small_local);
                        p_g1 = amrex::max(p_g1, 1.0e-6);

                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, i, j_bnd + 1, k, idir, rho_g1, un_g1, ut_g1, p_g1);

                        // Layer 2: Motheau Eq. 32
                        Set::Scalar rho_g2 = -2.0 * rho_m1 - 3.0 * rho_bnd + 6.0 * rho_g1 - 6.0 * delta * drho_dy_new;
                        Set::Scalar un_g2 = -2.0 * un_m1 - 3.0 * un_bnd + 6.0 * un_g1 - 6.0 * delta * dun_dy_new;
                        Set::Scalar ut_g2 = -2.0 * ut_m1 - 3.0 * ut_bnd + 6.0 * ut_g1 - 6.0 * delta * dut_dy_new;
                        Set::Scalar p_g2 = -2.0 * p_m1 - 3.0 * p_bnd + 6.0 * p_g1 - 6.0 * delta * dp_dy_new;

                        rho_g2 = amrex::max(rho_g2, small_local);
                        p_g2 = amrex::max(p_g2, 1.0e-6);

                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, i, j_bnd + 2, k, idir, rho_g2, un_g2, ut_g2, p_g2);
                    });
                }
            } // end Y-HI boundary

        } // end MFIter loop

    } // end FillBoundary()

private:
    static BoundaryParams ParseFace(IO::ParmParse &pp, std::string face_name);

}; // end class NSCBC

} // namespace BC

#endif // BC_NSCBC_H
