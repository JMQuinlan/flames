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
///   1. CORNERS FIRST (all 4 layers)
///   2. EDGES SECOND (skip corners since already filled)
///
/// This ordering is essential to prevent NaN propagation from uninitialized corner data.

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
    ///   1. Fill ALL 4 CORNERS first (all 4 ghost cell layers)
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
        // Computes (rho, un, ut, p, a, gamma_mix, pi_mix) at cell (ii, jj, kk)
        // from conserved variables (eta, rho, M, E).
        //
        // dir = normal direction:
        //   dir=0 (x): un=vx, ut=vy
        //   dir=1 (y): un=vy, ut=vx
        //   dir=2 (z): un=vz, ut=vx (3D only)
        //
        // All outputs returned by reference.
        // Uses Tammann EOS for mixed air/water system.
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
                UE = 0.0; // handles NaN and negative

            // Mixed EOS properties (volume-fraction weighted Tammann EOS)
            Set::Scalar A = eta_val / (g0_ - 1.0) + (1.0 - eta_val) / (g1_ - 1.0);
            Set::Scalar B = (eta_val * g0_ * pi0_) / (g0_ - 1.0)
                            + ((1.0 - eta_val) * g1_ * pi1_) / (g1_ - 1.0);
            Set::Scalar gamma_mix = 1.0 + 1.0 / A;
            Set::Scalar pi_mix = (B / A) / gamma_mix;

            // Pressure (Tammann EOS)
            Set::Scalar p_val = (gamma_mix - 1.0) * UE - gamma_mix * pi_mix + pref_local;
            if (!(p_val > 0.0))
                p_val = 1.0e-6; // handles NaN and negative

            // Sound speed
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
                un_out = vy; // normal velocity
                ut_out = vx; // tangential velocity
#if AMREX_SPACEDIM == 3
            }
            else
            {                // dir == 2
                un_out = vz; // normal velocity
                ut_out = vx; // tangential velocity
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
        // Writes conserved variables (rho, M, E) to ghost cell (ii, jj, kk)
        // from primitive values (rho, un, ut, p).
        //
        // Uses eta already present in the ghost cell and EOS to compute UE from p.
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

            // Inverse EOS: compute UE from p
            Set::Scalar UE = (p_val + gamma_mix * pi_mix - pref_local) / (gamma_mix - 1.0);
            if (UE < 0.0)
                UE = 0.0;

            // Velocities (convert un, ut back to vx, vy, vz)
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
            { // dir == 2
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

            // Get Array4 views of conserved variables
            amrex::Array4<const amrex::Real> const &eta_arr = eta_mf.const_array(mfi);
            amrex::Array4<amrex::Real> const &rho_arr = density_mf.array(mfi);
            amrex::Array4<amrex::Real> const &M_arr = momentum_mf.array(mfi);
            amrex::Array4<amrex::Real> const &E_arr = energy_mf.array(mfi);

        // =====================================================================
        // STEP 1: FILL ALL 4 CORNERS FIRST (Motheau Figure 2)
        // =====================================================================
        // CRITICAL: Corners MUST be filled before edges to prevent NaN propagation
        // Each corner requires solving a 2x2 coupled system (Motheau Eq. 56-58)
        // and filling all 4 ghost cell layers (Motheau Eq. 27-30)
        // =====================================================================

        // =====================================================================
        // CORNER FILLING LOOP - ALL 4 CORNERS
        // =====================================================================
        // Loop over all 4 corners: (xlo,ylo), (xhi,ylo), (xlo,yhi), (xhi,yhi)
        // cx=0 -> xlo, cx=1 -> xhi
        // cy=0 -> ylo, cy=1 -> yhi
        // =====================================================================
        for (int cx = 0; cx < 2; ++cx)
            for (int cy = 0; cy < 2; ++cy)
            {
                BoundaryParams params_x = (cx == 0) ? params_xlo_local : params_xhi_local;
                BoundaryParams params_y = (cy == 0) ? params_ylo_local : params_yhi_local;

                // Skip if either boundary is not NSCBC
                if (params_x.type == Type::None || params_y.type == Type::None)
                    continue;

                // Determine boundary indices
                int ibdry_x = (cx == 0) ? domain.smallEnd(0) : domain.bigEnd(0);
                int jbdry_y = (cy == 0) ? domain.smallEnd(1) : domain.bigEnd(1);

                // Check if this box touches the corner
                bool touch_x = (cx == 0) ? (bx.smallEnd(0) == domain.smallEnd(0))
                                            : (bx.bigEnd(0) == domain.bigEnd(0));
                bool touch_y = (cy == 0) ? (bx.smallEnd(1) == domain.smallEnd(1))
                                            : (bx.bigEnd(1) == domain.bigEnd(1));

                if (!touch_x || !touch_y)
                    continue;

                // =================================================================
                // BUILD CORNER GHOST BOX - ALL 4 LAYERS (Motheau Figure 2)
                // =================================================================
                // CRITICAL: Must fill all 4 ghost cell layers for PPM reconstruction
                // =================================================================
                int ng = 4;                  // Number of ghost cell layers
                int sx = (cx == 1) ? 1 : -1; // Sign for x-direction
                int sy = (cy == 1) ? 1 : -1; // Sign for y-direction

                amrex::Box corner_box;
                corner_box.setSmall(0, (cx == 1) ? ibdry_x + 1 : ibdry_x - ng);
                corner_box.setBig(0, (cx == 1) ? ibdry_x + ng : ibdry_x - 1);
                corner_box.setSmall(1, (cy == 1) ? jbdry_y + 1 : jbdry_y - ng);
                corner_box.setBig(1, (cy == 1) ? jbdry_y + ng : jbdry_y - 1);
#if AMREX_SPACEDIM == 3
                corner_box.setSmall(2, domain.smallEnd(2));
                corner_box.setBig(2, domain.bigEnd(2));
#endif

                // Capture parameters for GPU
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

                // =================================================================
                // PARALLEL LOOP OVER ALL CORNER GHOST CELLS
                // =================================================================
                amrex::ParallelFor(corner_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    // =============================================================
                    // Physical corner cell (at domain boundary intersection)
                    // =============================================================
                    int ic = ibdry_x_local;
                    int jc = jbdry_y_local;

                    // Distance from corner (1, 2, 3, or 4)
                    int ox = std::abs(i - ic);
                    int oy = std::abs(j - jc);

                    // =============================================================
                    // STEP 1: Compute primitives at corner boundary cell
                    // =============================================================
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

                    // =============================================================
                    // STEP 2: X-direction one-sided derivatives at corner
                    // Uses Motheau Eq. 39 (forward) or Eq. 40 (backward)
                    // =============================================================
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

                    // One-sided derivative (Eq. 39 or 40 depending on sx_local)
                    Set::Scalar fdx = (Set::Scalar)sx_local;
                    Set::Scalar drho_dx = fdx * (3.0 * rho_c - 4.0 * rho_xm1 + rho_xm2) / (2.0 * dx[0]);
                    Set::Scalar dvx_dx = fdx * (3.0 * vx_c - 4.0 * vx_xm1 + vx_xm2) / (2.0 * dx[0]);
                    Set::Scalar dvy_dx = fdx * (3.0 * vy_c - 4.0 * vy_xm1 + vy_xm2) / (2.0 * dx[0]);
                    Set::Scalar dp_dx = fdx * (3.0 * p_c - 4.0 * p_xm1 + p_xm2) / (2.0 * dx[0]);

                    // =============================================================
                    // STEP 3: Y-direction one-sided derivatives at corner
                    // Uses Motheau Eq. 39 (forward) or Eq. 40 (backward)
                    // =============================================================
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

                    // One-sided derivative (Eq. 39 or 40 depending on sy_local)
                    Set::Scalar fdy = (Set::Scalar)sy_local;
                    Set::Scalar drho_dy = fdy * (3.0 * rho_c - 4.0 * rho_ym1 + rho_ym2) / (2.0 * dx[1]);
                    Set::Scalar dvy_dy = fdy * (3.0 * vy_c - 4.0 * vy_ym1 + vy_ym2) / (2.0 * dx[1]);
                    Set::Scalar dvx_dy = fdy * (3.0 * vx_c - 4.0 * vx_ym1 + vx_ym2) / (2.0 * dx[1]);
                    Set::Scalar dp_dy = fdy * (3.0 * p_c - 4.0 * p_ym1 + p_ym2) / (2.0 * dx[1]);

                    // =============================================================
                    // STEP 4: LODI waves in X-direction (Motheau Eq. 23)
                    // L_1 = lambda_1 * (dp/dx - rho*c*du/dx)
                    // L_2 = lambda_2 * (c^2*drho/dx - dp/dx)
                    // L_3 = lambda_3 * dv/dx
                    // L_4 = lambda_4 * (dp/dx + rho*c*du/dx)
                    // =============================================================
                    Set::Scalar L1_x = (vx_c - a_c) / (2.0 * a2_c) * (dp_dx - rho_c * a_c * dvx_dx);
                    Set::Scalar L2_x = (std::abs(vx_c) > 1.0e-14) ? vx_c / a2_c * (a2_c * drho_dx - dp_dx) : 0.0;
                    Set::Scalar L3_x = (std::abs(vx_c) > 1.0e-14) ? vx_c * dvy_dx : 0.0;
                    Set::Scalar L4_x = (vx_c + a_c) / (2.0 * a2_c) * (dp_dx + rho_c * a_c * dvx_dx);

                    // =============================================================
                    // STEP 5: LODI waves in Y-direction (Motheau Eq. 23)
                    // CORRECTED: L2 is entropy, L3 is vorticity
                    // =============================================================
                    Set::Scalar L1_y = (vy_c - a_c) / (2.0 * a2_c) * (dp_dy - rho_c * a_c * dvy_dy);
                    Set::Scalar L2_y = (std::abs(vy_c) > 1.0e-14) ? vy_c / a2_c * (a2_c * drho_dy - dp_dy) : 0.0;
                    Set::Scalar L3_y = (std::abs(vy_c) > 1.0e-14) ? vy_c * dvx_dy : 0.0;
                    Set::Scalar L4_y = (vy_c + a_c) / (2.0 * a2_c) * (dp_dy + rho_c * a_c * dvy_dy);

                    // =============================================================
                    // STEP 6: Solve 2x2 coupled system for incoming waves
                    // (Motheau Eq. 56-58)
                    // =============================================================
                    Set::Scalar Lx_in = (cx == 1) ? L1_x : L4_x; // Incoming wave in X
                    Set::Scalar Ly_in = (cy == 1) ? L1_y : L4_y; // Incoming wave in Y

                    // Relaxation coefficients (Motheau Eq. 38)
                    Set::Scalar Kx = sigma_x * (1.0 - Mx_c * Mx_c) / (a_c * Lx_ref);
                    Set::Scalar Ky = sigma_y * (1.0 - My_c * My_c) / (a_c * Ly_ref);

                    // Coupling coefficient (Motheau Eq. 56)
                    Set::Scalar ab = (1.0 - 0.5 * (beta_x + beta_y)) / 2.0;

                    if (ftx == Type::Outflow && fty == Type::Outflow)
                    {
                        // Motheau Eq. 56: Coupled outflow/outflow corner
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

                    // Update wave amplitudes with coupled values
                    if (cx == 1)
                        L1_x = Lx_in;
                    else
                        L4_x = Lx_in;
                    if (cy == 1)
                        L1_y = Ly_in;
                    else
                        L4_y = Ly_in;

                    // =============================================================
                    // STEP 7: Reconstruct gradients from modified L vectors
                    // (Motheau Eq. 54-55, inverse of Eq. 23)
                    // =============================================================
                    // X-direction gradients
                    Set::Scalar xi1_x = L1_x / (vx_c - a_c + small_local);
                    Set::Scalar xi2_x = (std::abs(vx_c) > 1.0e-14) ? L2_x / vx_c : 0.0;
                    Set::Scalar xi3_x = (std::abs(vx_c) > 1.0e-14) ? L3_x / vx_c : 0.0;
                    Set::Scalar xi4_x = L4_x / (vx_c + a_c + small_local);

                    Set::Scalar drho_dx_new = xi1_x + xi2_x + xi4_x;
                    Set::Scalar dvx_dx_new = (-a_c / (rho_c + small_local)) * xi1_x + (a_c / (rho_c + small_local)) * xi4_x;
                    Set::Scalar dvy_dx_new = xi3_x;
                    Set::Scalar dp_dx_new = a2_c * (xi1_x + xi4_x);

                    // Y-direction gradients
                    Set::Scalar xi1_y = L1_y / (vy_c - a_c + small_local);
                    Set::Scalar xi2_y = (std::abs(vy_c) > 1.0e-14) ? L2_y / vy_c : 0.0;
                    Set::Scalar xi3_y = (std::abs(vy_c) > 1.0e-14) ? L3_y / vy_c : 0.0;
                    Set::Scalar xi4_y = L4_y / (vy_c + a_c + small_local);

                    Set::Scalar drho_dy_new = xi1_y + xi2_y + xi4_y;
                    Set::Scalar dvy_dy_new = (-a_c / (rho_c + small_local)) * xi1_y + (a_c / (rho_c + small_local)) * xi4_y;
                    Set::Scalar dvx_dy_new = xi3_y;
                    Set::Scalar dp_dy_new = a2_c * (xi1_y + xi4_y);

                    // =============================================================
                    // STEP 8: Fill ghost cells using layer-appropriate formula
                    // (Motheau Eqs. 27-30 for layers 1-4)
                    // =============================================================
                    Set::Scalar rho_g, vx_g, vy_g, p_g;

                    // Determine which layer (1, 2, 3, or 4) based on distance from corner
                    int layer = std::max(ox, oy);

                    if (layer == 1)
                    {
                        // ==========================================================
                        // LAYER 1: Motheau Eq. 27/31 - Linear extrapolation
                        // Q_{-1} = Q_{+1} - 2*dx*dQ/dx
                        // ==========================================================
                        Set::Scalar dx_off = (Set::Scalar)(i - ic) * dx[0];
                        Set::Scalar dy_off = (Set::Scalar)(j - jc) * dx[1];

                        rho_g = rho_c + dx_off * drho_dx_new + dy_off * drho_dy_new;
                        vx_g = vx_c + dx_off * dvx_dx_new + dy_off * dvx_dy_new;
                        vy_g = vy_c + dx_off * dvy_dx_new + dy_off * dvy_dy_new;
                        p_g = p_c + dx_off * dp_dx_new + dy_off * dp_dy_new;
                    }
                    else if (layer == 2)
                    {
                        // ==========================================================
                        // LAYER 2: Motheau Eq. 28/32 - Quadratic extrapolation
                        // Q_{-2} = -2*Q_{+1} - 3*Q_0 + 6*Q_{-1} + 6*dx*dQ/dx
                        // ==========================================================
                        // Get layer 1 ghost cell values
                        int i1 = ic + sx_local * ((ox > 0) ? 1 : 0);
                        int j1 = jc + sy_local * ((oy > 0) ? 1 : 0);

                        Set::Scalar rho_1 = rho_arr(i1, j1, k);
                        Set::Scalar vx_1 = M_arr(i1, j1, k, 0) / (rho_1 + small_local);
                        Set::Scalar vy_1 = M_arr(i1, j1, k, 1) / (rho_1 + small_local);

                        // Compute pressure at layer 1
                        Set::Scalar eta_1 = eta_arr(i1, j1, k);
                        Set::Scalar A_1 = eta_1 / (g0_ - 1.0) + (1.0 - eta_1) / (g1_ - 1.0);
                        Set::Scalar gm_1 = 1.0 + 1.0 / A_1;
                        Set::Scalar KE_1 = 0.5 * rho_1 * (vx_1 * vx_1 + vy_1 * vy_1);
                        Set::Scalar UE_1 = E_arr(i1, j1, k) - KE_1;
                        Set::Scalar B_1 = (eta_1 * g0_ * pi0_) / (g0_ - 1.0) + ((1.0 - eta_1) * g1_ * pi1_) / (g1_ - 1.0);
                        Set::Scalar pi_1 = (B_1 / A_1) / gm_1;
                        Set::Scalar p_1 = (gm_1 - 1.0) * UE_1 - gm_1 * pi_1 + pref_local;

                        // Motheau Eq. 28/32
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
                    else if (layer == 3)
                    {
                        // ==========================================================
                        // LAYER 3: Motheau Eq. 29/33 - Cubic extrapolation
                        // Q_{-3} = 3*Q_{+1} + 10*Q_0 - 18*Q_{-1} + 6*Q_{-2} - 12*dx*dQ/dx
                        // ==========================================================
                        // Get layer 1 and 2 ghost cell values
                        int i1 = ic + sx_local * 1;
                        int j1 = jc + sy_local * ((oy >= 1) ? 1 : 0);
                        int i2 = ic + sx_local * ((ox >= 2) ? 2 : 0);
                        int j2 = jc + sy_local * ((oy >= 2) ? 2 : 0);

                        Set::Scalar rho_1 = rho_arr(i1, j1, k);
                        Set::Scalar vx_1 = M_arr(i1, j1, k, 0) / (rho_1 + small_local);
                        Set::Scalar vy_1 = M_arr(i1, j1, k, 1) / (rho_1 + small_local);

                        Set::Scalar rho_2 = rho_arr(i2, j2, k);
                        Set::Scalar vx_2 = M_arr(i2, j2, k, 0) / (rho_2 + small_local);
                        Set::Scalar vy_2 = M_arr(i2, j2, k, 1) / (rho_2 + small_local);

                        // Compute pressures
                        Set::Scalar eta_1 = eta_arr(i1, j1, k);
                        Set::Scalar A_1 = eta_1 / (g0_ - 1.0) + (1.0 - eta_1) / (g1_ - 1.0);
                        Set::Scalar gm_1 = 1.0 + 1.0 / A_1;
                        Set::Scalar KE_1 = 0.5 * rho_1 * (vx_1 * vx_1 + vy_1 * vy_1);
                        Set::Scalar UE_1 = E_arr(i1, j1, k) - KE_1;
                        Set::Scalar B_1 = (eta_1 * g0_ * pi0_) / (g0_ - 1.0) + ((1.0 - eta_1) * g1_ * pi1_) / (g1_ - 1.0);
                        Set::Scalar pi_1 = (B_1 / A_1) / gm_1;
                        Set::Scalar p_1 = (gm_1 - 1.0) * UE_1 - gm_1 * pi_1 + pref_local;

                        Set::Scalar eta_2 = eta_arr(i2, j2, k);
                        Set::Scalar A_2 = eta_2 / (g0_ - 1.0) + (1.0 - eta_2) / (g1_ - 1.0);
                        Set::Scalar gm_2 = 1.0 + 1.0 / A_2;
                        Set::Scalar KE_2 = 0.5 * rho_2 * (vx_2 * vx_2 + vy_2 * vy_2);
                        Set::Scalar UE_2 = E_arr(i2, j2, k) - KE_2;
                        Set::Scalar B_2 = (eta_2 * g0_ * pi0_) / (g0_ - 1.0) + ((1.0 - eta_2) * g1_ * pi1_) / (g1_ - 1.0);
                        Set::Scalar pi_2 = (B_2 / A_2) / gm_2;
                        Set::Scalar p_2 = (gm_2 - 1.0) * UE_2 - gm_2 * pi_2 + pref_local;

                        // Motheau Eq. 29/33
                        Set::Scalar dx_grad = (ox == 3) ? -12.0 * dx[0] * drho_dx_new : 0.0;
                        Set::Scalar dy_grad = (oy == 3) ? -12.0 * dx[1] * drho_dy_new : 0.0;
                        rho_g = 3.0 * rho_xm1 + 10.0 * rho_c - 18.0 * rho_1 + 6.0 * rho_2 + dx_grad + dy_grad;

                        dx_grad = (ox == 3) ? -12.0 * dx[0] * dvx_dx_new : 0.0;
                        dy_grad = (oy == 3) ? -12.0 * dx[1] * dvx_dy_new : 0.0;
                        vx_g = 3.0 * vx_xm1 + 10.0 * vx_c - 18.0 * vx_1 + 6.0 * vx_2 + dx_grad + dy_grad;

                        dx_grad = (ox == 3) ? -12.0 * dx[0] * dvy_dx_new : 0.0;
                        dy_grad = (oy == 3) ? -12.0 * dx[1] * dvy_dy_new : 0.0;
                        vy_g = 3.0 * vy_xm1 + 10.0 * vy_c - 18.0 * vy_1 + 6.0 * vy_2 + dx_grad + dy_grad;

                        dx_grad = (ox == 3) ? -12.0 * dx[0] * dp_dx_new : 0.0;
                        dy_grad = (oy == 3) ? -12.0 * dx[1] * dp_dy_new : 0.0;
                        p_g = 3.0 * p_xm1 + 10.0 * p_c - 18.0 * p_1 + 6.0 * p_2 + dx_grad + dy_grad;
                    }
                    else // layer == 4
                    {
                        // ==========================================================
                        // LAYER 4: Motheau Eq. 30/34 - Quartic extrapolation
                        // Q_{-4} = -2*Q_{+1} - 13*Q_0 + 24*Q_{-1} - 12*Q_{-2} + 4*Q_{-3} + 12*dx*dQ/dx
                        // ==========================================================
                        // Get layers 1, 2, 3 ghost cell values
                        int i1 = ic + sx_local * 1;
                        int j1 = jc + sy_local * ((oy >= 1) ? 1 : 0);
                        int i2 = ic + sx_local * ((ox >= 2) ? 2 : 0);
                        int j2 = jc + sy_local * ((oy >= 2) ? 2 : 0);
                        int i3 = ic + sx_local * ((ox >= 3) ? 3 : 0);
                        int j3 = jc + sy_local * ((oy >= 3) ? 3 : 0);

                        Set::Scalar rho_1 = rho_arr(i1, j1, k);
                        Set::Scalar vx_1 = M_arr(i1, j1, k, 0) / (rho_1 + small_local);
                        Set::Scalar vy_1 = M_arr(i1, j1, k, 1) / (rho_1 + small_local);

                        Set::Scalar rho_2 = rho_arr(i2, j2, k);
                        Set::Scalar vx_2 = M_arr(i2, j2, k, 0) / (rho_2 + small_local);
                        Set::Scalar vy_2 = M_arr(i2, j2, k, 1) / (rho_2 + small_local);

                        Set::Scalar rho_3 = rho_arr(i3, j3, k);
                        Set::Scalar vx_3 = M_arr(i3, j3, k, 0) / (rho_3 + small_local);
                        Set::Scalar vy_3 = M_arr(i3, j3, k, 1) / (rho_3 + small_local);

                        // Compute pressures (same pattern as layer 3)
                        Set::Scalar eta_1 = eta_arr(i1, j1, k);
                        Set::Scalar A_1 = eta_1 / (g0_ - 1.0) + (1.0 - eta_1) / (g1_ - 1.0);
                        Set::Scalar gm_1 = 1.0 + 1.0 / A_1;
                        Set::Scalar KE_1 = 0.5 * rho_1 * (vx_1 * vx_1 + vy_1 * vy_1);
                        Set::Scalar UE_1 = E_arr(i1, j1, k) - KE_1;
                        Set::Scalar B_1 = (eta_1 * g0_ * pi0_) / (g0_ - 1.0) + ((1.0 - eta_1) * g1_ * pi1_) / (g1_ - 1.0);
                        Set::Scalar pi_1 = (B_1 / A_1) / gm_1;
                        Set::Scalar p_1 = (gm_1 - 1.0) * UE_1 - gm_1 * pi_1 + pref_local;

                        Set::Scalar eta_2 = eta_arr(i2, j2, k);
                        Set::Scalar A_2 = eta_2 / (g0_ - 1.0) + (1.0 - eta_2) / (g1_ - 1.0);
                        Set::Scalar gm_2 = 1.0 + 1.0 / A_2;
                        Set::Scalar KE_2 = 0.5 * rho_2 * (vx_2 * vx_2 + vy_2 * vy_2);
                        Set::Scalar UE_2 = E_arr(i2, j2, k) - KE_2;
                        Set::Scalar B_2 = (eta_2 * g0_ * pi0_) / (g0_ - 1.0) + ((1.0 - eta_2) * g1_ * pi1_) / (g1_ - 1.0);
                        Set::Scalar pi_2 = (B_2 / A_2) / gm_2;
                        Set::Scalar p_2 = (gm_2 - 1.0) * UE_2 - gm_2 * pi_2 + pref_local;

                        Set::Scalar eta_3 = eta_arr(i3, j3, k);
                        Set::Scalar A_3 = eta_3 / (g0_ - 1.0) + (1.0 - eta_3) / (g1_ - 1.0);
                        Set::Scalar gm_3 = 1.0 + 1.0 / A_3;
                        Set::Scalar KE_3 = 0.5 * rho_3 * (vx_3 * vx_3 + vy_3 * vy_3);
                        Set::Scalar UE_3 = E_arr(i3, j3, k) - KE_3;
                        Set::Scalar B_3 = (eta_3 * g0_ * pi0_) / (g0_ - 1.0) + ((1.0 - eta_3) * g1_ * pi1_) / (g1_ - 1.0);
                        Set::Scalar pi_3 = (B_3 / A_3) / gm_3;
                        Set::Scalar p_3 = (gm_3 - 1.0) * UE_3 - gm_3 * pi_3 + pref_local;

                        // Motheau Eq. 30/34
                        Set::Scalar dx_grad = (ox == 4) ? 12.0 * dx[0] * drho_dx_new : 0.0;
                        Set::Scalar dy_grad = (oy == 4) ? 12.0 * dx[1] * drho_dy_new : 0.0;
                        rho_g = -2.0 * rho_xm1 - 13.0 * rho_c + 24.0 * rho_1 - 12.0 * rho_2 + 4.0 * rho_3 + dx_grad + dy_grad;

                        dx_grad = (ox == 4) ? 12.0 * dx[0] * dvx_dx_new : 0.0;
                        dy_grad = (oy == 4) ? 12.0 * dx[1] * dvx_dy_new : 0.0;
                        vx_g = -2.0 * vx_xm1 - 13.0 * vx_c + 24.0 * vx_1 - 12.0 * vx_2 + 4.0 * vx_3 + dx_grad + dy_grad;

                        dx_grad = (ox == 4) ? 12.0 * dx[0] * dvy_dx_new : 0.0;
                        dy_grad = (oy == 4) ? 12.0 * dx[1] * dvy_dy_new : 0.0;
                        vy_g = -2.0 * vy_xm1 - 13.0 * vy_c + 24.0 * vy_1 - 12.0 * vy_2 + 4.0 * vy_3 + dx_grad + dy_grad;

                        dx_grad = (ox == 4) ? 12.0 * dx[0] * dp_dx_new : 0.0;
                        dy_grad = (oy == 4) ? 12.0 * dx[1] * dp_dy_new : 0.0;
                        p_g = -2.0 * p_xm1 - 13.0 * p_c + 24.0 * p_1 - 12.0 * p_2 + 4.0 * p_3 + dx_grad + dy_grad;
                    }

                    // =============================================================
                    // STEP 9: Enforce physical bounds
                    // =============================================================
                    rho_g = amrex::min(amrex::max(rho_g, 1.0e-8), 10000.0);
                    p_g = amrex::min(amrex::max(p_g, 1.0), 1.0e8);
                    vx_g = amrex::min(amrex::max(vx_g, -1000.0), 1000.0);
                    vy_g = amrex::min(amrex::max(vy_g, -1000.0), 1000.0);

                    // =============================================================
                    // STEP 10: Write to conserved arrays
                    // =============================================================
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

            } // end corner loops (cx, cy)
        // =====================================================================
        // STEP 2: FILL EDGES (SKIP CORNERS - ALREADY FILLED)
        // =====================================================================
        // Now that all corners are filled, we can fill the edges.
        // Edge loops explicitly skip corner cells to avoid overwriting.
        // Each edge fills all 4 ghost cell layers using Motheau Eq. 27-30 or 31-34.
        // =====================================================================

        // =====================================================================
        // X-LO BOUNDARY (LEFT EDGE - SKIP CORNERS)
        // =====================================================================
        if (bx.smallEnd(0) == domain.smallEnd(0) && params_xlo_local.type != Type::None)
        {
            int idir = 0;   // x-direction
            int isign = -1; // left boundary
            Set::Scalar delta = dx[0];

            // Boundary index
            int i_bnd = domain.smallEnd(0);

            // Loop over j indices, EXCLUDING corners
            int j_start = domain.smallEnd(1) + 1; // Skip bottom corner
            int j_end = domain.bigEnd(1) - 1;     // Skip top corner

            // Only process if this box touches the boundary
            for (int jj = j_start; jj <= j_end; ++jj)
            {
                // Check if this j is in our box
                if (jj < bx.smallEnd(1) || jj > bx.bigEnd(1))
                    continue;

                // Create a box for this j-slice with 4 ghost cell layers
                amrex::Box ghost_box;
                ghost_box.setSmall(0, i_bnd - 4);
                ghost_box.setBig(0, i_bnd - 1);
                ghost_box.setSmall(1, jj);
                ghost_box.setBig(1, jj);
#if AMREX_SPACEDIM == 3
                ghost_box.setSmall(2, bx.smallEnd(2));
                ghost_box.setBig(2, bx.bigEnd(2));
#endif

                amrex::ParallelFor(ghost_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    // =============================================================
                    // STEP 1: Compute primitives at boundary cell
                    // =============================================================
                    Set::Scalar rho_bnd, un_bnd, ut_bnd, p_bnd, a_bnd, gm_bnd, pi_bnd;
                    compute_prim(eta_arr, rho_arr, M_arr, E_arr, i_bnd, j, k, idir, rho_bnd, un_bnd, ut_bnd, p_bnd, a_bnd, gm_bnd, pi_bnd);

                    // =============================================================
                    // STEP 2: Compute primitives at interior neighbors (for FD)
                    // =============================================================
                    Set::Scalar rho_m1, un_m1, ut_m1, p_m1, a_m1, gm_m1, pi_m1;
                    compute_prim(eta_arr, rho_arr, M_arr, E_arr, i_bnd + 1, j, k, idir, rho_m1, un_m1, ut_m1, p_m1, a_m1, gm_m1, pi_m1);

                    Set::Scalar rho_m2, un_m2, ut_m2, p_m2, a_m2, gm_m2, pi_m2;
                    compute_prim(eta_arr, rho_arr, M_arr, E_arr, i_bnd + 2, j, k, idir, rho_m2, un_m2, ut_m2, p_m2, a_m2, gm_m2, pi_m2);

                    // =============================================================
                    // STEP 3: One-sided normal derivatives (2nd order)
                    // Motheau Eq. 39: Forward difference for left boundary
                    // df/dx = (-3*f_0 + 4*f_1 - f_2) / (2*dx)
                    // =============================================================
                    Set::Scalar drho_dx = (-3.0 * rho_bnd + 4.0 * rho_m1 - rho_m2) / (2.0 * delta);
                    Set::Scalar dun_dx = (-3.0 * un_bnd + 4.0 * un_m1 - un_m2) / (2.0 * delta);
                    Set::Scalar dut_dx = (-3.0 * ut_bnd + 4.0 * ut_m1 - ut_m2) / (2.0 * delta);
                    Set::Scalar dp_dx = (-3.0 * p_bnd + 4.0 * p_m1 - p_m2) / (2.0 * delta);

                    // =============================================================
                    // STEP 4: Tangential derivatives (centered, 2nd order)
                    // Motheau Eq. 41: Y-direction df/dy = (f_{j+1} - f_{j-1}) / (2*dy)
                    // =============================================================
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

                    // =============================================================
                    // STEP 5: LODI wave amplitudes (Motheau Eq. 23)
                    // L_1 = lambda_1 * (dp/dx - rho*c*du/dx)
                    // L_2 = lambda_2 * (c^2*drho/dx - dp/dx)
                    // L_3 = lambda_3 * dv/dx
                    // L_4 = lambda_4 * (dp/dx + rho*c*du/dx)
                    // =============================================================
                    Set::Scalar L1 = (un_bnd - a_bnd) / (2.0 * a2_bnd) * (dp_dx - rho_bnd * a_bnd * dun_dx);
                    Set::Scalar L2 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd / a2_bnd * (a2_bnd * drho_dx - dp_dx) : 0.0;
                    Set::Scalar L3 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd * dut_dx : 0.0;
                    Set::Scalar L4 = (un_bnd + a_bnd) / (2.0 * a2_bnd) * (dp_dx + rho_bnd * a_bnd * dun_dx);

                    // =============================================================
                    // STEP 6: Transverse terms (Motheau Eq. 36-37 with gamma)
                    // CORRECTED: Added gamma multiplier
                    // =============================================================
                    Set::Scalar T1 = (un_bnd - a_bnd) / (2.0 * a2_bnd) * (-rho_bnd * a_bnd * ut_bnd * dun_dy + a2_bnd * rho_bnd * dut_dy + ut_bnd * dp_dy);
                    Set::Scalar T4 = (un_bnd + a_bnd) / (2.0 * a2_bnd) * (rho_bnd * a_bnd * ut_bnd * dun_dy + a2_bnd * rho_bnd * dut_dy + ut_bnd * dp_dy);
                    Set::Scalar T2 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd * ut_bnd / a2_bnd * (a2_bnd * drho_dy - dp_dy) : 0.0;
                    Set::Scalar T3 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd * (ut_bnd * dut_dy + (1.0 / rho_bnd) * dp_dy) : 0.0;

                    // =============================================================
                    // STEP 7: Apply BC-specific modification to incoming wave
                    // For X-LO: L4 is incoming (speed un+a > 0 entering domain)
                    // =============================================================
                    Set::Scalar K = params_xlo_local.sigma * (1.0 - M_loc * M_loc) / (a_bnd * params_xlo_local.L_ref);

                    if (params_xlo_local.type == Type::Outflow)
                    {
                        L4 = K * (p_bnd - params_xlo_local.target_p) - (1.0 - params_xlo_local.beta) * T4;
                    }
                    else if (params_xlo_local.type == Type::Inflow)
                    {
                        Set::Scalar er = params_xlo_local.relax_u;
                        Set::Scalar K_acoustic = er * rho_bnd * (1.0 - M_loc * M_loc) / params_xlo_local.L_ref;
                        Set::Scalar K_entropy = er * rho_bnd * a_bnd * (gm_bnd - 1.0) / params_xlo_local.L_ref;
                        Set::Scalar K_shear = er * a_bnd / params_xlo_local.L_ref;

                        // Temperature (for entropy relaxation)
                        Set::Scalar eta_bnd = eta_arr(i_bnd, j, k);
                        Set::Scalar cv_mix = eta_bnd * cv0_ + (1.0 - eta_bnd) * cv1_;
                        Set::Scalar T_bnd = (p_bnd + pi_bnd) / (rho_bnd * cv_mix * (gm_bnd - 1.0) + small_local);

                        L4 = K_acoustic * (un_bnd - params_xlo_local.target_u) - T4;
                        L2 = K_entropy * (T_bnd - params_xlo_local.target_T) - T2;
                        L3 = K_shear * (ut_bnd - params_xlo_local.target_v) - T3;
                    }

                    // =============================================================
                    // STEP 8: Reconstruct modified gradients (Motheau Eq. 54)
                    // =============================================================
                    Set::Scalar xi1 = L1 / (un_bnd - a_bnd + small_local);
                    Set::Scalar xi2 = (std::abs(un_bnd) > 1.0e-14) ? L2 / un_bnd : 0.0;
                    Set::Scalar xi3 = (std::abs(un_bnd) > 1.0e-14) ? L3 / un_bnd : 0.0;
                    Set::Scalar xi4 = L4 / (un_bnd + a_bnd + small_local);

                    Set::Scalar drho_dx_new = xi1 + xi2 + xi4;
                    Set::Scalar dun_dx_new = (-a_bnd / (rho_bnd + small_local)) * xi1 + (a_bnd / (rho_bnd + small_local)) * xi4;
                    Set::Scalar dut_dx_new = xi3;
                    Set::Scalar dp_dx_new = a2_bnd * (xi1 + xi4);

                    // =============================================================
                    // STEP 9: Fill ALL 4 ghost cell layers (Motheau Eq. 27-30)
                    // =============================================================

                    // Layer 1: Motheau Eq. 27
                    // Q_{i-1} = Q_{i+1} - 2*dx*dQ/dx
                    Set::Scalar rho_g1 = rho_m1 - 2.0 * delta * drho_dx_new;
                    Set::Scalar un_g1 = un_m1 - 2.0 * delta * dun_dx_new;
                    Set::Scalar ut_g1 = ut_m1 - 2.0 * delta * dut_dx_new;
                    Set::Scalar p_g1 = p_m1 - 2.0 * delta * dp_dx_new;

                    rho_g1 = amrex::max(rho_g1, small_local);
                    p_g1 = amrex::max(p_g1, 1.0e-6);
                    set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, i_bnd - 1, j, k, idir, rho_g1, un_g1, ut_g1, p_g1);

                    // Layer 2: Motheau Eq. 28
                    // Q_{i-2} = -2*Q_{i+1} - 3*Q_i + 6*Q_{i-1} + 6*dx*dQ/dx
                    Set::Scalar rho_g2 = -2.0 * rho_m1 - 3.0 * rho_bnd + 6.0 * rho_g1 + 6.0 * delta * drho_dx_new;
                    Set::Scalar un_g2 = -2.0 * un_m1 - 3.0 * un_bnd + 6.0 * un_g1 + 6.0 * delta * dun_dx_new;
                    Set::Scalar ut_g2 = -2.0 * ut_m1 - 3.0 * ut_bnd + 6.0 * ut_g1 + 6.0 * delta * dut_dx_new;
                    Set::Scalar p_g2 = -2.0 * p_m1 - 3.0 * p_bnd + 6.0 * p_g1 + 6.0 * delta * dp_dx_new;

                    rho_g2 = amrex::max(rho_g2, small_local);
                    p_g2 = amrex::max(p_g2, 1.0e-6);
                    set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, i_bnd - 2, j, k, idir, rho_g2, un_g2, ut_g2, p_g2);

                    // Layer 3: Motheau Eq. 29
                    // Q_{i-3} = 3*Q_{i+1} + 10*Q_i - 18*Q_{i-1} + 6*Q_{i-2} - 12*dx*dQ/dx
                    Set::Scalar rho_g3 = 3.0 * rho_m1 + 10.0 * rho_bnd - 18.0 * rho_g1 + 6.0 * rho_g2 - 12.0 * delta * drho_dx_new;
                    Set::Scalar un_g3 = 3.0 * un_m1 + 10.0 * un_bnd - 18.0 * un_g1 + 6.0 * un_g2 - 12.0 * delta * dun_dx_new;
                    Set::Scalar ut_g3 = 3.0 * ut_m1 + 10.0 * ut_bnd - 18.0 * ut_g1 + 6.0 * ut_g2 - 12.0 * delta * dut_dx_new;
                    Set::Scalar p_g3 = 3.0 * p_m1 + 10.0 * p_bnd - 18.0 * p_g1 + 6.0 * p_g2 - 12.0 * delta * dp_dx_new;

                    rho_g3 = amrex::max(rho_g3, small_local);
                    p_g3 = amrex::max(p_g3, 1.0e-6);
                    set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, i_bnd - 3, j, k, idir, rho_g3, un_g3, ut_g3, p_g3);

                    // Layer 4: Motheau Eq. 30
                    // Q_{i-4} = -2*Q_{i+1} - 13*Q_i + 24*Q_{i-1} - 12*Q_{i-2} + 4*Q_{i-3} + 12*dx*dQ/dx
                    Set::Scalar rho_g4 = -2.0 * rho_m1 - 13.0 * rho_bnd + 24.0 * rho_g1 - 12.0 * rho_g2 + 4.0 * rho_g3 + 12.0 * delta * drho_dx_new;
                    Set::Scalar un_g4 = -2.0 * un_m1 - 13.0 * un_bnd + 24.0 * un_g1 - 12.0 * un_g2 + 4.0 * un_g3 + 12.0 * delta * dun_dx_new;
                    Set::Scalar ut_g4 = -2.0 * ut_m1 - 13.0 * ut_bnd + 24.0 * ut_g1 - 12.0 * ut_g2 + 4.0 * ut_g3 + 12.0 * delta * dut_dx_new;
                    Set::Scalar p_g4 = -2.0 * p_m1 - 13.0 * p_bnd + 24.0 * p_g1 - 12.0 * p_g2 + 4.0 * p_g3 + 12.0 * delta * dp_dx_new;

                    rho_g4 = amrex::max(rho_g4, small_local);
                    p_g4 = amrex::max(p_g4, 1.0e-6);
                    set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, i_bnd - 4, j, k, idir, rho_g4, un_g4, ut_g4, p_g4);
                }); // end ParallelFor ghost_box
            } // end j loop
        } // end X-LO boundary

        // =====================================================================
        // X-HI BOUNDARY (RIGHT EDGE - SKIP CORNERS)
        // =====================================================================
        if (bx.bigEnd(0) == domain.bigEnd(0) && params_xhi_local.type != Type::None)
        {
            int idir = 0;  // x-direction
            int isign = 1; // right boundary
            Set::Scalar delta = dx[0];

            // Boundary index
            int i_bnd = domain.bigEnd(0);

            // Loop over j indices, EXCLUDING corners
            int j_start = domain.smallEnd(1) + 1; // Skip bottom corner
            int j_end = domain.bigEnd(1) - 1;     // Skip top corner

            for (int jj = j_start; jj <= j_end; ++jj)
            {
                // Check if this j is in our box
                if (jj < bx.smallEnd(1) || jj > bx.bigEnd(1))
                    continue;

                // Create a box for this j-slice with 4 ghost cell layers
                amrex::Box ghost_box;
                ghost_box.setSmall(0, i_bnd + 1);
                ghost_box.setBig(0, i_bnd + 4);
                ghost_box.setSmall(1, jj);
                ghost_box.setBig(1, jj);
#if AMREX_SPACEDIM == 3
                ghost_box.setSmall(2, bx.smallEnd(2));
                ghost_box.setBig(2, bx.bigEnd(2));
#endif

                amrex::ParallelFor(ghost_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    // =============================================================
                    // STEP 1: Compute primitives at boundary cell
                    // =============================================================
                    Set::Scalar rho_bnd, un_bnd, ut_bnd, p_bnd, a_bnd, gm_bnd, pi_bnd;
                    compute_prim(eta_arr, rho_arr, M_arr, E_arr, i_bnd, j, k, idir, rho_bnd, un_bnd, ut_bnd, p_bnd, a_bnd, gm_bnd, pi_bnd);

                    // =============================================================
                    // STEP 2: Compute primitives at interior neighbors (for FD)
                    // =============================================================
                    Set::Scalar rho_m1, un_m1, ut_m1, p_m1, a_m1, gm_m1, pi_m1;
                    compute_prim(eta_arr, rho_arr, M_arr, E_arr, i_bnd - 1, j, k, idir, rho_m1, un_m1, ut_m1, p_m1, a_m1, gm_m1, pi_m1);

                    Set::Scalar rho_m2, un_m2, ut_m2, p_m2, a_m2, gm_m2, pi_m2;
                    compute_prim(eta_arr, rho_arr, M_arr, E_arr, i_bnd - 2, j, k, idir, rho_m2, un_m2, ut_m2, p_m2, a_m2, gm_m2, pi_m2);

                    // =============================================================
                    // STEP 3: One-sided normal derivatives (2nd order)
                    // Motheau Eq. 40: Backward difference for right boundary
                    // df/dx = (3*f_0 - 4*f_{-1} + f_{-2}) / (2*dx)
                    // =============================================================
                    Set::Scalar drho_dx = (3.0 * rho_bnd - 4.0 * rho_m1 + rho_m2) / (2.0 * delta);
                    Set::Scalar dun_dx = (3.0 * un_bnd - 4.0 * un_m1 + un_m2) / (2.0 * delta);
                    Set::Scalar dut_dx = (3.0 * ut_bnd - 4.0 * ut_m1 + ut_m2) / (2.0 * delta);
                    Set::Scalar dp_dx = (3.0 * p_bnd - 4.0 * p_m1 + p_m2) / (2.0 * delta);

                    // =============================================================
                    // STEP 4: Tangential derivatives (centered, 2nd order)
                    // Motheau Eq. 41: Y-direction df/dy = (f_{j+1} - f_{j-1}) / (2*dy)
                    // =============================================================
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

                    // =============================================================
                    // STEP 5: LODI wave amplitudes (Motheau Eq. 23)
                    // =============================================================
                    Set::Scalar L1 = (un_bnd - a_bnd) / (2.0 * a2_bnd) * (dp_dx - rho_bnd * a_bnd * dun_dx);
                    Set::Scalar L2 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd / a2_bnd * (a2_bnd * drho_dx - dp_dx) : 0.0;
                    Set::Scalar L3 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd * dut_dx : 0.0;
                    Set::Scalar L4 = (un_bnd + a_bnd) / (2.0 * a2_bnd) * (dp_dx + rho_bnd * a_bnd * dun_dx);

                    // =============================================================
                    // STEP 6: Transverse terms (Motheau Eq. 36-37 with gamma)
                    // CORRECTED: Added gamma multiplier
                    // =============================================================
                    Set::Scalar T1 = (un_bnd - a_bnd) / (2.0 * a2_bnd) * (-rho_bnd * a_bnd * ut_bnd * dun_dy + a2_bnd * rho_bnd * dut_dy + ut_bnd * dp_dy);
                    Set::Scalar T4 = (un_bnd + a_bnd) / (2.0 * a2_bnd) * (rho_bnd * a_bnd * ut_bnd * dun_dy + a2_bnd * rho_bnd * dut_dy + ut_bnd * dp_dy);
                    Set::Scalar T2 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd * ut_bnd / a2_bnd * (a2_bnd * drho_dy - dp_dy) : 0.0;
                    Set::Scalar T3 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd * (ut_bnd * dut_dy + (1.0 / rho_bnd) * dp_dy) : 0.0;

                    // =============================================================
                    // STEP 7: Apply BC-specific modification to incoming wave
                    // For X-HI: L1 is incoming (speed un-a < 0 entering domain)
                    // =============================================================
                    Set::Scalar K = params_xhi_local.sigma * (1.0 - M_loc * M_loc) / (a_bnd * params_xhi_local.L_ref);

                    if (params_xhi_local.type == Type::Outflow)
                    {
                        L1 = K * (p_bnd - params_xhi_local.target_p) - (1.0 - params_xhi_local.beta) * T1;
                    }
                    else if (params_xhi_local.type == Type::Inflow)
                    {
                        Set::Scalar er = params_xhi_local.relax_u;
                        Set::Scalar K_acoustic = er * rho_bnd * (1.0 - M_loc * M_loc) / params_xhi_local.L_ref;
                        Set::Scalar K_entropy = er * rho_bnd * a_bnd * (gm_bnd - 1.0) / params_xhi_local.L_ref;
                        Set::Scalar K_shear = er * a_bnd / params_xhi_local.L_ref;

                        // Temperature (for entropy relaxation)
                        Set::Scalar eta_bnd = eta_arr(i_bnd, j, k);
                        Set::Scalar cv_mix = eta_bnd * cv0_ + (1.0 - eta_bnd) * cv1_;
                        Set::Scalar T_bnd = (p_bnd + pi_bnd) / (rho_bnd * cv_mix * (gm_bnd - 1.0) + small_local);

                        L1 = -K_acoustic * (un_bnd - params_xhi_local.target_u) - T1;
                        L2 = K_entropy * (T_bnd - params_xhi_local.target_T) - T2;
                        L3 = K_shear * (ut_bnd - params_xhi_local.target_v) - T3;
                    }

                    // =============================================================
                    // STEP 8: Reconstruct modified gradients (Motheau Eq. 54)
                    // =============================================================
                    Set::Scalar xi1 = L1 / (un_bnd - a_bnd + small_local);
                    Set::Scalar xi2 = (std::abs(un_bnd) > 1.0e-14) ? L2 / un_bnd : 0.0;
                    Set::Scalar xi3 = (std::abs(un_bnd) > 1.0e-14) ? L3 / un_bnd : 0.0;
                    Set::Scalar xi4 = L4 / (un_bnd + a_bnd + small_local);

                    Set::Scalar drho_dx_new = xi1 + xi2 + xi4;
                    Set::Scalar dun_dx_new = (-a_bnd / (rho_bnd + small_local)) * xi1 + (a_bnd / (rho_bnd + small_local)) * xi4;
                    Set::Scalar dut_dx_new = xi3;
                    Set::Scalar dp_dx_new = a2_bnd * (xi1 + xi4);

                    // =============================================================
                    // STEP 9: Fill ALL 4 ghost cell layers (Motheau Eq. 31-34)
                    // =============================================================

                    // Layer 1: Motheau Eq. 31
                    // Q_{i+1} = Q_{i-1} + 2*dx*dQ/dx
                    Set::Scalar rho_g1 = rho_m1 + 2.0 * delta * drho_dx_new;
                    Set::Scalar un_g1 = un_m1 + 2.0 * delta * dun_dx_new;
                    Set::Scalar ut_g1 = ut_m1 + 2.0 * delta * dut_dx_new;
                    Set::Scalar p_g1 = p_m1 + 2.0 * delta * dp_dx_new;

                    rho_g1 = amrex::max(rho_g1, small_local);
                    p_g1 = amrex::max(p_g1, 1.0e-6);
                    set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, i_bnd + 1, j, k, idir, rho_g1, un_g1, ut_g1, p_g1);

                    // Layer 2: Motheau Eq. 32
                    // Q_{i+2} = -2*Q_{i-1} - 3*Q_i + 6*Q_{i+1} - 6*dx*dQ/dx
                    Set::Scalar rho_g2 = -2.0 * rho_m1 - 3.0 * rho_bnd + 6.0 * rho_g1 - 6.0 * delta * drho_dx_new;
                    Set::Scalar un_g2 = -2.0 * un_m1 - 3.0 * un_bnd + 6.0 * un_g1 - 6.0 * delta * dun_dx_new;
                    Set::Scalar ut_g2 = -2.0 * ut_m1 - 3.0 * ut_bnd + 6.0 * ut_g1 - 6.0 * delta * dut_dx_new;
                    Set::Scalar p_g2 = -2.0 * p_m1 - 3.0 * p_bnd + 6.0 * p_g1 - 6.0 * delta * dp_dx_new;

                    rho_g2 = amrex::max(rho_g2, small_local);
                    p_g2 = amrex::max(p_g2, 1.0e-6);
                    set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, i_bnd + 2, j, k, idir, rho_g2, un_g2, ut_g2, p_g2);

                    // Layer 3: Motheau Eq. 33
                    // Q_{i+3} = 3*Q_{i-1} + 10*Q_i - 18*Q_{i+1} + 6*Q_{i+2} + 12*dx*dQ/dx
                    Set::Scalar rho_g3 = 3.0 * rho_m1 + 10.0 * rho_bnd - 18.0 * rho_g1 + 6.0 * rho_g2 + 12.0 * delta * drho_dx_new;
                    Set::Scalar un_g3 = 3.0 * un_m1 + 10.0 * un_bnd - 18.0 * un_g1 + 6.0 * un_g2 + 12.0 * delta * dun_dx_new;
                    Set::Scalar ut_g3 = 3.0 * ut_m1 + 10.0 * ut_bnd - 18.0 * ut_g1 + 6.0 * ut_g2 + 12.0 * delta * dut_dx_new;
                    Set::Scalar p_g3 = 3.0 * p_m1 + 10.0 * p_bnd - 18.0 * p_g1 + 6.0 * p_g2 + 12.0 * delta * dp_dx_new;

                    rho_g3 = amrex::max(rho_g3, small_local);
                    p_g3 = amrex::max(p_g3, 1.0e-6);
                    set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, i_bnd + 3, j, k, idir, rho_g3, un_g3, ut_g3, p_g3);

                    // Layer 4: Motheau Eq. 34
                    // Q_{i+4} = -2*Q_{i-1} - 13*Q_i + 24*Q_{i+1} - 12*Q_{i+2} + 4*Q_{i+3} - 12*dx*dQ/dx
                    Set::Scalar rho_g4 = -2.0 * rho_m1 - 13.0 * rho_bnd + 24.0 * rho_g1 - 12.0 * rho_g2 + 4.0 * rho_g3 - 12.0 * delta * drho_dx_new;
                    Set::Scalar un_g4 = -2.0 * un_m1 - 13.0 * un_bnd + 24.0 * un_g1 - 12.0 * un_g2 + 4.0 * un_g3 - 12.0 * delta * dun_dx_new;
                    Set::Scalar ut_g4 = -2.0 * ut_m1 - 13.0 * ut_bnd + 24.0 * ut_g1 - 12.0 * ut_g2 + 4.0 * ut_g3 - 12.0 * delta * dut_dx_new;
                    Set::Scalar p_g4 = -2.0 * p_m1 - 13.0 * p_bnd + 24.0 * p_g1 - 12.0 * p_g2 + 4.0 * p_g3 - 12.0 * delta * dp_dx_new;

                    rho_g4 = amrex::max(rho_g4, small_local);
                    p_g4 = amrex::max(p_g4, 1.0e-6);
                    set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, i_bnd + 4, j, k, idir, rho_g4, un_g4, ut_g4, p_g4);
                }); // end ParallelFor ghost_box
            } // end j loop
        } // end X-HI boundary

        // =====================================================================
        // Y-LO BOUNDARY (BOTTOM EDGE - SKIP CORNERS)
        // =====================================================================
        if (bx.smallEnd(1) == domain.smallEnd(1) && params_ylo_local.type != Type::None)
        {
            int idir = 1;   // y-direction
            int isign = -1; // bottom boundary
            Set::Scalar delta = dx[1];

            // Boundary index
            int j_bnd = domain.smallEnd(1);

            // Loop over i indices, EXCLUDING corners
            int i_start = domain.smallEnd(0) + 1; // Skip left corner
            int i_end = domain.bigEnd(0) - 1;     // Skip right corner

            for (int ii = i_start; ii <= i_end; ++ii)
            {
                // Check if this i is in our box
                if (ii < bx.smallEnd(0) || ii > bx.bigEnd(0))
                    continue;

                // Create a box for this i-slice with 4 ghost cell layers
                amrex::Box ghost_box;
                ghost_box.setSmall(0, ii);
                ghost_box.setBig(0, ii);
                ghost_box.setSmall(1, j_bnd - 4);
                ghost_box.setBig(1, j_bnd - 1);
#if AMREX_SPACEDIM == 3
                ghost_box.setSmall(2, bx.smallEnd(2));
                ghost_box.setBig(2, bx.bigEnd(2));
#endif

                amrex::ParallelFor(ghost_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    // =============================================================
                    // STEP 1: Compute primitives at boundary cell
                    // =============================================================
                    Set::Scalar rho_bnd, un_bnd, ut_bnd, p_bnd, a_bnd, gm_bnd, pi_bnd;
                    compute_prim(eta_arr, rho_arr, M_arr, E_arr, i, j_bnd, k, idir, rho_bnd, un_bnd, ut_bnd, p_bnd, a_bnd, gm_bnd, pi_bnd);

                    // =============================================================
                    // STEP 2: Compute primitives at interior neighbors (for FD)
                    // =============================================================
                    Set::Scalar rho_m1, un_m1, ut_m1, p_m1, a_m1, gm_m1, pi_m1;
                    compute_prim(eta_arr, rho_arr, M_arr, E_arr, i, j_bnd + 1, k, idir, rho_m1, un_m1, ut_m1, p_m1, a_m1, gm_m1, pi_m1);

                    Set::Scalar rho_m2, un_m2, ut_m2, p_m2, a_m2, gm_m2, pi_m2;
                    compute_prim(eta_arr, rho_arr, M_arr, E_arr, i, j_bnd + 2, k, idir, rho_m2, un_m2, ut_m2, p_m2, a_m2, gm_m2, pi_m2);

                    // =============================================================
                    // STEP 3: One-sided normal derivatives (2nd order)
                    // Motheau Eq. 39: Forward difference for bottom boundary
                    // df/dy = (-3*f_0 + 4*f_1 - f_2) / (2*dy)
                    // =============================================================
                    Set::Scalar drho_dy = (-3.0 * rho_bnd + 4.0 * rho_m1 - rho_m2) / (2.0 * delta);
                    Set::Scalar dun_dy = (-3.0 * un_bnd + 4.0 * un_m1 - un_m2) / (2.0 * delta);
                    Set::Scalar dut_dy = (-3.0 * ut_bnd + 4.0 * ut_m1 - ut_m2) / (2.0 * delta);
                    Set::Scalar dp_dy = (-3.0 * p_bnd + 4.0 * p_m1 - p_m2) / (2.0 * delta);

                    // =============================================================
                    // STEP 4: Tangential derivatives (centered, 2nd order)
                    // Motheau Eq. 41: X-direction df/dx = (f_{i+1} - f_{i-1}) / (2*dx)
                    // =============================================================
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

                    // =============================================================
                    // STEP 5: LODI wave amplitudes (Motheau Eq. 23)
                    // For Y-direction: un=vy, ut=vx
                    // CORRECTED: L2 is entropy, L3 is vorticity
                    // =============================================================
                    Set::Scalar L1 = (un_bnd - a_bnd) / (2.0 * a2_bnd) * (dp_dy - rho_bnd * a_bnd * dun_dy);
                    Set::Scalar L2 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd / a2_bnd * (a2_bnd * drho_dy - dp_dy) : 0.0;
                    Set::Scalar L3 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd * dut_dy : 0.0;
                    Set::Scalar L4 = (un_bnd + a_bnd) / (2.0 * a2_bnd) * (dp_dy + rho_bnd * a_bnd * dun_dy);

                    // =============================================================
                    // STEP 6: Transverse terms (Motheau Eq. 36-37 with gamma)
                    // For Y-normal: transverse = X-direction
                    // CORRECTED: Added gamma multiplier
                    // =============================================================
                    Set::Scalar T1 = (un_bnd - a_bnd) / (2.0 * a2_bnd) * (-rho_bnd * a_bnd * ut_bnd * dun_dx + a2_bnd * rho_bnd * dut_dx + ut_bnd * dp_dx);
                    Set::Scalar T4 = (un_bnd + a_bnd) / (2.0 * a2_bnd) * (rho_bnd * a_bnd * ut_bnd * dun_dx + a2_bnd * rho_bnd * dut_dx + ut_bnd * dp_dx);
                    Set::Scalar T2 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd * ut_bnd / a2_bnd * (a2_bnd * drho_dx - dp_dx) : 0.0;
                    Set::Scalar T3 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd * (ut_bnd * dut_dx + (1.0 / rho_bnd) * dp_dx) : 0.0;

                    // =============================================================
                    // STEP 7: Apply BC-specific modification to incoming wave
                    // For Y-LO: L4 is incoming (speed un+a > 0 entering domain)
                    // =============================================================
                    Set::Scalar K = params_ylo_local.sigma * (1.0 - M_loc * M_loc) / (a_bnd * params_ylo_local.L_ref);

                    if (params_ylo_local.type == Type::Outflow)
                    {
                        L4 = K * (p_bnd - params_ylo_local.target_p) - (1.0 - params_ylo_local.beta) * T4;
                    }
                    else if (params_ylo_local.type == Type::Inflow)
                    {
                        Set::Scalar er = params_ylo_local.relax_u;
                        Set::Scalar K_acoustic = er * rho_bnd * (1.0 - M_loc * M_loc) / params_ylo_local.L_ref;
                        Set::Scalar K_entropy = er * rho_bnd * a_bnd * (gm_bnd - 1.0) / params_ylo_local.L_ref;
                        Set::Scalar K_shear = er * a_bnd / params_ylo_local.L_ref;

                        // Temperature
                        Set::Scalar eta_bnd = eta_arr(i, j_bnd, k);
                        Set::Scalar cv_mix = eta_bnd * cv0_ + (1.0 - eta_bnd) * cv1_;
                        Set::Scalar T_bnd = (p_bnd + pi_bnd) / (rho_bnd * cv_mix * (gm_bnd - 1.0) + small_local);

                        L4 = K_acoustic * (un_bnd - params_ylo_local.target_v) - T4;
                        L2 = K_entropy * (T_bnd - params_ylo_local.target_T) - T2;
                        L3 = K_shear * (ut_bnd - params_ylo_local.target_u) - T3;
                    }

                    // =============================================================
                    // STEP 8: Reconstruct modified gradients (Motheau Eq. 54)
                    // =============================================================
                    Set::Scalar xi1 = L1 / (un_bnd - a_bnd + small_local);
                    Set::Scalar xi2 = (std::abs(un_bnd) > 1.0e-14) ? L2 / un_bnd : 0.0;
                    Set::Scalar xi3 = (std::abs(un_bnd) > 1.0e-14) ? L3 / un_bnd : 0.0;
                    Set::Scalar xi4 = L4 / (un_bnd + a_bnd + small_local);

                    Set::Scalar drho_dy_new = xi1 + xi2 + xi4;
                    Set::Scalar dun_dy_new = (-a_bnd / (rho_bnd + small_local)) * xi1 + (a_bnd / (rho_bnd + small_local)) * xi4;
                    Set::Scalar dut_dy_new = xi3;
                    Set::Scalar dp_dy_new = a2_bnd * (xi1 + xi4);

                    // =============================================================
                    // STEP 9: Fill ALL 4 ghost cell layers (Motheau Eq. 27-30)
                    // =============================================================

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

                    // Layer 3: Motheau Eq. 29
                    Set::Scalar rho_g3 = 3.0 * rho_m1 + 10.0 * rho_bnd - 18.0 * rho_g1 + 6.0 * rho_g2 - 12.0 * delta * drho_dy_new;
                    Set::Scalar un_g3 = 3.0 * un_m1 + 10.0 * un_bnd - 18.0 * un_g1 + 6.0 * un_g2 - 12.0 * delta * dun_dy_new;
                    Set::Scalar ut_g3 = 3.0 * ut_m1 + 10.0 * ut_bnd - 18.0 * ut_g1 + 6.0 * ut_g2 - 12.0 * delta * dut_dy_new;
                    Set::Scalar p_g3 = 3.0 * p_m1 + 10.0 * p_bnd - 18.0 * p_g1 + 6.0 * p_g2 - 12.0 * delta * dp_dy_new;

                    rho_g3 = amrex::max(rho_g3, small_local);
                    p_g3 = amrex::max(p_g3, 1.0e-6);
                    set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, i, j_bnd - 3, k, idir, rho_g3, un_g3, ut_g3, p_g3);

                    // Layer 4: Motheau Eq. 30
                    Set::Scalar rho_g4 = -2.0 * rho_m1 - 13.0 * rho_bnd + 24.0 * rho_g1 - 12.0 * rho_g2 + 4.0 * rho_g3 + 12.0 * delta * drho_dy_new;
                    Set::Scalar un_g4 = -2.0 * un_m1 - 13.0 * un_bnd + 24.0 * un_g1 - 12.0 * un_g2 + 4.0 * un_g3 + 12.0 * delta * dun_dy_new;
                    Set::Scalar ut_g4 = -2.0 * ut_m1 - 13.0 * ut_bnd + 24.0 * ut_g1 - 12.0 * ut_g2 + 4.0 * ut_g3 + 12.0 * delta * dun_dy_new;
                    Set::Scalar p_g4 = -2.0 * p_m1 - 13.0 * p_bnd + 24.0 * p_g1 - 12.0 * p_g2 + 4.0 * p_g3 + 12.0 * delta * dp_dy_new;

                    rho_g4 = amrex::max(rho_g4, small_local);
                    p_g4 = amrex::max(p_g4, 1.0e-6);
                    set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, i, j_bnd - 4, k, idir, rho_g4, un_g4, ut_g4, p_g4);
                }); // end ParallelFor ghost_box
            } // end i loop
        } // end Y-LO boundary

        // =====================================================================
        // Y-HI BOUNDARY (TOP EDGE - SKIP CORNERS)
        // =====================================================================
        if (bx.bigEnd(1) == domain.bigEnd(1) && params_yhi_local.type != Type::None)
        {
            int idir = 1;  // y-direction
            int isign = 1; // top boundary
            Set::Scalar delta = dx[1];

            // Boundary index
            int j_bnd = domain.bigEnd(1);

            // Loop over i indices, EXCLUDING corners
            int i_start = domain.smallEnd(0) + 1; // Skip left corner
            int i_end = domain.bigEnd(0) - 1;     // Skip right corner

            for (int ii = i_start; ii <= i_end; ++ii)
            {
                // Check if this i is in our box
                if (ii < bx.smallEnd(0) || ii > bx.bigEnd(0))
                    continue;

                // Create a box for this i-slice with 4 ghost cell layers
                amrex::Box ghost_box;
                ghost_box.setSmall(0, ii);
                ghost_box.setBig(0, ii);
                ghost_box.setSmall(1, j_bnd + 1);
                ghost_box.setBig(1, j_bnd + 4);
#if AMREX_SPACEDIM == 3
                ghost_box.setSmall(2, bx.smallEnd(2));
                ghost_box.setBig(2, bx.bigEnd(2));
#endif

                amrex::ParallelFor(ghost_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    // =============================================================
                    // STEP 1: Compute primitives at boundary cell
                    // =============================================================
                    Set::Scalar rho_bnd, un_bnd, ut_bnd, p_bnd, a_bnd, gm_bnd, pi_bnd;
                    compute_prim(eta_arr, rho_arr, M_arr, E_arr, i, j_bnd, k, idir, rho_bnd, un_bnd, ut_bnd, p_bnd, a_bnd, gm_bnd, pi_bnd);

                    // =============================================================
                    // STEP 2: Compute primitives at interior neighbors (for FD)
                    // =============================================================
                    Set::Scalar rho_m1, un_m1, ut_m1, p_m1, a_m1, gm_m1, pi_m1;
                    compute_prim(eta_arr, rho_arr, M_arr, E_arr, i, j_bnd - 1, k, idir, rho_m1, un_m1, ut_m1, p_m1, a_m1, gm_m1, pi_m1);

                    Set::Scalar rho_m2, un_m2, ut_m2, p_m2, a_m2, gm_m2, pi_m2;
                    compute_prim(eta_arr, rho_arr, M_arr, E_arr, i, j_bnd - 2, k, idir, rho_m2, un_m2, ut_m2, p_m2, a_m2, gm_m2, pi_m2);

                    // =============================================================
                    // STEP 3: One-sided normal derivatives (2nd order)
                    // Motheau Eq. 40: Backward difference for top boundary
                    // df/dy = (3*f_0 - 4*f_{-1} + f_{-2}) / (2*dy)
                    // =============================================================
                    Set::Scalar drho_dy = (3.0 * rho_bnd - 4.0 * rho_m1 + rho_m2) / (2.0 * delta);
                    Set::Scalar dun_dy = (3.0 * un_bnd - 4.0 * un_m1 + un_m2) / (2.0 * delta);
                    Set::Scalar dut_dy = (3.0 * ut_bnd - 4.0 * ut_m1 + ut_m2) / (2.0 * delta);
                    Set::Scalar dp_dy = (3.0 * p_bnd - 4.0 * p_m1 + p_m2) / (2.0 * delta);

                    // =============================================================
                    // STEP 4: Tangential derivatives (centered, 2nd order)
                    // Motheau Eq. 41: X-direction df/dx = (f_{i+1} - f_{i-1}) / (2*dx)
                    // =============================================================
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

                    // =============================================================
                    // STEP 5: LODI wave amplitudes (Motheau Eq. 23)
                    // For Y-direction: un=vy, ut=vx
                    // CORRECTED: L2 is entropy, L3 is vorticity
                    // =============================================================
                    Set::Scalar L1 = (un_bnd - a_bnd) / (2.0 * a2_bnd) * (dp_dy - rho_bnd * a_bnd * dun_dy);
                    Set::Scalar L2 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd / a2_bnd * (a2_bnd * drho_dy - dp_dy) : 0.0;
                    Set::Scalar L3 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd * dut_dy : 0.0;
                    Set::Scalar L4 = (un_bnd + a_bnd) / (2.0 * a2_bnd) * (dp_dy + rho_bnd * a_bnd * dun_dy);

                    // =============================================================
                    // STEP 6: Transverse terms (Motheau Eq. 36-37 with gamma)
                    // For Y-normal: transverse = X-direction
                    // CORRECTED: Added gamma multiplier
                    // =============================================================
                    Set::Scalar T1 = (un_bnd - a_bnd) / (2.0 * a2_bnd) * (-rho_bnd * a_bnd * ut_bnd * dun_dx + a2_bnd * rho_bnd * dut_dx + ut_bnd * dp_dx);
                    Set::Scalar T4 = (un_bnd + a_bnd) / (2.0 * a2_bnd) * (rho_bnd * a_bnd * ut_bnd * dun_dx + a2_bnd * rho_bnd * dut_dx + ut_bnd * dp_dx);
                    Set::Scalar T2 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd * ut_bnd / a2_bnd * (a2_bnd * drho_dx - dp_dx) : 0.0;
                    Set::Scalar T3 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd * (ut_bnd * dut_dx + (1.0 / rho_bnd) * dp_dx) : 0.0;

                    // =============================================================
                    // STEP 7: Apply BC-specific modification to incoming wave
                    // For Y-HI: L1 is incoming (speed un-a < 0 entering domain)
                    // =============================================================
                    Set::Scalar K = params_yhi_local.sigma * (1.0 - M_loc * M_loc) / (a_bnd * params_yhi_local.L_ref);

                    if (params_yhi_local.type == Type::Outflow)
                    {
                        L1 = K * (p_bnd - params_yhi_local.target_p) - (1.0 - params_yhi_local.beta) * T1;
                    }
                    else if (params_yhi_local.type == Type::Inflow)
                    {
                        Set::Scalar er = params_yhi_local.relax_u;
                        Set::Scalar K_acoustic = er * rho_bnd * (1.0 - M_loc * M_loc) / params_yhi_local.L_ref;
                        Set::Scalar K_entropy = er * rho_bnd * a_bnd * (gm_bnd - 1.0) / params_yhi_local.L_ref;
                        Set::Scalar K_shear = er * a_bnd / params_yhi_local.L_ref;

                        // Temperature
                        Set::Scalar eta_bnd = eta_arr(i, j_bnd, k);
                        Set::Scalar cv_mix = eta_bnd * cv0_ + (1.0 - eta_bnd) * cv1_;
                        Set::Scalar T_bnd = (p_bnd + pi_bnd) / (rho_bnd * cv_mix * (gm_bnd - 1.0) + small_local);

                        L1 = -K_acoustic * (un_bnd - params_yhi_local.target_v) - T1;
                        L2 = K_entropy * (T_bnd - params_yhi_local.target_T) - T2;
                        L3 = K_shear * (ut_bnd - params_yhi_local.target_u) - T3;
                    }

                    // =============================================================
                    // STEP 8: Reconstruct modified gradients (Motheau Eq. 54)
                    // =============================================================
                    Set::Scalar xi1 = L1 / (un_bnd - a_bnd + small_local);
                    Set::Scalar xi2 = (std::abs(un_bnd) > 1.0e-14) ? L2 / un_bnd : 0.0;
                    Set::Scalar xi3 = (std::abs(un_bnd) > 1.0e-14) ? L3 / un_bnd : 0.0;
                    Set::Scalar xi4 = L4 / (un_bnd + a_bnd + small_local);

                    Set::Scalar drho_dy_new = xi1 + xi2 + xi4;
                    Set::Scalar dun_dy_new = (-a_bnd / (rho_bnd + small_local)) * xi1 + (a_bnd / (rho_bnd + small_local)) * xi4;
                    Set::Scalar dut_dy_new = xi3;
                    Set::Scalar dp_dy_new = a2_bnd * (xi1 + xi4);

                    // =============================================================
                    // STEP 9: Fill ALL 4 ghost cell layers (Motheau Eq. 31-34)
                    // =============================================================

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

                    // Layer 3: Motheau Eq. 33
                    Set::Scalar rho_g3 = 3.0 * rho_m1 + 10.0 * rho_bnd - 18.0 * rho_g1 + 6.0 * rho_g2 + 12.0 * delta * drho_dy_new;
                    Set::Scalar un_g3 = 3.0 * un_m1 + 10.0 * un_bnd - 18.0 * un_g1 + 6.0 * un_g2 + 12.0 * delta * dun_dy_new;
                    Set::Scalar ut_g3 = 3.0 * ut_m1 + 10.0 * ut_bnd - 18.0 * ut_g1 + 6.0 * ut_g2 + 12.0 * delta * dut_dy_new;
                    Set::Scalar p_g3 = 3.0 * p_m1 + 10.0 * p_bnd - 18.0 * p_g1 + 6.0 * p_g2 + 12.0 * delta * dp_dy_new;

                    rho_g3 = amrex::max(rho_g3, small_local);
                    p_g3 = amrex::max(p_g3, 1.0e-6);
                    set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, i, j_bnd + 3, k, idir, rho_g3, un_g3, ut_g3, p_g3);

                    // Layer 4: Motheau Eq. 34
                    Set::Scalar rho_g4 = -2.0 * rho_m1 - 13.0 * rho_bnd + 24.0 * rho_g1 - 12.0 * rho_g2 + 4.0 * rho_g3 - 12.0 * delta * drho_dy_new;
                    Set::Scalar un_g4 = -2.0 * un_m1 - 13.0 * un_bnd + 24.0 * un_g1 - 12.0 * un_g2 + 4.0 * un_g3 - 12.0 * delta * dun_dy_new;
                    Set::Scalar ut_g4 = -2.0 * ut_m1 - 13.0 * ut_bnd + 24.0 * ut_g1 - 12.0 * ut_g2 + 4.0 * ut_g3 - 12.0 * delta * dut_dy_new;
                    Set::Scalar p_g4 = -2.0 * p_m1 - 13.0 * p_bnd + 24.0 * p_g1 - 12.0 * p_g2 + 4.0 * p_g3 - 12.0 * delta * dp_dy_new;

                    rho_g4 = amrex::max(rho_g4, small_local);
                    p_g4 = amrex::max(p_g4, 1.0e-6);
                    set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, i, j_bnd + 4, k, idir, rho_g4, un_g4, ut_g4, p_g4);
                }); // end ParallelFor ghost_box
            } // end i loop
        } // end Y-HI boundary

        } // end MFIter loop

    } // end FillBoundary()

private:
    /// Parse parameters for a single boundary face
    static BoundaryParams ParseFace(IO::ParmParse &pp, std::string face_name);

}; // end class NSCBC

} // namespace BC

#endif // BC_NSCBC_H
