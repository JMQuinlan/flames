#ifndef BC_NSCBC4_H
#define BC_NSCBC4_H

#include "BC/BC.H"
#include "IO/ParmParse.H"
#include "Set/Set.H"
#include "Solver/EOS/EOS.H"
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

//EOS
#include "Solver/EOS/EOS.H"
#include "Solver/EOS/Tammann.H"
#include "Solver/EOS/CPG.H"

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

class NSCBC4
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

        // Time-dependent pressure drive (acoustic forcing through the
        // incoming characteristic -- the well-posed replacement for the
        // primitive-Dirichlet drive, which rectified band acoustics into
        // mass injection at any inflow-capable zero-gradient face):
        //   p_target_eff(t) = target_p + drive_amp * sin(drive_omega*t + drive_phase)
        // Outgoing waves still leave through the LODI outflow treatment, so
        // the face drives WITHOUT acting as a resonator wall.
        Set::Scalar drive_amp   = 0.0; ///< Drive amplitude [Pa] (0 = static target)
        Set::Scalar drive_omega = 0.0; ///< Drive angular frequency [rad/s]
        Set::Scalar drive_phase = 0.0; ///< Drive phase [rad]

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
    NSCBC4() = default;

    /// Constructor with input file parsing
    NSCBC4(IO::ParmParse &pp, std::string prefix = "nscbc4");

    /// Parse boundary parameters from input file
    static void Parse(NSCBC4 &value, IO::ParmParse &pp);

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
    /// @param gamma_mf      Gamma field (written to ghost cells for Riemann solver)
    /// @param pi_mf         Pi (Tammann p0) field (written to ghost cells)
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

        // =========================================================================
        // CAPTURE EOS PARAMETERS FOR GPU
        // =========================================================================
        // (gamma/p0 captures removed - all EOS calls go through Solver::EOS::EOS::*)
        Set::Scalar cv0_ = eos0.Cv();
        Set::Scalar cv1_ = eos1.Cv();
        Set::Scalar pref_local = pref;
        Set::Scalar small_local = small;

        // Capture boundary parameters for GPU
        BoundaryParams params_xlo_local = params_xlo;
        BoundaryParams params_xhi_local = params_xhi;
        BoundaryParams params_ylo_local = params_ylo;
        BoundaryParams params_yhi_local = params_yhi;
        // Apply the time-dependent pressure drive to the per-call effective
        // targets (everything downstream -- faces, edges, corners -- reads
        // these locals, so one application covers all use sites).
        auto apply_drive = [time](BoundaryParams &prm) {
            prm.target_p += prm.drive_amp * std::sin(prm.drive_omega * time + prm.drive_phase);
        };
        apply_drive(params_xlo_local);
        apply_drive(params_xhi_local);
        apply_drive(params_ylo_local);
        apply_drive(params_yhi_local);
#if AMREX_SPACEDIM == 3
        BoundaryParams params_zlo_local = params_zlo;
        BoundaryParams params_zhi_local = params_zhi;
        apply_drive(params_zlo_local);
        apply_drive(params_zhi_local);
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

            // EOS
            Set::Scalar gamma_mix = Solver::EOS::EOS::MixedGamma(eta_val, eos0, eos1);
            Set::Scalar pi_mix = Solver::EOS::EOS::MixedP0(eta_val, eos0, eos1);

            // Pressure (Tammann EOS)
            Set::Scalar p_val = Solver::EOS::EOS::MixedPressure(rho_val, UE, eta_val, eos0, eos1, pref, small_local);

            // Sound speed
            Set::Scalar a_val = Solver::EOS::EOS::TammannSoundSpeed(rho_val, p_val, gamma_mix, pi_mix, small_local);

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

            // EOS quantities via the EOS::EOS class (consistent with compute_prim)
            Set::Scalar gamma_mix = Solver::EOS::EOS::MixedGamma(eta_val, eos0, eos1);
            Set::Scalar pi_mix    = Solver::EOS::EOS::MixedP0(eta_val, eos0, eos1);
            Set::Scalar UE        = Solver::EOS::EOS::MixedInternalEnergy(p_val, eta_val, eos0, eos1, pref_local, small_local);

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

            // Write gamma, pi, pressure for the Riemann solver to read at flux time
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

            // Get Array4 views of conserved variables
            amrex::Array4<const amrex::Real> const &eta_arr = eta_mf.const_array(mfi);
            amrex::Array4<amrex::Real> const &rho_arr = density_mf.array(mfi);
            amrex::Array4<amrex::Real> const &M_arr = momentum_mf.array(mfi);
            amrex::Array4<amrex::Real> const &E_arr = energy_mf.array(mfi);
            amrex::Array4<amrex::Real> const &gamma_arr = gamma_mf.array(mfi);
            amrex::Array4<amrex::Real> const &pi_arr = pi_mf.array(mfi);
            amrex::Array4<amrex::Real> const &p_arr = pressure_mf.array(mfi);

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
                    // Clamp to THIS box's z-range (validbox), not the full domain:
                    // with MPI z-decomposition a corner-touching box covers only
                    // part of z, and writing the full domain z-range would run off
                    // the FAB end -> heap corruption.  (The z-parallel edge is the
                    // union over all z-boxes.)
                    corner_box.setSmall(2, bx.smallEnd(2));
                    corner_box.setBig(2, bx.bigEnd(2));
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
#if AMREX_SPACEDIM == 3
                        // z-velocity is tangential to this (x,y) edge (the edge runs
                        // along z); extrapolate it as a zero-gradient passive scalar.
                        const Set::Scalar vz_c = M_arr(ic, jc, k, 2) / (rho_c + small_local);
#endif

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
                        // CORRECTED: Removed extra division factors
                        // L_1 = (u - c) * (dp/dx - rho*c*du/dx)
                        // L_2 = u * (c^2*drho/dx - dp/dx)
                        // L_3 = u * dv/dx
                        // L_4 = (u + c) * (dp/dx + rho*c*du/dx)
                        // =============================================================
                        Set::Scalar L1_x = (vx_c - a_c) * (dp_dx - rho_c * a_c * dvx_dx);
                        Set::Scalar L2_x = (std::abs(vx_c) > 1.0e-14) ? vx_c * (a2_c * drho_dx - dp_dx) : 0.0;
                        Set::Scalar L3_x = (std::abs(vx_c) > 1.0e-14) ? vx_c * dvy_dx : 0.0;
                        Set::Scalar L4_x = (vx_c + a_c) * (dp_dx + rho_c * a_c * dvx_dx);

                        // =============================================================
                        // STEP 5: LODI waves in Y-direction (Motheau Eq. 23)
                        // CORRECTED: Removed extra division factors
                        // =============================================================
                        Set::Scalar L1_y = (vy_c - a_c) * (dp_dy - rho_c * a_c * dvy_dy);
                        Set::Scalar L2_y = (std::abs(vy_c) > 1.0e-14) ? vy_c * (a2_c * drho_dy - dp_dy) : 0.0;
                        Set::Scalar L3_y = (std::abs(vy_c) > 1.0e-14) ? vy_c * dvx_dy : 0.0;
                        Set::Scalar L4_y = (vy_c + a_c) * (dp_dy + rho_c * a_c * dvy_dy);

                        // =============================================================
                        // STEP 6: Solve 2x2 coupled system for incoming waves
                        // (Motheau Eq. 56-58)
                        // =============================================================
                        Set::Scalar Lx_in = (cx == 1) ? L1_x : L4_x; // Incoming wave in X
                        Set::Scalar Ly_in = (cy == 1) ? L1_y : L4_y; // Incoming wave in Y

                        // Relaxation coefficients (Motheau Eq. 38)
                        // Motheau 2017 Eq. 38: K = sigma * c * (1 - M^2) / L
                        // (matches the four edge K formulas; original divided by a_c
                        // which made the corner relaxation 1/a^2 too weak vs the paper)
                        Set::Scalar Kx = sigma_x * (1.0 - Mx_c * Mx_c) * a_c / Lx_ref;
                        Set::Scalar Ky = sigma_y * (1.0 - My_c * My_c) * a_c / Ly_ref;

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
                        // CORRECTED: Account for removed division factors in L
                        // =============================================================
                        // X-direction gradients
                        Set::Scalar xi1_x = L1_x / (vx_c - a_c + small_local);
                        Set::Scalar xi2_x = (std::abs(vx_c) > 1.0e-14) ? L2_x / vx_c : 0.0;
                        Set::Scalar xi3_x = (std::abs(vx_c) > 1.0e-14) ? L3_x / vx_c : 0.0;
                        Set::Scalar xi4_x = L4_x / (vx_c + a_c + small_local);

                        // Motheau Eq. 26 inversion for paper-style L (= lambda * inner_factor):
                        //   dp/dn  = 0.5 * (xi1 + xi4)
                        //   du/dn  = (xi4 - xi1) / (2 * rho * a)
                        //   drho/dn = (xi2 + 0.5*(xi1+xi4)) / a^2
                        //   dv/dn  = xi3
                        // (Original implementation gave gradients off by 2*a^2, causing
                        //  the boundary relaxation to be wildly over-aggressive.)
                        Set::Scalar dp_dx_new   = 0.5 * (xi1_x + xi4_x);
                        Set::Scalar dvx_dx_new  = (xi4_x - xi1_x) / (2.0 * (rho_c + small_local) * a_c + small_local);
                        Set::Scalar drho_dx_new = (xi2_x + 0.5 * (xi1_x + xi4_x)) / (a2_c + small_local);
                        Set::Scalar dvy_dx_new  = xi3_x;

                        // Y-direction gradients
                        Set::Scalar xi1_y = L1_y / (vy_c - a_c + small_local);
                        Set::Scalar xi2_y = (std::abs(vy_c) > 1.0e-14) ? L2_y / vy_c : 0.0;
                        Set::Scalar xi3_y = (std::abs(vy_c) > 1.0e-14) ? L3_y / vy_c : 0.0;
                        Set::Scalar xi4_y = L4_y / (vy_c + a_c + small_local);

                        Set::Scalar dp_dy_new   = 0.5 * (xi1_y + xi4_y);
                        Set::Scalar dvy_dy_new  = (xi4_y - xi1_y) / (2.0 * (rho_c + small_local) * a_c + small_local);
                        Set::Scalar drho_dy_new = (xi2_y + 0.5 * (xi1_y + xi4_y)) / (a2_c + small_local);
                        Set::Scalar dvx_dy_new  = xi3_y;

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
                            Set::Scalar KE_1 = 0.5 * rho_1 * (vx_1 * vx_1 + vy_1 * vy_1);
                            Set::Scalar UE_1 = E_arr(i1, j1, k) - KE_1;
                            Set::Scalar p_1 = Solver::EOS::EOS::MixedPressure(rho_1, UE_1, eta_1, eos0, eos1, pref_local, small_local);

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
                            Set::Scalar KE_1 = 0.5 * rho_1 * (vx_1 * vx_1 + vy_1 * vy_1);
                            Set::Scalar UE_1 = E_arr(i1, j1, k) - KE_1;
                            Set::Scalar p_1 = Solver::EOS::EOS::MixedPressure(rho_1, UE_1, eta_1, eos0, eos1, pref_local, small_local);


                            Set::Scalar eta_2 = eta_arr(i2, j2, k);
                            Set::Scalar KE_2 = 0.5 * rho_2 * (vx_2 * vx_2 + vy_2 * vy_2);
                            Set::Scalar UE_2 = E_arr(i2, j2, k) - KE_2;
                            Set::Scalar p_2 = Solver::EOS::EOS::MixedPressure(rho_2, UE_2, eta_2, eos0, eos1, pref_local, small_local);

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
                            Set::Scalar KE_1 = 0.5 * rho_1 * (vx_1 * vx_1 + vy_1 * vy_1);
                            Set::Scalar UE_1 = E_arr(i1, j1, k) - KE_1;
                            Set::Scalar p_1 = Solver::EOS::EOS::MixedPressure(rho_1, UE_1, eta_1, eos0, eos1, pref_local, small_local);

                            Set::Scalar eta_2 = eta_arr(i2, j2, k);
                            Set::Scalar KE_2 = 0.5 * rho_2 * (vx_2 * vx_2 + vy_2 * vy_2);
                            Set::Scalar UE_2 = E_arr(i2, j2, k) - KE_2;
                            Set::Scalar p_2 = Solver::EOS::EOS::MixedPressure(rho_2, UE_2, eta_2, eos0, eos1, pref_local, small_local);

                            Set::Scalar eta_3 = eta_arr(i3, j3, k);
                            Set::Scalar KE_3 = 0.5 * rho_3 * (vx_3 * vx_3 + vy_3 * vy_3);
                            Set::Scalar UE_3 = E_arr(i3, j3, k) - KE_3;
                            Set::Scalar p_3 = Solver::EOS::EOS::MixedPressure(rho_3, UE_3, eta_3, eos0, eos1, pref_local, small_local);

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
                        rho_g = amrex::min(amrex::max(rho_g, pref_local), 1.0e20);
                        p_g = amrex::min(amrex::max(p_g, 1.0), 1.0e20);
                        vx_g = amrex::min(amrex::max(vx_g, -1.0e20), 1.0e20);
                        vy_g = amrex::min(amrex::max(vy_g, -1.0e20), 1.0e20);

                        // =============================================================
                        // STEP 10: Write to conserved arrays
                        // EOS quantities via the EOS::EOS class (consistent with set_ghost_cons)
                        // =============================================================
                        Set::Scalar eta_g = eta_arr(i, j, k);
                        Set::Scalar gm_g  = Solver::EOS::EOS::MixedGamma(eta_g, eos0, eos1);
                        Set::Scalar pi_g  = Solver::EOS::EOS::MixedP0(eta_g, eos0, eos1);
                        Set::Scalar UE_g  = Solver::EOS::EOS::MixedInternalEnergy(p_g, eta_g, eos0, eos1, pref_local, small_local);
                        if (UE_g < 0.0)
                            UE_g = 0.0;

#if AMREX_SPACEDIM == 3
                        Set::Scalar KE_g = 0.5 * rho_g * (vx_g * vx_g + vy_g * vy_g + vz_c * vz_c);
#else
                        Set::Scalar KE_g = 0.5 * rho_g * (vx_g * vx_g + vy_g * vy_g);
#endif

                        rho_arr(i, j, k) = rho_g;
                        M_arr(i, j, k, 0) = rho_g * vx_g;
                        M_arr(i, j, k, 1) = rho_g * vy_g;
#if AMREX_SPACEDIM == 3
                        M_arr(i, j, k, 2) = rho_g * vz_c;
#endif
                        E_arr(i, j, k) = UE_g + KE_g;

                        // Write gamma, pi, pressure for the Riemann solver to read at flux time
                        gamma_arr(i, j, k) = gm_g;
                        pi_arr(i, j, k) = pi_g;
                        p_arr(i, j, k) = p_g;
                    }); // end ParallelFor corner_box

                } // end corner loops (cx, cy)

#if AMREX_SPACEDIM == 3
            // =====================================================================
            // STEP 1b: 3D EDGES in the (x,z) and (y,z) planes.
            // =====================================================================
            // The (x,y) plane (z-parallel edges) is handled by the corner block
            // above. Here we treat the remaining 8 edges: the (x,z) plane
            // (y-parallel, 4 edges) and the (y,z) plane (x-parallel, 4 edges).
            // Each edge is a 2x2-coupled LODI in its two boundary-normal
            // directions (d1,d2); the edge-tangential velocity (dt) is carried as
            // a zero-gradient passive scalar. Ghosts use linear, gradient-corrected
            // extrapolation in the (d1,d2) ghost quadrant (the d1,d2 gradients are
            // the LODI-reconstructed ones). dt range = valid only, so the 3D
            // corners (all three ghost) are excluded and handled separately.
            // =====================================================================
            {
                const int edge_planes[2][3] = { {0, 2, 1}, {1, 2, 0} }; // {d1,d2,dt}
                for (int ep = 0; ep < 2; ++ep)
                {
                    const int d1 = edge_planes[ep][0];
                    const int d2 = edge_planes[ep][1];
                    const int dt = edge_planes[ep][2];
                    BoundaryParams P1lo = (d1 == 0) ? params_xlo_local : params_ylo_local;
                    BoundaryParams P1hi = (d1 == 0) ? params_xhi_local : params_yhi_local;
                    BoundaryParams P2lo = params_zlo_local;
                    BoundaryParams P2hi = params_zhi_local;

                    for (int c1 = 0; c1 < 2; ++c1)
                        for (int c2 = 0; c2 < 2; ++c2)
                        {
                            BoundaryParams p1 = (c1 == 0) ? P1lo : P1hi;
                            BoundaryParams p2 = (c2 == 0) ? P2lo : P2hi;
                            if (p1.type == Type::None || p2.type == Type::None)
                                continue;

                            int b1 = (c1 == 0) ? domain.smallEnd(d1) : domain.bigEnd(d1);
                            int b2 = (c2 == 0) ? domain.smallEnd(d2) : domain.bigEnd(d2);
                            bool t1 = (c1 == 0) ? (bx.smallEnd(d1) == domain.smallEnd(d1))
                                                : (bx.bigEnd(d1) == domain.bigEnd(d1));
                            bool t2 = (c2 == 0) ? (bx.smallEnd(d2) == domain.smallEnd(d2))
                                                : (bx.bigEnd(d2) == domain.bigEnd(d2));
                            if (!t1 || !t2)
                                continue;

                            int ng = 4;
                            int s1 = (c1 == 1) ? 1 : -1;
                            int s2 = (c2 == 1) ? 1 : -1;

                            amrex::Box ebox;
                            ebox.setSmall(d1, (c1 == 1) ? b1 + 1 : b1 - ng);
                            ebox.setBig(d1,   (c1 == 1) ? b1 + ng : b1 - 1);
                            ebox.setSmall(d2, (c2 == 1) ? b2 + 1 : b2 - ng);
                            ebox.setBig(d2,   (c2 == 1) ? b2 + ng : b2 - 1);
                            ebox.setSmall(dt, bx.smallEnd(dt));
                            ebox.setBig(dt,   bx.bigEnd(dt));

                            Set::Scalar p1t = p1.target_p, p2t = p2.target_p;
                            Set::Scalar sig1 = p1.sigma, sig2 = p2.sigma;
                            Set::Scalar bet1 = p1.beta, bet2 = p2.beta;
                            Set::Scalar Lr1 = p1.L_ref, Lr2 = p2.L_ref;
                            Type ty1 = p1.type, ty2 = p2.type;
                            int b1l = b1, b2l = b2, s1l = s1, s2l = s2;
                            int d1l = d1, d2l = d2, dtl = dt, cc1 = c1, cc2 = c2;

                            amrex::ParallelFor(ebox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                                int gidx[3] = {i, j, k};
                                int cidx[3] = {i, j, k};
                                cidx[d1l] = b1l;
                                cidx[d2l] = b2l;

                                // corner-cell primitives (rho/p/a/gm/pi are dir-independent)
                                Set::Scalar rho_c, ua_t, ub_t, p_c, a_c, gm_c, pi_c;
                                compute_prim(eta_arr, rho_arr, M_arr, E_arr, cidx[0], cidx[1], cidx[2], 0, rho_c, ua_t, ub_t, p_c, a_c, gm_c, pi_c);
                                Set::Scalar inv_rc = 1.0 / (rho_c + small_local);
                                Set::Scalar v1_c = M_arr(cidx[0], cidx[1], cidx[2], d1l) * inv_rc;
                                Set::Scalar v2_c = M_arr(cidx[0], cidx[1], cidx[2], d2l) * inv_rc;
                                Set::Scalar vt_c = M_arr(cidx[0], cidx[1], cidx[2], dtl) * inv_rc;
                                Set::Scalar a2_c = a_c * a_c;
                                Set::Scalar Mn1 = std::abs(v1_c) / (a_c + small_local);
                                Set::Scalar Mn2 = std::abs(v2_c) / (a_c + small_local);

                                // d1 one-sided derivatives
                                int n1a[3] = {cidx[0], cidx[1], cidx[2]}; n1a[d1l] = b1l - s1l;
                                int n1b[3] = {cidx[0], cidx[1], cidx[2]}; n1b[d1l] = b1l - 2 * s1l;
                                Set::Scalar r1a, t1a, t1b, pp1a, aa1, gg1, ii1;
                                Set::Scalar r1b, t1c, t1d, pp1b, aa2, gg2, ii2;
                                compute_prim(eta_arr, rho_arr, M_arr, E_arr, n1a[0], n1a[1], n1a[2], 0, r1a, t1a, t1b, pp1a, aa1, gg1, ii1);
                                compute_prim(eta_arr, rho_arr, M_arr, E_arr, n1b[0], n1b[1], n1b[2], 0, r1b, t1c, t1d, pp1b, aa2, gg2, ii2);
                                Set::Scalar v1_1a = M_arr(n1a[0], n1a[1], n1a[2], d1l) / (r1a + small_local);
                                Set::Scalar v2_1a = M_arr(n1a[0], n1a[1], n1a[2], d2l) / (r1a + small_local);
                                Set::Scalar v1_1b = M_arr(n1b[0], n1b[1], n1b[2], d1l) / (r1b + small_local);
                                Set::Scalar v2_1b = M_arr(n1b[0], n1b[1], n1b[2], d2l) / (r1b + small_local);
                                Set::Scalar fd1 = (Set::Scalar)s1l;
                                Set::Scalar drho_d1 = fd1 * (3.0 * rho_c - 4.0 * r1a + r1b) / (2.0 * dx[d1l]);
                                Set::Scalar dv1_d1  = fd1 * (3.0 * v1_c - 4.0 * v1_1a + v1_1b) / (2.0 * dx[d1l]);
                                Set::Scalar dv2_d1  = fd1 * (3.0 * v2_c - 4.0 * v2_1a + v2_1b) / (2.0 * dx[d1l]);
                                Set::Scalar dp_d1   = fd1 * (3.0 * p_c - 4.0 * pp1a + pp1b) / (2.0 * dx[d1l]);

                                // d2 one-sided derivatives
                                int n2a[3] = {cidx[0], cidx[1], cidx[2]}; n2a[d2l] = b2l - s2l;
                                int n2b[3] = {cidx[0], cidx[1], cidx[2]}; n2b[d2l] = b2l - 2 * s2l;
                                Set::Scalar r2a, s2a, s2b, pp2a, ba1, hg1, ji1;
                                Set::Scalar r2b, s2c, s2d, pp2b, ba2, hg2, ji2;
                                compute_prim(eta_arr, rho_arr, M_arr, E_arr, n2a[0], n2a[1], n2a[2], 0, r2a, s2a, s2b, pp2a, ba1, hg1, ji1);
                                compute_prim(eta_arr, rho_arr, M_arr, E_arr, n2b[0], n2b[1], n2b[2], 0, r2b, s2c, s2d, pp2b, ba2, hg2, ji2);
                                Set::Scalar v1_2a = M_arr(n2a[0], n2a[1], n2a[2], d1l) / (r2a + small_local);
                                Set::Scalar v2_2a = M_arr(n2a[0], n2a[1], n2a[2], d2l) / (r2a + small_local);
                                Set::Scalar v1_2b = M_arr(n2b[0], n2b[1], n2b[2], d1l) / (r2b + small_local);
                                Set::Scalar v2_2b = M_arr(n2b[0], n2b[1], n2b[2], d2l) / (r2b + small_local);
                                Set::Scalar fd2 = (Set::Scalar)s2l;
                                Set::Scalar drho_d2 = fd2 * (3.0 * rho_c - 4.0 * r2a + r2b) / (2.0 * dx[d2l]);
                                Set::Scalar dv1_d2  = fd2 * (3.0 * v1_c - 4.0 * v1_2a + v1_2b) / (2.0 * dx[d2l]);
                                Set::Scalar dv2_d2  = fd2 * (3.0 * v2_c - 4.0 * v2_2a + v2_2b) / (2.0 * dx[d2l]);
                                Set::Scalar dp_d2   = fd2 * (3.0 * p_c - 4.0 * pp2a + pp2b) / (2.0 * dx[d2l]);

                                // LODI waves (d1, d2)
                                Set::Scalar L1_1 = (v1_c - a_c) * (dp_d1 - rho_c * a_c * dv1_d1);
                                Set::Scalar L2_1 = (std::abs(v1_c) > 1.0e-14) ? v1_c * (a2_c * drho_d1 - dp_d1) : 0.0;
                                Set::Scalar L3_1 = (std::abs(v1_c) > 1.0e-14) ? v1_c * dv2_d1 : 0.0;
                                Set::Scalar L4_1 = (v1_c + a_c) * (dp_d1 + rho_c * a_c * dv1_d1);
                                Set::Scalar L1_2 = (v2_c - a_c) * (dp_d2 - rho_c * a_c * dv2_d2);
                                Set::Scalar L2_2 = (std::abs(v2_c) > 1.0e-14) ? v2_c * (a2_c * drho_d2 - dp_d2) : 0.0;
                                Set::Scalar L3_2 = (std::abs(v2_c) > 1.0e-14) ? v2_c * dv1_d2 : 0.0;
                                Set::Scalar L4_2 = (v2_c + a_c) * (dp_d2 + rho_c * a_c * dv2_d2);

                                // 2x2 coupled incoming waves
                                Set::Scalar K1 = sig1 * (1.0 - Mn1 * Mn1) * a_c / Lr1;
                                Set::Scalar K2 = sig2 * (1.0 - Mn2 * Mn2) * a_c / Lr2;
                                Set::Scalar ab2 = (1.0 - 0.5 * (bet1 + bet2)) / 2.0;
                                Set::Scalar Lin1 = (cc1 == 1) ? L1_1 : L4_1;
                                Set::Scalar Lin2 = (cc2 == 1) ? L1_2 : L4_2;
                                if (ty1 == Type::Outflow && ty2 == Type::Outflow)
                                {
                                    Set::Scalar bb1 = K1 * (p_c - p1t), bb2 = K2 * (p_c - p2t);
                                    Set::Scalar det = 1.0 / (1.0 - ab2 * ab2 + 1.0e-20);
                                    Lin1 = det * (bb1 - ab2 * bb2);
                                    Lin2 = det * (bb2 - ab2 * bb1);
                                }
                                else if (ty1 == Type::Outflow && ty2 == Type::Inflow) { Lin1 = K1 * (p_c - p1t); Lin2 = 0.0; }
                                else if (ty1 == Type::Inflow && ty2 == Type::Outflow) { Lin1 = 0.0; Lin2 = K2 * (p_c - p2t); }
                                else { Lin1 = 0.0; Lin2 = 0.0; }
                                if (cc1 == 1) L1_1 = Lin1; else L4_1 = Lin1;
                                if (cc2 == 1) L1_2 = Lin2; else L4_2 = Lin2;

                                // reconstruct LODI-corrected gradients
                                Set::Scalar xi1_1 = L1_1 / (v1_c - a_c + small_local);
                                Set::Scalar xi2_1 = (std::abs(v1_c) > 1.0e-14) ? L2_1 / v1_c : 0.0;
                                Set::Scalar xi3_1 = (std::abs(v1_c) > 1.0e-14) ? L3_1 / v1_c : 0.0;
                                Set::Scalar xi4_1 = L4_1 / (v1_c + a_c + small_local);
                                Set::Scalar dp_d1n   = 0.5 * (xi1_1 + xi4_1);
                                Set::Scalar dv1_d1n  = (xi4_1 - xi1_1) / (2.0 * (rho_c + small_local) * a_c + small_local);
                                Set::Scalar drho_d1n = (xi2_1 + 0.5 * (xi1_1 + xi4_1)) / (a2_c + small_local);
                                Set::Scalar dv2_d1n  = xi3_1;
                                Set::Scalar xi1_2 = L1_2 / (v2_c - a_c + small_local);
                                Set::Scalar xi2_2 = (std::abs(v2_c) > 1.0e-14) ? L2_2 / v2_c : 0.0;
                                Set::Scalar xi3_2 = (std::abs(v2_c) > 1.0e-14) ? L3_2 / v2_c : 0.0;
                                Set::Scalar xi4_2 = L4_2 / (v2_c + a_c + small_local);
                                Set::Scalar dp_d2n   = 0.5 * (xi1_2 + xi4_2);
                                Set::Scalar dv2_d2n  = (xi4_2 - xi1_2) / (2.0 * (rho_c + small_local) * a_c + small_local);
                                Set::Scalar drho_d2n = (xi2_2 + 0.5 * (xi1_2 + xi4_2)) / (a2_c + small_local);
                                Set::Scalar dv1_d2n  = xi3_2;

                                // linear (gradient-corrected) ghost extrapolation
                                Set::Scalar off1 = (Set::Scalar)(gidx[d1l] - b1l) * dx[d1l];
                                Set::Scalar off2 = (Set::Scalar)(gidx[d2l] - b2l) * dx[d2l];
                                Set::Scalar rho_g = rho_c + off1 * drho_d1n + off2 * drho_d2n;
                                Set::Scalar v1_g  = v1_c  + off1 * dv1_d1n  + off2 * dv1_d2n;
                                Set::Scalar v2_g  = v2_c  + off1 * dv2_d1n  + off2 * dv2_d2n;
                                Set::Scalar p_g   = p_c   + off1 * dp_d1n   + off2 * dp_d2n;

                                rho_g = amrex::min(amrex::max(rho_g, pref_local), 1.0e20);
                                p_g   = amrex::min(amrex::max(p_g, 1.0e-6), 1.0e20);

                                Set::Scalar eta_g = eta_arr(gidx[0], gidx[1], gidx[2]);
                                Set::Scalar gm_g  = Solver::EOS::EOS::MixedGamma(eta_g, eos0, eos1);
                                Set::Scalar pi_g  = Solver::EOS::EOS::MixedP0(eta_g, eos0, eos1);
                                Set::Scalar UE_g  = Solver::EOS::EOS::MixedInternalEnergy(p_g, eta_g, eos0, eos1, pref_local, small_local);
                                if (UE_g < 0.0) UE_g = 0.0;
                                Set::Scalar KE_g = 0.5 * rho_g * (v1_g * v1_g + v2_g * v2_g + vt_c * vt_c);

                                rho_arr(gidx[0], gidx[1], gidx[2]) = rho_g;
                                M_arr(gidx[0], gidx[1], gidx[2], d1l) = rho_g * v1_g;
                                M_arr(gidx[0], gidx[1], gidx[2], d2l) = rho_g * v2_g;
                                M_arr(gidx[0], gidx[1], gidx[2], dtl) = rho_g * vt_c;
                                E_arr(gidx[0], gidx[1], gidx[2]) = UE_g + KE_g;
                                gamma_arr(gidx[0], gidx[1], gidx[2]) = gm_g;
                                pi_arr(gidx[0], gidx[1], gidx[2]) = pi_g;
                                p_arr(gidx[0], gidx[1], gidx[2]) = p_g;
                            });
                        }
                }
            }

            // =====================================================================
            // STEP 1c: 3D CORNERS (8 of them, triple-ghost cubes).
            // =====================================================================
            // The dimension-split flux reconstruction never reads the triple-ghost
            // corner cells, so an elaborate 3x3-coupled LODI buys nothing; fill them
            // by constant extrapolation from the corner boundary cell so they hold a
            // physical, non-stale state (robustness + clean output). Runs after the
            // edges; the triple-ghost region is disjoint from faces (1 ghost) and
            // edges (2 ghost / 1 valid).
            // =====================================================================
            for (int cx = 0; cx < 2; ++cx)
                for (int cy = 0; cy < 2; ++cy)
                    for (int cz = 0; cz < 2; ++cz)
                    {
                        BoundaryParams pcx = (cx == 0) ? params_xlo_local : params_xhi_local;
                        BoundaryParams pcy = (cy == 0) ? params_ylo_local : params_yhi_local;
                        BoundaryParams pcz = (cz == 0) ? params_zlo_local : params_zhi_local;
                        if (pcx.type == Type::None || pcy.type == Type::None || pcz.type == Type::None)
                            continue;
                        int ibx = (cx == 0) ? domain.smallEnd(0) : domain.bigEnd(0);
                        int jby = (cy == 0) ? domain.smallEnd(1) : domain.bigEnd(1);
                        int kbz = (cz == 0) ? domain.smallEnd(2) : domain.bigEnd(2);
                        bool tcx = (cx == 0) ? (bx.smallEnd(0) == domain.smallEnd(0)) : (bx.bigEnd(0) == domain.bigEnd(0));
                        bool tcy = (cy == 0) ? (bx.smallEnd(1) == domain.smallEnd(1)) : (bx.bigEnd(1) == domain.bigEnd(1));
                        bool tcz = (cz == 0) ? (bx.smallEnd(2) == domain.smallEnd(2)) : (bx.bigEnd(2) == domain.bigEnd(2));
                        if (!tcx || !tcy || !tcz)
                            continue;

                        int ng = 4;
                        amrex::Box cbox;
                        cbox.setSmall(0, (cx == 1) ? ibx + 1 : ibx - ng); cbox.setBig(0, (cx == 1) ? ibx + ng : ibx - 1);
                        cbox.setSmall(1, (cy == 1) ? jby + 1 : jby - ng); cbox.setBig(1, (cy == 1) ? jby + ng : jby - 1);
                        cbox.setSmall(2, (cz == 1) ? kbz + 1 : kbz - ng); cbox.setBig(2, (cz == 1) ? kbz + ng : kbz - 1);
                        int ibxl = ibx, jbyl = jby, kbzl = kbz;

                        amrex::ParallelFor(cbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                            Set::Scalar rho_b, u1, u2, p_b, a_b, gm_b, pi_b;
                            compute_prim(eta_arr, rho_arr, M_arr, E_arr, ibxl, jbyl, kbzl, 0, rho_b, u1, u2, p_b, a_b, gm_b, pi_b);
                            Set::Scalar inv = 1.0 / (rho_b + small_local);
                            Set::Scalar vx_b = M_arr(ibxl, jbyl, kbzl, 0) * inv;
                            Set::Scalar vy_b = M_arr(ibxl, jbyl, kbzl, 1) * inv;
                            Set::Scalar vz_b = M_arr(ibxl, jbyl, kbzl, 2) * inv;
                            Set::Scalar eta_g = eta_arr(i, j, k);
                            Set::Scalar gm_g = Solver::EOS::EOS::MixedGamma(eta_g, eos0, eos1);
                            Set::Scalar pi_g = Solver::EOS::EOS::MixedP0(eta_g, eos0, eos1);
                            Set::Scalar p_g = amrex::max(p_b, 1.0e-6);
                            Set::Scalar UE_g = Solver::EOS::EOS::MixedInternalEnergy(p_g, eta_g, eos0, eos1, pref_local, small_local);
                            if (UE_g < 0.0) UE_g = 0.0;
                            Set::Scalar rho_g = amrex::max(rho_b, pref_local);
                            Set::Scalar KE_g = 0.5 * rho_g * (vx_b * vx_b + vy_b * vy_b + vz_b * vz_b);
                            rho_arr(i, j, k) = rho_g;
                            M_arr(i, j, k, 0) = rho_g * vx_b;
                            M_arr(i, j, k, 1) = rho_g * vy_b;
                            M_arr(i, j, k, 2) = rho_g * vz_b;
                            E_arr(i, j, k) = UE_g + KE_g;
                            gamma_arr(i, j, k) = gm_g;
                            pi_arr(i, j, k) = pi_g;
                            p_arr(i, j, k) = p_g;
                        });
                    }
#endif

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
                // Fix #4: include full transverse range. The corner *quadrant*
                // block fills i-ghost cells at j in [smallEnd-ng, smallEnd-1]
                // and [bigEnd+1, bigEnd+ng]; it does NOT fill (i_ghost, j=smallEnd)
                // or (i_ghost, j=bigEnd). Skipping those rows in the edge would
                // leave them at construction defaults, creating a stiff fictitious
                // jump at the corner row.
                int j_start = domain.smallEnd(1);
                int j_end = domain.bigEnd(1);

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
                        // CORRECTED: Removed extra division factors
                        // L_1 = (u - c) * (dp/dx - rho*c*du/dx)
                        // L_2 = u * (c^2*drho/dx - dp/dx)
                        // L_3 = u * dv/dx
                        // L_4 = (u + c) * (dp/dx + rho*c*du/dx)
                        // =============================================================
                        Set::Scalar L1 = (un_bnd - a_bnd) * (dp_dx - rho_bnd * a_bnd * dun_dx);
                        Set::Scalar L2 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd * (a2_bnd * drho_dx - dp_dx) : 0.0;
                        Set::Scalar L3 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd * dut_dx : 0.0;
                        Set::Scalar L4 = (un_bnd + a_bnd) * (dp_dx + rho_bnd * a_bnd * dun_dx);

                        // =============================================================
                        // STEP 6: Transverse terms (Motheau Eq. 36-37, 45-46)
                        // CORRECTED: Removed eigenvalue factors, fixed structure
                        // T_1 = v(dp/dy - rho*c*du/dy) + gamma*p*dv/dy                (Eq. 36)
                        // T_4 = v(dp/dy + rho*c*du/dy) + gamma*p*dv/dy                (Eq. 37)
                        // T_2 = v(c^2*drho/dy - dp/dy) + c^2*rho*dv/dy - gamma*p*dv/dy (Eq. 45)
                        // T_3 = v*dv/dy + (1/rho)*dp/dy                                (Eq. 46)
                        // =============================================================
                        Set::Scalar T1 = ut_bnd * (dp_dy - rho_bnd * a_bnd * dun_dy) + gm_bnd * p_bnd * dut_dy;
                        Set::Scalar T4 = ut_bnd * (dp_dy + rho_bnd * a_bnd * dun_dy) + gm_bnd * p_bnd * dut_dy;
                        Set::Scalar T2 = (std::abs(un_bnd) > 1.0e-14) ? ut_bnd * (a2_bnd * drho_dy - dp_dy) + (a2_bnd * rho_bnd - gm_bnd * p_bnd) * dut_dy : 0.0;
                        Set::Scalar T3 = (std::abs(un_bnd) > 1.0e-14) ? ut_bnd * dut_dy + (1.0 / rho_bnd) * dp_dy : 0.0;

                        // =============================================================
                        // STEP 7: Apply BC-specific modification to incoming wave
                        // For X-LO: L4 is incoming (speed un+a > 0 entering domain)
                        // =============================================================
#if AMREX_SPACEDIM == 3
                        // 2nd transverse (z) contribution to the transverse terms.
                        {
                            Set::Scalar ut2b = M_arr(i_bnd, j, k, 2) / rho_bnd;
                            if (k > domain.smallEnd(2) && k < domain.bigEnd(2))
                            {
                                Set::Scalar r_p,u_p,t_p,pp_p,a_p,g_p,pi_p;
                                compute_prim(eta_arr, rho_arr, M_arr, E_arr, i_bnd, j, k + 1, idir, r_p,u_p,t_p,pp_p,a_p,g_p,pi_p);
                                Set::Scalar r_m,u_m,t_m,pp_m,a_m,g_m,pi_m;
                                compute_prim(eta_arr, rho_arr, M_arr, E_arr, i_bnd, j, k - 1, idir, r_m,u_m,t_m,pp_m,a_m,g_m,pi_m);
                                Set::Scalar inv2 = 1.0 / (2.0 * dx[2]);
                                Set::Scalar dr2 = (r_p - r_m) * inv2;
                                Set::Scalar du2 = (u_p - u_m) * inv2;
                                Set::Scalar dp2 = (pp_p - pp_m) * inv2;
                                Set::Scalar dw2 = (M_arr(i_bnd,j,k+1,2)/(r_p+small_local) - M_arr(i_bnd,j,k-1,2)/(r_m+small_local)) * inv2;
                                T1 += ut2b * (dp2 - rho_bnd * a_bnd * du2) + gm_bnd * p_bnd * dw2;
                                T4 += ut2b * (dp2 + rho_bnd * a_bnd * du2) + gm_bnd * p_bnd * dw2;
                                T2 += ut2b * (a2_bnd * dr2 - dp2) + (a2_bnd * rho_bnd - gm_bnd * p_bnd) * dw2;
                            }
                        }
#endif
                        Set::Scalar K = params_xlo_local.sigma * (1.0 - M_loc * M_loc) * a_bnd / (params_xlo_local.L_ref);

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

                        // Motheau Eq. 26 inversion for paper-style L (see corner block).
                        Set::Scalar dp_dx_new   = 0.5 * (xi1 + xi4);
                        Set::Scalar dun_dx_new  = (xi4 - xi1) / (2.0 * (rho_bnd + small_local) * a_bnd + small_local);
                        Set::Scalar drho_dx_new = (xi2 + 0.5 * (xi1 + xi4)) / (a2_bnd + small_local);
                        Set::Scalar dut_dx_new  = xi3;

#if AMREX_SPACEDIM == 3
                        // --- 2nd tangential velocity (vz for an x-face): advected shear wave. ---
                        // ut2 is carried by the shear characteristic (speed un); for OUTFLOW it
                        // passes through unchanged (dut2/dn keeps its interior value), for INFLOW
                        // it is relaxed toward target_w.  Computed inline from M[2] (set_ghost
                        // zeroes the out-of-plane component; we overwrite it + its KE below).
                        Set::Scalar ut2_bnd = M_arr(i_bnd,     j, k, 2) / rho_bnd;
                        Set::Scalar ut2_m1  = M_arr(i_bnd + 1, j, k, 2) / (rho_m1 + small_local);
                        Set::Scalar ut2_m2  = M_arr(i_bnd + 2, j, k, 2) / (rho_m2 + small_local);
                        Set::Scalar dut2_dx_new = (-3.0 * ut2_bnd + 4.0 * ut2_m1 - ut2_m2) / (2.0 * delta);
                        if (params_xlo_local.type == Type::Inflow)
                        {
                            Set::Scalar K_shear = params_xlo_local.relax_w * a_bnd / params_xlo_local.L_ref;
                            Set::Scalar L3b = K_shear * (ut2_bnd - params_xlo_local.target_w);
                            dut2_dx_new = (std::abs(un_bnd) > 1.0e-14) ? L3b / un_bnd : 0.0;
                        }
                        Set::Scalar ut2_g1 = ut2_m1 - 2.0 * delta * dut2_dx_new;
                        Set::Scalar ut2_g2 = -2.0 * ut2_m1 - 3.0 * ut2_bnd + 6.0 * ut2_g1 + 6.0 * delta * dut2_dx_new;
                        Set::Scalar ut2_g3 = 3.0 * ut2_m1 + 10.0 * ut2_bnd - 18.0 * ut2_g1 + 6.0 * ut2_g2 - 12.0 * delta * dut2_dx_new;
                        Set::Scalar ut2_g4 = -2.0 * ut2_m1 - 13.0 * ut2_bnd + 24.0 * ut2_g1 - 12.0 * ut2_g2 + 4.0 * ut2_g3 + 12.0 * delta * dut2_dx_new;
#endif

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
                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, gamma_arr, pi_arr, p_arr, i_bnd - 1, j, k, idir, rho_g1, un_g1, ut_g1, p_g1);

                        // Layer 2: Motheau Eq. 28
                        // Q_{i-2} = -2*Q_{i+1} - 3*Q_i + 6*Q_{i-1} + 6*dx*dQ/dx
                        Set::Scalar rho_g2 = -2.0 * rho_m1 - 3.0 * rho_bnd + 6.0 * rho_g1 + 6.0 * delta * drho_dx_new;
                        Set::Scalar un_g2 = -2.0 * un_m1 - 3.0 * un_bnd + 6.0 * un_g1 + 6.0 * delta * dun_dx_new;
                        Set::Scalar ut_g2 = -2.0 * ut_m1 - 3.0 * ut_bnd + 6.0 * ut_g1 + 6.0 * delta * dut_dx_new;
                        Set::Scalar p_g2 = -2.0 * p_m1 - 3.0 * p_bnd + 6.0 * p_g1 + 6.0 * delta * dp_dx_new;

                        rho_g2 = amrex::max(rho_g2, small_local);
                        p_g2 = amrex::max(p_g2, 1.0e-6);
                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, gamma_arr, pi_arr, p_arr, i_bnd - 2, j, k, idir, rho_g2, un_g2, ut_g2, p_g2);

                        // Layer 3: Motheau Eq. 29
                        // Q_{i-3} = 3*Q_{i+1} + 10*Q_i - 18*Q_{i-1} + 6*Q_{i-2} - 12*dx*dQ/dx
                        Set::Scalar rho_g3 = 3.0 * rho_m1 + 10.0 * rho_bnd - 18.0 * rho_g1 + 6.0 * rho_g2 - 12.0 * delta * drho_dx_new;
                        Set::Scalar un_g3 = 3.0 * un_m1 + 10.0 * un_bnd - 18.0 * un_g1 + 6.0 * un_g2 - 12.0 * delta * dun_dx_new;
                        Set::Scalar ut_g3 = 3.0 * ut_m1 + 10.0 * ut_bnd - 18.0 * ut_g1 + 6.0 * ut_g2 - 12.0 * delta * dut_dx_new;
                        Set::Scalar p_g3 = 3.0 * p_m1 + 10.0 * p_bnd - 18.0 * p_g1 + 6.0 * p_g2 - 12.0 * delta * dp_dx_new;

                        rho_g3 = amrex::max(rho_g3, small_local);
                        p_g3 = amrex::max(p_g3, 1.0e-6);
                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, gamma_arr, pi_arr, p_arr, i_bnd - 3, j, k, idir, rho_g3, un_g3, ut_g3, p_g3);

                        // Layer 4: Motheau Eq. 30
                        // Q_{i-4} = -2*Q_{i+1} - 13*Q_i + 24*Q_{i-1} - 12*Q_{i-2} + 4*Q_{i-3} + 12*dx*dQ/dx
                        Set::Scalar rho_g4 = -2.0 * rho_m1 - 13.0 * rho_bnd + 24.0 * rho_g1 - 12.0 * rho_g2 + 4.0 * rho_g3 + 12.0 * delta * drho_dx_new;
                        Set::Scalar un_g4 = -2.0 * un_m1 - 13.0 * un_bnd + 24.0 * un_g1 - 12.0 * un_g2 + 4.0 * un_g3 + 12.0 * delta * dun_dx_new;
                        Set::Scalar ut_g4 = -2.0 * ut_m1 - 13.0 * ut_bnd + 24.0 * ut_g1 - 12.0 * ut_g2 + 4.0 * ut_g3 + 12.0 * delta * dut_dx_new;
                        Set::Scalar p_g4 = -2.0 * p_m1 - 13.0 * p_bnd + 24.0 * p_g1 - 12.0 * p_g2 + 4.0 * p_g3 + 12.0 * delta * dp_dx_new;

                        rho_g4 = amrex::max(rho_g4, small_local);
                        p_g4 = amrex::max(p_g4, 1.0e-6);
                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, gamma_arr, pi_arr, p_arr, i_bnd - 4, j, k, idir, rho_g4, un_g4, ut_g4, p_g4);

#if AMREX_SPACEDIM == 3
                        // set_ghost zeroed the out-of-plane momentum and its KE; write the
                        // advected 2nd-tangential momentum and add its kinetic energy.
                        M_arr(i_bnd - 1, j, k, 2) = rho_g1 * ut2_g1; E_arr(i_bnd - 1, j, k) += 0.5 * rho_g1 * ut2_g1 * ut2_g1;
                        M_arr(i_bnd - 2, j, k, 2) = rho_g2 * ut2_g2; E_arr(i_bnd - 2, j, k) += 0.5 * rho_g2 * ut2_g2 * ut2_g2;
                        M_arr(i_bnd - 3, j, k, 2) = rho_g3 * ut2_g3; E_arr(i_bnd - 3, j, k) += 0.5 * rho_g3 * ut2_g3 * ut2_g3;
                        M_arr(i_bnd - 4, j, k, 2) = rho_g4 * ut2_g4; E_arr(i_bnd - 4, j, k) += 0.5 * rho_g4 * ut2_g4 * ut2_g4;
#endif
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
                // Fix #4: include full transverse range. The corner *quadrant*
                // block fills i-ghost cells at j in [smallEnd-ng, smallEnd-1]
                // and [bigEnd+1, bigEnd+ng]; it does NOT fill (i_ghost, j=smallEnd)
                // or (i_ghost, j=bigEnd). Skipping those rows in the edge would
                // leave them at construction defaults, creating a stiff fictitious
                // jump at the corner row.
                int j_start = domain.smallEnd(1);
                int j_end = domain.bigEnd(1);

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
                        // CORRECTED: Removed extra division factors
                        // =============================================================
                        Set::Scalar L1 = (un_bnd - a_bnd) * (dp_dx - rho_bnd * a_bnd * dun_dx);
                        Set::Scalar L2 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd * (a2_bnd * drho_dx - dp_dx) : 0.0;
                        Set::Scalar L3 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd * dut_dx : 0.0;
                        Set::Scalar L4 = (un_bnd + a_bnd) * (dp_dx + rho_bnd * a_bnd * dun_dx);

                        // =============================================================
                        // STEP 6: Transverse terms (Motheau Eq. 36-37, 45-46)
                        // CORRECTED: Removed eigenvalue factors, fixed structure
                        // =============================================================
                        Set::Scalar T1 = ut_bnd * (dp_dy - rho_bnd * a_bnd * dun_dy) + gm_bnd * p_bnd * dut_dy;
                        Set::Scalar T4 = ut_bnd * (dp_dy + rho_bnd * a_bnd * dun_dy) + gm_bnd * p_bnd * dut_dy;
                        Set::Scalar T2 = (std::abs(un_bnd) > 1.0e-14) ? ut_bnd * (a2_bnd * drho_dy - dp_dy) + (a2_bnd * rho_bnd - gm_bnd * p_bnd) * dut_dy : 0.0;
                        Set::Scalar T3 = (std::abs(un_bnd) > 1.0e-14) ? ut_bnd * dut_dy + (1.0 / rho_bnd) * dp_dy : 0.0;

                        // =============================================================
                        // STEP 7: Apply BC-specific modification to incoming wave
                        // For X-HI: L1 is incoming (speed un-a < 0 entering domain)
                        // =============================================================
#if AMREX_SPACEDIM == 3
                        // 2nd transverse (z) contribution to the transverse terms.
                        {
                            Set::Scalar ut2b = M_arr(i_bnd, j, k, 2) / rho_bnd;
                            if (k > domain.smallEnd(2) && k < domain.bigEnd(2))
                            {
                                Set::Scalar r_p,u_p,t_p,pp_p,a_p,g_p,pi_p;
                                compute_prim(eta_arr, rho_arr, M_arr, E_arr, i_bnd, j, k + 1, idir, r_p,u_p,t_p,pp_p,a_p,g_p,pi_p);
                                Set::Scalar r_m,u_m,t_m,pp_m,a_m,g_m,pi_m;
                                compute_prim(eta_arr, rho_arr, M_arr, E_arr, i_bnd, j, k - 1, idir, r_m,u_m,t_m,pp_m,a_m,g_m,pi_m);
                                Set::Scalar inv2 = 1.0 / (2.0 * dx[2]);
                                Set::Scalar dr2 = (r_p - r_m) * inv2;
                                Set::Scalar du2 = (u_p - u_m) * inv2;
                                Set::Scalar dp2 = (pp_p - pp_m) * inv2;
                                Set::Scalar dw2 = (M_arr(i_bnd,j,k+1,2)/(r_p+small_local) - M_arr(i_bnd,j,k-1,2)/(r_m+small_local)) * inv2;
                                T1 += ut2b * (dp2 - rho_bnd * a_bnd * du2) + gm_bnd * p_bnd * dw2;
                                T4 += ut2b * (dp2 + rho_bnd * a_bnd * du2) + gm_bnd * p_bnd * dw2;
                                T2 += ut2b * (a2_bnd * dr2 - dp2) + (a2_bnd * rho_bnd - gm_bnd * p_bnd) * dw2;
                            }
                        }
#endif
                        Set::Scalar K = params_xhi_local.sigma * (1.0 - M_loc * M_loc) * a_bnd / (params_xhi_local.L_ref);

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

                        // Motheau Eq. 26 inversion for paper-style L (see corner block).
                        Set::Scalar dp_dx_new   = 0.5 * (xi1 + xi4);
                        Set::Scalar dun_dx_new  = (xi4 - xi1) / (2.0 * (rho_bnd + small_local) * a_bnd + small_local);
                        Set::Scalar drho_dx_new = (xi2 + 0.5 * (xi1 + xi4)) / (a2_bnd + small_local);
                        Set::Scalar dut_dx_new  = xi3;

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
                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, gamma_arr, pi_arr, p_arr, i_bnd + 1, j, k, idir, rho_g1, un_g1, ut_g1, p_g1);

                        // Layer 2: Motheau Eq. 32
                        // Q_{i+2} = -2*Q_{i-1} - 3*Q_i + 6*Q_{i+1} - 6*dx*dQ/dx
                        Set::Scalar rho_g2 = -2.0 * rho_m1 - 3.0 * rho_bnd + 6.0 * rho_g1 - 6.0 * delta * drho_dx_new;
                        Set::Scalar un_g2 = -2.0 * un_m1 - 3.0 * un_bnd + 6.0 * un_g1 - 6.0 * delta * dun_dx_new;
                        Set::Scalar ut_g2 = -2.0 * ut_m1 - 3.0 * ut_bnd + 6.0 * ut_g1 - 6.0 * delta * dut_dx_new;
                        Set::Scalar p_g2 = -2.0 * p_m1 - 3.0 * p_bnd + 6.0 * p_g1 - 6.0 * delta * dp_dx_new;

                        rho_g2 = amrex::max(rho_g2, small_local);
                        p_g2 = amrex::max(p_g2, 1.0e-6);
                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, gamma_arr, pi_arr, p_arr, i_bnd + 2, j, k, idir, rho_g2, un_g2, ut_g2, p_g2);

                        // Layer 3: Motheau Eq. 33
                        // Q_{i+3} = 3*Q_{i-1} + 10*Q_i - 18*Q_{i+1} + 6*Q_{i+2} + 12*dx*dQ/dx
                        Set::Scalar rho_g3 = 3.0 * rho_m1 + 10.0 * rho_bnd - 18.0 * rho_g1 + 6.0 * rho_g2 + 12.0 * delta * drho_dx_new;
                        Set::Scalar un_g3 = 3.0 * un_m1 + 10.0 * un_bnd - 18.0 * un_g1 + 6.0 * un_g2 + 12.0 * delta * dun_dx_new;
                        Set::Scalar ut_g3 = 3.0 * ut_m1 + 10.0 * ut_bnd - 18.0 * ut_g1 + 6.0 * ut_g2 + 12.0 * delta * dut_dx_new;
                        Set::Scalar p_g3 = 3.0 * p_m1 + 10.0 * p_bnd - 18.0 * p_g1 + 6.0 * p_g2 + 12.0 * delta * dp_dx_new;

                        rho_g3 = amrex::max(rho_g3, small_local);
                        p_g3 = amrex::max(p_g3, 1.0e-6);
                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, gamma_arr, pi_arr, p_arr, i_bnd + 3, j, k, idir, rho_g3, un_g3, ut_g3, p_g3);

                        // Layer 4: Motheau Eq. 34
                        // Q_{i+4} = -2*Q_{i-1} - 13*Q_i + 24*Q_{i+1} - 12*Q_{i+2} + 4*Q_{i+3} - 12*dx*dQ/dx
                        Set::Scalar rho_g4 = -2.0 * rho_m1 - 13.0 * rho_bnd + 24.0 * rho_g1 - 12.0 * rho_g2 + 4.0 * rho_g3 - 12.0 * delta * drho_dx_new;
                        Set::Scalar un_g4 = -2.0 * un_m1 - 13.0 * un_bnd + 24.0 * un_g1 - 12.0 * un_g2 + 4.0 * un_g3 - 12.0 * delta * dun_dx_new;
                        Set::Scalar ut_g4 = -2.0 * ut_m1 - 13.0 * ut_bnd + 24.0 * ut_g1 - 12.0 * ut_g2 + 4.0 * ut_g3 - 12.0 * delta * dut_dx_new;
                        Set::Scalar p_g4 = -2.0 * p_m1 - 13.0 * p_bnd + 24.0 * p_g1 - 12.0 * p_g2 + 4.0 * p_g3 - 12.0 * delta * dp_dx_new;

                        rho_g4 = amrex::max(rho_g4, small_local);
                        p_g4 = amrex::max(p_g4, 1.0e-6);
                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, gamma_arr, pi_arr, p_arr, i_bnd + 4, j, k, idir, rho_g4, un_g4, ut_g4, p_g4);

#if AMREX_SPACEDIM == 3
                        {
                            // 2nd tangential velocity (vz, comp 2) advected shear wave + ghost KE.
                            Set::Scalar ut2_bnd = M_arr(i_bnd,     j, k, 2) / rho_bnd;
                            Set::Scalar ut2_m1  = M_arr(i_bnd - 1, j, k, 2) / (rho_m1 + small_local);
                            Set::Scalar ut2_m2  = M_arr(i_bnd - 2, j, k, 2) / (rho_m2 + small_local);
                            Set::Scalar g = (3.0 * ut2_bnd - 4.0 * ut2_m1 + ut2_m2) / (2.0 * delta);
                            if (params_xhi_local.type == Type::Inflow) {
                                Set::Scalar K_shear = params_xhi_local.relax_w * a_bnd / params_xhi_local.L_ref;
                                g = (std::abs(un_bnd) > 1.0e-14) ? (K_shear * (ut2_bnd - params_xhi_local.target_w)) / un_bnd : 0.0;
                            }
                            Set::Scalar u1 = ut2_m1 + 2.0 * delta * g;
                            Set::Scalar u2 = -2.0*ut2_m1 - 3.0*ut2_bnd + 6.0*u1 - 6.0*delta*g;
                            Set::Scalar u3 = 3.0*ut2_m1 + 10.0*ut2_bnd - 18.0*u1 + 6.0*u2 + 12.0*delta*g;
                            Set::Scalar u4 = -2.0*ut2_m1 - 13.0*ut2_bnd + 24.0*u1 - 12.0*u2 + 4.0*u3 - 12.0*delta*g;
                            M_arr(i_bnd+1,j,k,2)=rho_g1*u1; E_arr(i_bnd+1,j,k)+=0.5*rho_g1*u1*u1;
                            M_arr(i_bnd+2,j,k,2)=rho_g2*u2; E_arr(i_bnd+2,j,k)+=0.5*rho_g2*u2*u2;
                            M_arr(i_bnd+3,j,k,2)=rho_g3*u3; E_arr(i_bnd+3,j,k)+=0.5*rho_g3*u3*u3;
                            M_arr(i_bnd+4,j,k,2)=rho_g4*u4; E_arr(i_bnd+4,j,k)+=0.5*rho_g4*u4*u4;
                        }
#endif
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
                // Fix #4: include full transverse range (see comment in xlo branch).
                int i_start = domain.smallEnd(0);
                int i_end = domain.bigEnd(0);

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
                        // CORRECTED: Removed extra division factors
                        // L_1 = (v - c) * (dp/dy - rho*c*dv/dy)
                        // L_2 = v * (c^2*drho/dy - dp/dy)
                        // L_3 = v * du/dy
                        // L_4 = (v + c) * (dp/dy + rho*c*dv/dy)
                        // =============================================================
                        Set::Scalar L1 = (un_bnd - a_bnd) * (dp_dy - rho_bnd * a_bnd * dun_dy);
                        Set::Scalar L2 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd * (a2_bnd * drho_dy - dp_dy) : 0.0;
                        Set::Scalar L3 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd * dut_dy : 0.0;
                        Set::Scalar L4 = (un_bnd + a_bnd) * (dp_dy + rho_bnd * a_bnd * dun_dy);

                        // =============================================================
                        // STEP 6: Transverse terms (Motheau Eq. 36-37, 45-46)
                        // For Y-normal: transverse = X-direction
                        // CORRECTED: Removed eigenvalue factors, fixed structure
                        // T_1 = u(dp/dx - rho*c*dv/dx) + gamma*p*du/dx                (Eq. 36)
                        // T_4 = u(dp/dx + rho*c*dv/dx) + gamma*p*du/dx                (Eq. 37)
                        // T_2 = u(c^2*drho/dx - dp/dx) + c^2*rho*du/dx - gamma*p*du/dx (Eq. 45)
                        // T_3 = u*du/dx + (1/rho)*dp/dx                                (Eq. 46)
                        // =============================================================
                        Set::Scalar T1 = ut_bnd * (dp_dx - rho_bnd * a_bnd * dun_dx) + gm_bnd * p_bnd * dut_dx;
                        Set::Scalar T4 = ut_bnd * (dp_dx + rho_bnd * a_bnd * dun_dx) + gm_bnd * p_bnd * dut_dx;
                        Set::Scalar T2 = (std::abs(un_bnd) > 1.0e-14) ? ut_bnd * (a2_bnd * drho_dx - dp_dx) + (a2_bnd * rho_bnd - gm_bnd * p_bnd) * dut_dx : 0.0;
                        Set::Scalar T3 = (std::abs(un_bnd) > 1.0e-14) ? ut_bnd * dut_dx + (1.0 / rho_bnd) * dp_dx : 0.0;

                        // =============================================================
                        // STEP 7: Apply BC-specific modification to incoming wave
                        // For Y-LO: L4 is incoming (speed un+a > 0 entering domain)
                        // =============================================================
#if AMREX_SPACEDIM == 3
                        // 2nd transverse (z) contribution to the transverse terms.
                        {
                            Set::Scalar ut2b = M_arr(i, j_bnd, k, 2) / rho_bnd;
                            if (k > domain.smallEnd(2) && k < domain.bigEnd(2))
                            {
                                Set::Scalar r_p,u_p,t_p,pp_p,a_p,g_p,pi_p;
                                compute_prim(eta_arr, rho_arr, M_arr, E_arr, i, j_bnd, k + 1, idir, r_p,u_p,t_p,pp_p,a_p,g_p,pi_p);
                                Set::Scalar r_m,u_m,t_m,pp_m,a_m,g_m,pi_m;
                                compute_prim(eta_arr, rho_arr, M_arr, E_arr, i, j_bnd, k - 1, idir, r_m,u_m,t_m,pp_m,a_m,g_m,pi_m);
                                Set::Scalar inv2 = 1.0 / (2.0 * dx[2]);
                                Set::Scalar dr2 = (r_p - r_m) * inv2;
                                Set::Scalar du2 = (u_p - u_m) * inv2;
                                Set::Scalar dp2 = (pp_p - pp_m) * inv2;
                                Set::Scalar dw2 = (M_arr(i,j_bnd,k+1,2)/(r_p+small_local) - M_arr(i,j_bnd,k-1,2)/(r_m+small_local)) * inv2;
                                T1 += ut2b * (dp2 - rho_bnd * a_bnd * du2) + gm_bnd * p_bnd * dw2;
                                T4 += ut2b * (dp2 + rho_bnd * a_bnd * du2) + gm_bnd * p_bnd * dw2;
                                T2 += ut2b * (a2_bnd * dr2 - dp2) + (a2_bnd * rho_bnd - gm_bnd * p_bnd) * dw2;
                            }
                        }
#endif
                        Set::Scalar K = params_ylo_local.sigma * (1.0 - M_loc * M_loc) * a_bnd / (params_ylo_local.L_ref);

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

                        // Motheau Eq. 26 inversion for paper-style L (see corner block).
                        Set::Scalar dp_dy_new   = 0.5 * (xi1 + xi4);
                        Set::Scalar dun_dy_new  = (xi4 - xi1) / (2.0 * (rho_bnd + small_local) * a_bnd + small_local);
                        Set::Scalar drho_dy_new = (xi2 + 0.5 * (xi1 + xi4)) / (a2_bnd + small_local);
                        Set::Scalar dut_dy_new  = xi3;

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
                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, gamma_arr, pi_arr, p_arr, i, j_bnd - 1, k, idir, rho_g1, un_g1, ut_g1, p_g1);

                        // Layer 2: Motheau Eq. 28
                        Set::Scalar rho_g2 = -2.0 * rho_m1 - 3.0 * rho_bnd + 6.0 * rho_g1 + 6.0 * delta * drho_dy_new;
                        Set::Scalar un_g2 = -2.0 * un_m1 - 3.0 * un_bnd + 6.0 * un_g1 + 6.0 * delta * dun_dy_new;
                        Set::Scalar ut_g2 = -2.0 * ut_m1 - 3.0 * ut_bnd + 6.0 * ut_g1 + 6.0 * delta * dut_dy_new;
                        Set::Scalar p_g2 = -2.0 * p_m1 - 3.0 * p_bnd + 6.0 * p_g1 + 6.0 * delta * dp_dy_new;

                        rho_g2 = amrex::max(rho_g2, small_local);
                        p_g2 = amrex::max(p_g2, 1.0e-6);
                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, gamma_arr, pi_arr, p_arr, i, j_bnd - 2, k, idir, rho_g2, un_g2, ut_g2, p_g2);

                        // Layer 3: Motheau Eq. 29
                        Set::Scalar rho_g3 = 3.0 * rho_m1 + 10.0 * rho_bnd - 18.0 * rho_g1 + 6.0 * rho_g2 - 12.0 * delta * drho_dy_new;
                        Set::Scalar un_g3 = 3.0 * un_m1 + 10.0 * un_bnd - 18.0 * un_g1 + 6.0 * un_g2 - 12.0 * delta * dun_dy_new;
                        Set::Scalar ut_g3 = 3.0 * ut_m1 + 10.0 * ut_bnd - 18.0 * ut_g1 + 6.0 * ut_g2 - 12.0 * delta * dut_dy_new;
                        Set::Scalar p_g3 = 3.0 * p_m1 + 10.0 * p_bnd - 18.0 * p_g1 + 6.0 * p_g2 - 12.0 * delta * dp_dy_new;

                        rho_g3 = amrex::max(rho_g3, small_local);
                        p_g3 = amrex::max(p_g3, 1.0e-6);
                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, gamma_arr, pi_arr, p_arr, i, j_bnd - 3, k, idir, rho_g3, un_g3, ut_g3, p_g3);

                        // Layer 4: Motheau Eq. 30
                        Set::Scalar rho_g4 = -2.0 * rho_m1 - 13.0 * rho_bnd + 24.0 * rho_g1 - 12.0 * rho_g2 + 4.0 * rho_g3 + 12.0 * delta * drho_dy_new;
                        Set::Scalar un_g4 = -2.0 * un_m1 - 13.0 * un_bnd + 24.0 * un_g1 - 12.0 * un_g2 + 4.0 * un_g3 + 12.0 * delta * dun_dy_new;
                        Set::Scalar ut_g4 = -2.0 * ut_m1 - 13.0 * ut_bnd + 24.0 * ut_g1 - 12.0 * ut_g2 + 4.0 * ut_g3 + 12.0 * delta * dun_dy_new;
                        Set::Scalar p_g4 = -2.0 * p_m1 - 13.0 * p_bnd + 24.0 * p_g1 - 12.0 * p_g2 + 4.0 * p_g3 + 12.0 * delta * dp_dy_new;

                        rho_g4 = amrex::max(rho_g4, small_local);
                        p_g4 = amrex::max(p_g4, 1.0e-6);
                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, gamma_arr, pi_arr, p_arr, i, j_bnd - 4, k, idir, rho_g4, un_g4, ut_g4, p_g4);

#if AMREX_SPACEDIM == 3
                        {
                            // 2nd tangential velocity (vz, comp 2) advected shear wave + ghost KE.
                            Set::Scalar ut2_bnd = M_arr(i, j_bnd,     k, 2) / rho_bnd;
                            Set::Scalar ut2_m1  = M_arr(i, j_bnd + 1, k, 2) / (rho_m1 + small_local);
                            Set::Scalar ut2_m2  = M_arr(i, j_bnd + 2, k, 2) / (rho_m2 + small_local);
                            Set::Scalar g = (-3.0 * ut2_bnd + 4.0 * ut2_m1 - ut2_m2) / (2.0 * delta);
                            if (params_ylo_local.type == Type::Inflow) {
                                Set::Scalar K_shear = params_ylo_local.relax_w * a_bnd / params_ylo_local.L_ref;
                                g = (std::abs(un_bnd) > 1.0e-14) ? (K_shear * (ut2_bnd - params_ylo_local.target_w)) / un_bnd : 0.0;
                            }
                            Set::Scalar u1 = ut2_m1 - 2.0 * delta * g;
                            Set::Scalar u2 = -2.0*ut2_m1 - 3.0*ut2_bnd + 6.0*u1 + 6.0*delta*g;
                            Set::Scalar u3 = 3.0*ut2_m1 + 10.0*ut2_bnd - 18.0*u1 + 6.0*u2 - 12.0*delta*g;
                            Set::Scalar u4 = -2.0*ut2_m1 - 13.0*ut2_bnd + 24.0*u1 - 12.0*u2 + 4.0*u3 + 12.0*delta*g;
                            M_arr(i,j_bnd-1,k,2)=rho_g1*u1; E_arr(i,j_bnd-1,k)+=0.5*rho_g1*u1*u1;
                            M_arr(i,j_bnd-2,k,2)=rho_g2*u2; E_arr(i,j_bnd-2,k)+=0.5*rho_g2*u2*u2;
                            M_arr(i,j_bnd-3,k,2)=rho_g3*u3; E_arr(i,j_bnd-3,k)+=0.5*rho_g3*u3*u3;
                            M_arr(i,j_bnd-4,k,2)=rho_g4*u4; E_arr(i,j_bnd-4,k)+=0.5*rho_g4*u4*u4;
                        }
#endif
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
                // Fix #4: include full transverse range (see comment in xlo branch).
                int i_start = domain.smallEnd(0);
                int i_end = domain.bigEnd(0);

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
                        // CORRECTED: Removed extra division factors
                        // =============================================================
                        Set::Scalar L1 = (un_bnd - a_bnd) * (dp_dy - rho_bnd * a_bnd * dun_dy);
                        Set::Scalar L2 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd * (a2_bnd * drho_dy - dp_dy) : 0.0;
                        Set::Scalar L3 = (std::abs(un_bnd) > 1.0e-14) ? un_bnd * dut_dy : 0.0;
                        Set::Scalar L4 = (un_bnd + a_bnd) * (dp_dy + rho_bnd * a_bnd * dun_dy);

                        // =============================================================
                        // STEP 6: Transverse terms (Motheau Eq. 36-37, 45-46)
                        // For Y-normal: transverse = X-direction
                        // CORRECTED: Removed eigenvalue factors, fixed structure
                        // =============================================================
                        Set::Scalar T1 = ut_bnd * (dp_dx - rho_bnd * a_bnd * dun_dx) + gm_bnd * p_bnd * dut_dx;
                        Set::Scalar T4 = ut_bnd * (dp_dx + rho_bnd * a_bnd * dun_dx) + gm_bnd * p_bnd * dut_dx;
                        Set::Scalar T2 = (std::abs(un_bnd) > 1.0e-14) ? ut_bnd * (a2_bnd * drho_dx - dp_dx) + (a2_bnd * rho_bnd - gm_bnd * p_bnd) * dut_dx : 0.0;
                        Set::Scalar T3 = (std::abs(un_bnd) > 1.0e-14) ? ut_bnd * dut_dx + (1.0 / rho_bnd) * dp_dx : 0.0;

                        // =============================================================
                        // STEP 7: Apply BC-specific modification to incoming wave
                        // For Y-HI: L1 is incoming (speed un-a < 0 entering domain)
                        // =============================================================
#if AMREX_SPACEDIM == 3
                        // 2nd transverse (z) contribution to the transverse terms.
                        {
                            Set::Scalar ut2b = M_arr(i, j_bnd, k, 2) / rho_bnd;
                            if (k > domain.smallEnd(2) && k < domain.bigEnd(2))
                            {
                                Set::Scalar r_p,u_p,t_p,pp_p,a_p,g_p,pi_p;
                                compute_prim(eta_arr, rho_arr, M_arr, E_arr, i, j_bnd, k + 1, idir, r_p,u_p,t_p,pp_p,a_p,g_p,pi_p);
                                Set::Scalar r_m,u_m,t_m,pp_m,a_m,g_m,pi_m;
                                compute_prim(eta_arr, rho_arr, M_arr, E_arr, i, j_bnd, k - 1, idir, r_m,u_m,t_m,pp_m,a_m,g_m,pi_m);
                                Set::Scalar inv2 = 1.0 / (2.0 * dx[2]);
                                Set::Scalar dr2 = (r_p - r_m) * inv2;
                                Set::Scalar du2 = (u_p - u_m) * inv2;
                                Set::Scalar dp2 = (pp_p - pp_m) * inv2;
                                Set::Scalar dw2 = (M_arr(i,j_bnd,k+1,2)/(r_p+small_local) - M_arr(i,j_bnd,k-1,2)/(r_m+small_local)) * inv2;
                                T1 += ut2b * (dp2 - rho_bnd * a_bnd * du2) + gm_bnd * p_bnd * dw2;
                                T4 += ut2b * (dp2 + rho_bnd * a_bnd * du2) + gm_bnd * p_bnd * dw2;
                                T2 += ut2b * (a2_bnd * dr2 - dp2) + (a2_bnd * rho_bnd - gm_bnd * p_bnd) * dw2;
                            }
                        }
#endif
                        Set::Scalar K = params_yhi_local.sigma * (1.0 - M_loc * M_loc) * a_bnd  / (params_yhi_local.L_ref);

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

                        // Motheau Eq. 26 inversion for paper-style L (see corner block).
                        Set::Scalar dp_dy_new   = 0.5 * (xi1 + xi4);
                        Set::Scalar dun_dy_new  = (xi4 - xi1) / (2.0 * (rho_bnd + small_local) * a_bnd + small_local);
                        Set::Scalar drho_dy_new = (xi2 + 0.5 * (xi1 + xi4)) / (a2_bnd + small_local);
                        Set::Scalar dut_dy_new  = xi3;

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
                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, gamma_arr, pi_arr, p_arr, i, j_bnd + 1, k, idir, rho_g1, un_g1, ut_g1, p_g1);

                        // Layer 2: Motheau Eq. 32
                        Set::Scalar rho_g2 = -2.0 * rho_m1 - 3.0 * rho_bnd + 6.0 * rho_g1 - 6.0 * delta * drho_dy_new;
                        Set::Scalar un_g2 = -2.0 * un_m1 - 3.0 * un_bnd + 6.0 * un_g1 - 6.0 * delta * dun_dy_new;
                        Set::Scalar ut_g2 = -2.0 * ut_m1 - 3.0 * ut_bnd + 6.0 * ut_g1 - 6.0 * delta * dut_dy_new;
                        Set::Scalar p_g2 = -2.0 * p_m1 - 3.0 * p_bnd + 6.0 * p_g1 - 6.0 * delta * dp_dy_new;

                        rho_g2 = amrex::max(rho_g2, small_local);
                        p_g2 = amrex::max(p_g2, 1.0e-6);
                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, gamma_arr, pi_arr, p_arr, i, j_bnd + 2, k, idir, rho_g2, un_g2, ut_g2, p_g2);

                        // Layer 3: Motheau Eq. 33
                        Set::Scalar rho_g3 = 3.0 * rho_m1 + 10.0 * rho_bnd - 18.0 * rho_g1 + 6.0 * rho_g2 + 12.0 * delta * drho_dy_new;
                        Set::Scalar un_g3 = 3.0 * un_m1 + 10.0 * un_bnd - 18.0 * un_g1 + 6.0 * un_g2 + 12.0 * delta * dun_dy_new;
                        Set::Scalar ut_g3 = 3.0 * ut_m1 + 10.0 * ut_bnd - 18.0 * ut_g1 + 6.0 * ut_g2 + 12.0 * delta * dut_dy_new;
                        Set::Scalar p_g3 = 3.0 * p_m1 + 10.0 * p_bnd - 18.0 * p_g1 + 6.0 * p_g2 + 12.0 * delta * dp_dy_new;

                        rho_g3 = amrex::max(rho_g3, small_local);
                        p_g3 = amrex::max(p_g3, 1.0e-6);
                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, gamma_arr, pi_arr, p_arr, i, j_bnd + 3, k, idir, rho_g3, un_g3, ut_g3, p_g3);

                        // Layer 4: Motheau Eq. 34
                        Set::Scalar rho_g4 = -2.0 * rho_m1 - 13.0 * rho_bnd + 24.0 * rho_g1 - 12.0 * rho_g2 + 4.0 * rho_g3 - 12.0 * delta * drho_dy_new;
                        Set::Scalar un_g4 = -2.0 * un_m1 - 13.0 * un_bnd + 24.0 * un_g1 - 12.0 * un_g2 + 4.0 * un_g3 - 12.0 * delta * dun_dy_new;
                        Set::Scalar ut_g4 = -2.0 * ut_m1 - 13.0 * ut_bnd + 24.0 * ut_g1 - 12.0 * ut_g2 + 4.0 * ut_g3 - 12.0 * delta * dut_dy_new;
                        Set::Scalar p_g4 = -2.0 * p_m1 - 13.0 * p_bnd + 24.0 * p_g1 - 12.0 * p_g2 + 4.0 * p_g3 - 12.0 * delta * dp_dy_new;

                        rho_g4 = amrex::max(rho_g4, small_local);
                        p_g4 = amrex::max(p_g4, 1.0e-6);
                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, gamma_arr, pi_arr, p_arr, i, j_bnd + 4, k, idir, rho_g4, un_g4, ut_g4, p_g4);

#if AMREX_SPACEDIM == 3
                        {
                            // 2nd tangential velocity (vz, comp 2) advected shear wave + ghost KE.
                            Set::Scalar ut2_bnd = M_arr(i, j_bnd,     k, 2) / rho_bnd;
                            Set::Scalar ut2_m1  = M_arr(i, j_bnd - 1, k, 2) / (rho_m1 + small_local);
                            Set::Scalar ut2_m2  = M_arr(i, j_bnd - 2, k, 2) / (rho_m2 + small_local);
                            Set::Scalar g = (3.0 * ut2_bnd - 4.0 * ut2_m1 + ut2_m2) / (2.0 * delta);
                            if (params_yhi_local.type == Type::Inflow) {
                                Set::Scalar K_shear = params_yhi_local.relax_w * a_bnd / params_yhi_local.L_ref;
                                g = (std::abs(un_bnd) > 1.0e-14) ? (K_shear * (ut2_bnd - params_yhi_local.target_w)) / un_bnd : 0.0;
                            }
                            Set::Scalar u1 = ut2_m1 + 2.0 * delta * g;
                            Set::Scalar u2 = -2.0*ut2_m1 - 3.0*ut2_bnd + 6.0*u1 - 6.0*delta*g;
                            Set::Scalar u3 = 3.0*ut2_m1 + 10.0*ut2_bnd - 18.0*u1 + 6.0*u2 + 12.0*delta*g;
                            Set::Scalar u4 = -2.0*ut2_m1 - 13.0*ut2_bnd + 24.0*u1 - 12.0*u2 + 4.0*u3 - 12.0*delta*g;
                            M_arr(i,j_bnd+1,k,2)=rho_g1*u1; E_arr(i,j_bnd+1,k)+=0.5*rho_g1*u1*u1;
                            M_arr(i,j_bnd+2,k,2)=rho_g2*u2; E_arr(i,j_bnd+2,k)+=0.5*rho_g2*u2*u2;
                            M_arr(i,j_bnd+3,k,2)=rho_g3*u3; E_arr(i,j_bnd+3,k)+=0.5*rho_g3*u3*u3;
                            M_arr(i,j_bnd+4,k,2)=rho_g4*u4; E_arr(i,j_bnd+4,k)+=0.5*rho_g4*u4*u4;
                        }
#endif
                    }); // end ParallelFor ghost_box
                } // end i loop
            } // end Y-HI boundary

#if AMREX_SPACEDIM == 3
            // =====================================================================
            // Z-LO BOUNDARY (3D; normal=z, transverse=x; mirrors X-LO with k<->i)
            // =====================================================================
            if (bx.smallEnd(2) == domain.smallEnd(2) && params_zlo_local.type != Type::None)
            {
                int idir = 2;   // z-direction
                Set::Scalar delta = dx[2];
                int k_bnd = domain.smallEnd(2);

                // Valid transverse (x,y) plane clamped to this box (faces only;
                // edges/corners are handled separately).
                int ilo = amrex::max(bx.smallEnd(0), domain.smallEnd(0));
                int ihi = amrex::min(bx.bigEnd(0),   domain.bigEnd(0));
                int jlo = amrex::max(bx.smallEnd(1), domain.smallEnd(1));
                int jhi = amrex::min(bx.bigEnd(1),   domain.bigEnd(1));
                if (ilo <= ihi && jlo <= jhi)
                {
                    amrex::Box ghost_box;
                    ghost_box.setSmall(0, ilo);          ghost_box.setBig(0, ihi);
                    ghost_box.setSmall(1, jlo);          ghost_box.setBig(1, jhi);
                    ghost_box.setSmall(2, k_bnd - 4);    ghost_box.setBig(2, k_bnd - 1);

                    amrex::ParallelFor(ghost_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        Set::Scalar rho_bnd, un_bnd, ut_bnd, p_bnd, a_bnd, gm_bnd, pi_bnd;
                        compute_prim(eta_arr, rho_arr, M_arr, E_arr, i, j, k_bnd, idir, rho_bnd, un_bnd, ut_bnd, p_bnd, a_bnd, gm_bnd, pi_bnd);

                        Set::Scalar rho_m1, un_m1, ut_m1, p_m1, a_m1, gm_m1, pi_m1;
                        compute_prim(eta_arr, rho_arr, M_arr, E_arr, i, j, k_bnd + 1, idir, rho_m1, un_m1, ut_m1, p_m1, a_m1, gm_m1, pi_m1);
                        Set::Scalar rho_m2, un_m2, ut_m2, p_m2, a_m2, gm_m2, pi_m2;
                        compute_prim(eta_arr, rho_arr, M_arr, E_arr, i, j, k_bnd + 2, idir, rho_m2, un_m2, ut_m2, p_m2, a_m2, gm_m2, pi_m2);

                        // Normal (z) one-sided forward FD (lo side).
                        Set::Scalar drho_dx = (-3.0 * rho_bnd + 4.0 * rho_m1 - rho_m2) / (2.0 * delta);
                        Set::Scalar dun_dx  = (-3.0 * un_bnd  + 4.0 * un_m1  - un_m2 ) / (2.0 * delta);
                        Set::Scalar dut_dx  = (-3.0 * ut_bnd  + 4.0 * ut_m1  - ut_m2 ) / (2.0 * delta);
                        Set::Scalar dp_dx   = (-3.0 * p_bnd   + 4.0 * p_m1   - p_m2  ) / (2.0 * delta);

                        // Transverse (x) centered FD.
                        Set::Scalar drho_dy = 0.0, dun_dy = 0.0, dut_dy = 0.0, dp_dy = 0.0;
                        if (i > domain.smallEnd(0) && i < domain.bigEnd(0))
                        {
                            Set::Scalar rho_ip, un_ip, ut_ip, p_ip, a_ip, gm_ip, pi_ip;
                            compute_prim(eta_arr, rho_arr, M_arr, E_arr, i + 1, j, k_bnd, idir, rho_ip, un_ip, ut_ip, p_ip, a_ip, gm_ip, pi_ip);
                            Set::Scalar rho_im, un_im, ut_im, p_im, a_im, gm_im, pi_im;
                            compute_prim(eta_arr, rho_arr, M_arr, E_arr, i - 1, j, k_bnd, idir, rho_im, un_im, ut_im, p_im, a_im, gm_im, pi_im);
                            drho_dy = (rho_ip - rho_im) / (2.0 * dx[0]);
                            dun_dy  = (un_ip  - un_im ) / (2.0 * dx[0]);
                            dut_dy  = (ut_ip  - ut_im ) / (2.0 * dx[0]);
                            dp_dy   = (p_ip   - p_im  ) / (2.0 * dx[0]);
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

                        // lo side: L4 is incoming.
                        // 2nd transverse (y) contribution to the transverse terms.
                        {
                            Set::Scalar ut2b = M_arr(i, j, k_bnd, 1) / rho_bnd;
                            if (j > domain.smallEnd(1) && j < domain.bigEnd(1))
                            {
                                Set::Scalar r_p,u_p,t_p,pp_p,a_p,g_p,pi_p;
                                compute_prim(eta_arr, rho_arr, M_arr, E_arr, i, j + 1, k_bnd, idir, r_p,u_p,t_p,pp_p,a_p,g_p,pi_p);
                                Set::Scalar r_m,u_m,t_m,pp_m,a_m,g_m,pi_m;
                                compute_prim(eta_arr, rho_arr, M_arr, E_arr, i, j - 1, k_bnd, idir, r_m,u_m,t_m,pp_m,a_m,g_m,pi_m);
                                Set::Scalar inv2 = 1.0 / (2.0 * dx[1]);
                                Set::Scalar dr2 = (r_p - r_m) * inv2;
                                Set::Scalar du2 = (u_p - u_m) * inv2;
                                Set::Scalar dp2 = (pp_p - pp_m) * inv2;
                                Set::Scalar dw2 = (M_arr(i,j+1,k_bnd,1)/(r_p+small_local) - M_arr(i,j-1,k_bnd,1)/(r_m+small_local)) * inv2;
                                T1 += ut2b * (dp2 - rho_bnd * a_bnd * du2) + gm_bnd * p_bnd * dw2;
                                T4 += ut2b * (dp2 + rho_bnd * a_bnd * du2) + gm_bnd * p_bnd * dw2;
                                T2 += ut2b * (a2_bnd * dr2 - dp2) + (a2_bnd * rho_bnd - gm_bnd * p_bnd) * dw2;
                            }
                        }
                        Set::Scalar K = params_zlo_local.sigma * (1.0 - M_loc * M_loc) * a_bnd / (params_zlo_local.L_ref);
                        if (params_zlo_local.type == Type::Outflow)
                        {
                            L4 = K * (p_bnd - params_zlo_local.target_p) - (1.0 - params_zlo_local.beta) * T4;
                        }
                        else if (params_zlo_local.type == Type::Inflow)
                        {
                            Set::Scalar er = params_zlo_local.relax_u;
                            Set::Scalar K_acoustic = er * rho_bnd * (1.0 - M_loc * M_loc) / params_zlo_local.L_ref;
                            Set::Scalar K_entropy  = er * rho_bnd * a_bnd * (gm_bnd - 1.0) / params_zlo_local.L_ref;
                            Set::Scalar K_shear    = er * a_bnd / params_zlo_local.L_ref;
                            Set::Scalar eta_bnd = eta_arr(i, j, k_bnd);
                            Set::Scalar cv_mix = eta_bnd * cv0_ + (1.0 - eta_bnd) * cv1_;
                            Set::Scalar T_bnd = (p_bnd + pi_bnd) / (rho_bnd * cv_mix * (gm_bnd - 1.0) + small_local);
                            L4 = K_acoustic * (un_bnd - params_zlo_local.target_u) - T4;
                            L2 = K_entropy  * (T_bnd - params_zlo_local.target_T) - T2;
                            L3 = K_shear    * (ut_bnd - params_zlo_local.target_v) - T3;
                        }

                        Set::Scalar xi1 = L1 / (un_bnd - a_bnd + small_local);
                        Set::Scalar xi2 = (std::abs(un_bnd) > 1.0e-14) ? L2 / un_bnd : 0.0;
                        Set::Scalar xi3 = (std::abs(un_bnd) > 1.0e-14) ? L3 / un_bnd : 0.0;
                        Set::Scalar xi4 = L4 / (un_bnd + a_bnd + small_local);

                        Set::Scalar dp_dx_new   = 0.5 * (xi1 + xi4);
                        Set::Scalar dun_dx_new  = (xi4 - xi1) / (2.0 * (rho_bnd + small_local) * a_bnd + small_local);
                        Set::Scalar drho_dx_new = (xi2 + 0.5 * (xi1 + xi4)) / (a2_bnd + small_local);
                        Set::Scalar dut_dx_new  = xi3;

                        // Ghost layers (lo-side Motheau Eq. 27-30), in -z.
                        Set::Scalar rho_g1 = rho_m1 - 2.0 * delta * drho_dx_new;
                        Set::Scalar un_g1  = un_m1  - 2.0 * delta * dun_dx_new;
                        Set::Scalar ut_g1  = ut_m1  - 2.0 * delta * dut_dx_new;
                        Set::Scalar p_g1   = p_m1   - 2.0 * delta * dp_dx_new;
                        rho_g1 = amrex::max(rho_g1, small_local); p_g1 = amrex::max(p_g1, 1.0e-6);
                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, gamma_arr, pi_arr, p_arr, i, j, k_bnd - 1, idir, rho_g1, un_g1, ut_g1, p_g1);

                        Set::Scalar rho_g2 = -2.0 * rho_m1 - 3.0 * rho_bnd + 6.0 * rho_g1 + 6.0 * delta * drho_dx_new;
                        Set::Scalar un_g2  = -2.0 * un_m1  - 3.0 * un_bnd  + 6.0 * un_g1  + 6.0 * delta * dun_dx_new;
                        Set::Scalar ut_g2  = -2.0 * ut_m1  - 3.0 * ut_bnd  + 6.0 * ut_g1  + 6.0 * delta * dut_dx_new;
                        Set::Scalar p_g2   = -2.0 * p_m1   - 3.0 * p_bnd   + 6.0 * p_g1   + 6.0 * delta * dp_dx_new;
                        rho_g2 = amrex::max(rho_g2, small_local); p_g2 = amrex::max(p_g2, 1.0e-6);
                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, gamma_arr, pi_arr, p_arr, i, j, k_bnd - 2, idir, rho_g2, un_g2, ut_g2, p_g2);

                        Set::Scalar rho_g3 = 3.0 * rho_m1 + 10.0 * rho_bnd - 18.0 * rho_g1 + 6.0 * rho_g2 - 12.0 * delta * drho_dx_new;
                        Set::Scalar un_g3  = 3.0 * un_m1  + 10.0 * un_bnd  - 18.0 * un_g1  + 6.0 * un_g2  - 12.0 * delta * dun_dx_new;
                        Set::Scalar ut_g3  = 3.0 * ut_m1  + 10.0 * ut_bnd  - 18.0 * ut_g1  + 6.0 * ut_g2  - 12.0 * delta * dut_dx_new;
                        Set::Scalar p_g3   = 3.0 * p_m1   + 10.0 * p_bnd   - 18.0 * p_g1   + 6.0 * p_g2   - 12.0 * delta * dp_dx_new;
                        rho_g3 = amrex::max(rho_g3, small_local); p_g3 = amrex::max(p_g3, 1.0e-6);
                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, gamma_arr, pi_arr, p_arr, i, j, k_bnd - 3, idir, rho_g3, un_g3, ut_g3, p_g3);

                        Set::Scalar rho_g4 = -2.0 * rho_m1 - 13.0 * rho_bnd + 24.0 * rho_g1 - 12.0 * rho_g2 + 4.0 * rho_g3 + 12.0 * delta * drho_dx_new;
                        Set::Scalar un_g4  = -2.0 * un_m1  - 13.0 * un_bnd  + 24.0 * un_g1  - 12.0 * un_g2  + 4.0 * un_g3  + 12.0 * delta * dun_dx_new;
                        Set::Scalar ut_g4  = -2.0 * ut_m1  - 13.0 * ut_bnd  + 24.0 * ut_g1  - 12.0 * ut_g2  + 4.0 * ut_g3  + 12.0 * delta * dut_dx_new;
                        Set::Scalar p_g4   = -2.0 * p_m1   - 13.0 * p_bnd   + 24.0 * p_g1   - 12.0 * p_g2   + 4.0 * p_g3   + 12.0 * delta * dp_dx_new;
                        rho_g4 = amrex::max(rho_g4, small_local); p_g4 = amrex::max(p_g4, 1.0e-6);
                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, gamma_arr, pi_arr, p_arr, i, j, k_bnd - 4, idir, rho_g4, un_g4, ut_g4, p_g4);

                        {
                            // 2nd tangential velocity (vy, comp 1) advected shear wave + ghost KE.
                            Set::Scalar ut2_bnd = M_arr(i, j, k_bnd,     1) / rho_bnd;
                            Set::Scalar ut2_m1  = M_arr(i, j, k_bnd + 1, 1) / (rho_m1 + small_local);
                            Set::Scalar ut2_m2  = M_arr(i, j, k_bnd + 2, 1) / (rho_m2 + small_local);
                            Set::Scalar g = (-3.0 * ut2_bnd + 4.0 * ut2_m1 - ut2_m2) / (2.0 * delta);
                            if (params_zlo_local.type == Type::Inflow) {
                                Set::Scalar K_shear = params_zlo_local.relax_w * a_bnd / params_zlo_local.L_ref;
                                g = (std::abs(un_bnd) > 1.0e-14) ? (K_shear * (ut2_bnd - params_zlo_local.target_w)) / un_bnd : 0.0;
                            }
                            Set::Scalar u1 = ut2_m1 - 2.0 * delta * g;
                            Set::Scalar u2 = -2.0*ut2_m1 - 3.0*ut2_bnd + 6.0*u1 + 6.0*delta*g;
                            Set::Scalar u3 = 3.0*ut2_m1 + 10.0*ut2_bnd - 18.0*u1 + 6.0*u2 - 12.0*delta*g;
                            Set::Scalar u4 = -2.0*ut2_m1 - 13.0*ut2_bnd + 24.0*u1 - 12.0*u2 + 4.0*u3 + 12.0*delta*g;
                            M_arr(i,j,k_bnd-1,1)=rho_g1*u1; E_arr(i,j,k_bnd-1)+=0.5*rho_g1*u1*u1;
                            M_arr(i,j,k_bnd-2,1)=rho_g2*u2; E_arr(i,j,k_bnd-2)+=0.5*rho_g2*u2*u2;
                            M_arr(i,j,k_bnd-3,1)=rho_g3*u3; E_arr(i,j,k_bnd-3)+=0.5*rho_g3*u3*u3;
                            M_arr(i,j,k_bnd-4,1)=rho_g4*u4; E_arr(i,j,k_bnd-4)+=0.5*rho_g4*u4*u4;
                        }
                    }); // end ParallelFor Z-LO
                }
            } // end Z-LO boundary

            // =====================================================================
            // Z-HI BOUNDARY (3D; normal=z, transverse=x; mirrors X-HI with k<->i)
            // =====================================================================
            if (bx.bigEnd(2) == domain.bigEnd(2) && params_zhi_local.type != Type::None)
            {
                int idir = 2;   // z-direction
                Set::Scalar delta = dx[2];
                int k_bnd = domain.bigEnd(2);

                int ilo = amrex::max(bx.smallEnd(0), domain.smallEnd(0));
                int ihi = amrex::min(bx.bigEnd(0),   domain.bigEnd(0));
                int jlo = amrex::max(bx.smallEnd(1), domain.smallEnd(1));
                int jhi = amrex::min(bx.bigEnd(1),   domain.bigEnd(1));
                if (ilo <= ihi && jlo <= jhi)
                {
                    amrex::Box ghost_box;
                    ghost_box.setSmall(0, ilo);          ghost_box.setBig(0, ihi);
                    ghost_box.setSmall(1, jlo);          ghost_box.setBig(1, jhi);
                    ghost_box.setSmall(2, k_bnd + 1);    ghost_box.setBig(2, k_bnd + 4);

                    amrex::ParallelFor(ghost_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        Set::Scalar rho_bnd, un_bnd, ut_bnd, p_bnd, a_bnd, gm_bnd, pi_bnd;
                        compute_prim(eta_arr, rho_arr, M_arr, E_arr, i, j, k_bnd, idir, rho_bnd, un_bnd, ut_bnd, p_bnd, a_bnd, gm_bnd, pi_bnd);

                        Set::Scalar rho_m1, un_m1, ut_m1, p_m1, a_m1, gm_m1, pi_m1;
                        compute_prim(eta_arr, rho_arr, M_arr, E_arr, i, j, k_bnd - 1, idir, rho_m1, un_m1, ut_m1, p_m1, a_m1, gm_m1, pi_m1);
                        Set::Scalar rho_m2, un_m2, ut_m2, p_m2, a_m2, gm_m2, pi_m2;
                        compute_prim(eta_arr, rho_arr, M_arr, E_arr, i, j, k_bnd - 2, idir, rho_m2, un_m2, ut_m2, p_m2, a_m2, gm_m2, pi_m2);

                        // Normal (z) one-sided backward FD (hi side).
                        Set::Scalar drho_dx = (3.0 * rho_bnd - 4.0 * rho_m1 + rho_m2) / (2.0 * delta);
                        Set::Scalar dun_dx  = (3.0 * un_bnd  - 4.0 * un_m1  + un_m2 ) / (2.0 * delta);
                        Set::Scalar dut_dx  = (3.0 * ut_bnd  - 4.0 * ut_m1  + ut_m2 ) / (2.0 * delta);
                        Set::Scalar dp_dx   = (3.0 * p_bnd   - 4.0 * p_m1   + p_m2  ) / (2.0 * delta);

                        Set::Scalar drho_dy = 0.0, dun_dy = 0.0, dut_dy = 0.0, dp_dy = 0.0;
                        if (i > domain.smallEnd(0) && i < domain.bigEnd(0))
                        {
                            Set::Scalar rho_ip, un_ip, ut_ip, p_ip, a_ip, gm_ip, pi_ip;
                            compute_prim(eta_arr, rho_arr, M_arr, E_arr, i + 1, j, k_bnd, idir, rho_ip, un_ip, ut_ip, p_ip, a_ip, gm_ip, pi_ip);
                            Set::Scalar rho_im, un_im, ut_im, p_im, a_im, gm_im, pi_im;
                            compute_prim(eta_arr, rho_arr, M_arr, E_arr, i - 1, j, k_bnd, idir, rho_im, un_im, ut_im, p_im, a_im, gm_im, pi_im);
                            drho_dy = (rho_ip - rho_im) / (2.0 * dx[0]);
                            dun_dy  = (un_ip  - un_im ) / (2.0 * dx[0]);
                            dut_dy  = (ut_ip  - ut_im ) / (2.0 * dx[0]);
                            dp_dy   = (p_ip   - p_im  ) / (2.0 * dx[0]);
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

                        // hi side: L1 is incoming.
                        // 2nd transverse (y) contribution to the transverse terms.
                        {
                            Set::Scalar ut2b = M_arr(i, j, k_bnd, 1) / rho_bnd;
                            if (j > domain.smallEnd(1) && j < domain.bigEnd(1))
                            {
                                Set::Scalar r_p,u_p,t_p,pp_p,a_p,g_p,pi_p;
                                compute_prim(eta_arr, rho_arr, M_arr, E_arr, i, j + 1, k_bnd, idir, r_p,u_p,t_p,pp_p,a_p,g_p,pi_p);
                                Set::Scalar r_m,u_m,t_m,pp_m,a_m,g_m,pi_m;
                                compute_prim(eta_arr, rho_arr, M_arr, E_arr, i, j - 1, k_bnd, idir, r_m,u_m,t_m,pp_m,a_m,g_m,pi_m);
                                Set::Scalar inv2 = 1.0 / (2.0 * dx[1]);
                                Set::Scalar dr2 = (r_p - r_m) * inv2;
                                Set::Scalar du2 = (u_p - u_m) * inv2;
                                Set::Scalar dp2 = (pp_p - pp_m) * inv2;
                                Set::Scalar dw2 = (M_arr(i,j+1,k_bnd,1)/(r_p+small_local) - M_arr(i,j-1,k_bnd,1)/(r_m+small_local)) * inv2;
                                T1 += ut2b * (dp2 - rho_bnd * a_bnd * du2) + gm_bnd * p_bnd * dw2;
                                T4 += ut2b * (dp2 + rho_bnd * a_bnd * du2) + gm_bnd * p_bnd * dw2;
                                T2 += ut2b * (a2_bnd * dr2 - dp2) + (a2_bnd * rho_bnd - gm_bnd * p_bnd) * dw2;
                            }
                        }
                        Set::Scalar K = params_zhi_local.sigma * (1.0 - M_loc * M_loc) * a_bnd / (params_zhi_local.L_ref);
                        if (params_zhi_local.type == Type::Outflow)
                        {
                            L1 = K * (p_bnd - params_zhi_local.target_p) - (1.0 - params_zhi_local.beta) * T1;
                        }
                        else if (params_zhi_local.type == Type::Inflow)
                        {
                            Set::Scalar er = params_zhi_local.relax_u;
                            Set::Scalar K_acoustic = er * rho_bnd * (1.0 - M_loc * M_loc) / params_zhi_local.L_ref;
                            Set::Scalar K_entropy  = er * rho_bnd * a_bnd * (gm_bnd - 1.0) / params_zhi_local.L_ref;
                            Set::Scalar K_shear    = er * a_bnd / params_zhi_local.L_ref;
                            Set::Scalar eta_bnd = eta_arr(i, j, k_bnd);
                            Set::Scalar cv_mix = eta_bnd * cv0_ + (1.0 - eta_bnd) * cv1_;
                            Set::Scalar T_bnd = (p_bnd + pi_bnd) / (rho_bnd * cv_mix * (gm_bnd - 1.0) + small_local);
                            L1 = -K_acoustic * (un_bnd - params_zhi_local.target_u) - T1;
                            L2 = K_entropy  * (T_bnd - params_zhi_local.target_T) - T2;
                            L3 = K_shear    * (ut_bnd - params_zhi_local.target_v) - T3;
                        }

                        Set::Scalar xi1 = L1 / (un_bnd - a_bnd + small_local);
                        Set::Scalar xi2 = (std::abs(un_bnd) > 1.0e-14) ? L2 / un_bnd : 0.0;
                        Set::Scalar xi3 = (std::abs(un_bnd) > 1.0e-14) ? L3 / un_bnd : 0.0;
                        Set::Scalar xi4 = L4 / (un_bnd + a_bnd + small_local);

                        Set::Scalar dp_dx_new   = 0.5 * (xi1 + xi4);
                        Set::Scalar dun_dx_new  = (xi4 - xi1) / (2.0 * (rho_bnd + small_local) * a_bnd + small_local);
                        Set::Scalar drho_dx_new = (xi2 + 0.5 * (xi1 + xi4)) / (a2_bnd + small_local);
                        Set::Scalar dut_dx_new  = xi3;

                        // Ghost layers (hi-side Motheau Eq. 31-34), in +z.
                        Set::Scalar rho_g1 = rho_m1 + 2.0 * delta * drho_dx_new;
                        Set::Scalar un_g1  = un_m1  + 2.0 * delta * dun_dx_new;
                        Set::Scalar ut_g1  = ut_m1  + 2.0 * delta * dut_dx_new;
                        Set::Scalar p_g1   = p_m1   + 2.0 * delta * dp_dx_new;
                        rho_g1 = amrex::max(rho_g1, small_local); p_g1 = amrex::max(p_g1, 1.0e-6);
                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, gamma_arr, pi_arr, p_arr, i, j, k_bnd + 1, idir, rho_g1, un_g1, ut_g1, p_g1);

                        Set::Scalar rho_g2 = -2.0 * rho_m1 - 3.0 * rho_bnd + 6.0 * rho_g1 - 6.0 * delta * drho_dx_new;
                        Set::Scalar un_g2  = -2.0 * un_m1  - 3.0 * un_bnd  + 6.0 * un_g1  - 6.0 * delta * dun_dx_new;
                        Set::Scalar ut_g2  = -2.0 * ut_m1  - 3.0 * ut_bnd  + 6.0 * ut_g1  - 6.0 * delta * dut_dx_new;
                        Set::Scalar p_g2   = -2.0 * p_m1   - 3.0 * p_bnd   + 6.0 * p_g1   - 6.0 * delta * dp_dx_new;
                        rho_g2 = amrex::max(rho_g2, small_local); p_g2 = amrex::max(p_g2, 1.0e-6);
                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, gamma_arr, pi_arr, p_arr, i, j, k_bnd + 2, idir, rho_g2, un_g2, ut_g2, p_g2);

                        Set::Scalar rho_g3 = 3.0 * rho_m1 + 10.0 * rho_bnd - 18.0 * rho_g1 + 6.0 * rho_g2 + 12.0 * delta * drho_dx_new;
                        Set::Scalar un_g3  = 3.0 * un_m1  + 10.0 * un_bnd  - 18.0 * un_g1  + 6.0 * un_g2  + 12.0 * delta * dun_dx_new;
                        Set::Scalar ut_g3  = 3.0 * ut_m1  + 10.0 * ut_bnd  - 18.0 * ut_g1  + 6.0 * ut_g2  + 12.0 * delta * dut_dx_new;
                        Set::Scalar p_g3   = 3.0 * p_m1   + 10.0 * p_bnd   - 18.0 * p_g1   + 6.0 * p_g2   + 12.0 * delta * dp_dx_new;
                        rho_g3 = amrex::max(rho_g3, small_local); p_g3 = amrex::max(p_g3, 1.0e-6);
                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, gamma_arr, pi_arr, p_arr, i, j, k_bnd + 3, idir, rho_g3, un_g3, ut_g3, p_g3);

                        Set::Scalar rho_g4 = -2.0 * rho_m1 - 13.0 * rho_bnd + 24.0 * rho_g1 - 12.0 * rho_g2 + 4.0 * rho_g3 - 12.0 * delta * drho_dx_new;
                        Set::Scalar un_g4  = -2.0 * un_m1  - 13.0 * un_bnd  + 24.0 * un_g1  - 12.0 * un_g2  + 4.0 * un_g3  - 12.0 * delta * dun_dx_new;
                        Set::Scalar ut_g4  = -2.0 * ut_m1  - 13.0 * ut_bnd  + 24.0 * ut_g1  - 12.0 * ut_g2  + 4.0 * ut_g3  - 12.0 * delta * dut_dx_new;
                        Set::Scalar p_g4   = -2.0 * p_m1   - 13.0 * p_bnd   + 24.0 * p_g1   - 12.0 * p_g2   + 4.0 * p_g3   - 12.0 * delta * dp_dx_new;
                        rho_g4 = amrex::max(rho_g4, small_local); p_g4 = amrex::max(p_g4, 1.0e-6);
                        set_ghost_cons(eta_arr, rho_arr, M_arr, E_arr, gamma_arr, pi_arr, p_arr, i, j, k_bnd + 4, idir, rho_g4, un_g4, ut_g4, p_g4);

                        {
                            // 2nd tangential velocity (vy, comp 1) advected shear wave + ghost KE.
                            Set::Scalar ut2_bnd = M_arr(i, j, k_bnd,     1) / rho_bnd;
                            Set::Scalar ut2_m1  = M_arr(i, j, k_bnd - 1, 1) / (rho_m1 + small_local);
                            Set::Scalar ut2_m2  = M_arr(i, j, k_bnd - 2, 1) / (rho_m2 + small_local);
                            Set::Scalar g = (3.0 * ut2_bnd - 4.0 * ut2_m1 + ut2_m2) / (2.0 * delta);
                            if (params_zhi_local.type == Type::Inflow) {
                                Set::Scalar K_shear = params_zhi_local.relax_w * a_bnd / params_zhi_local.L_ref;
                                g = (std::abs(un_bnd) > 1.0e-14) ? (K_shear * (ut2_bnd - params_zhi_local.target_w)) / un_bnd : 0.0;
                            }
                            Set::Scalar u1 = ut2_m1 + 2.0 * delta * g;
                            Set::Scalar u2 = -2.0*ut2_m1 - 3.0*ut2_bnd + 6.0*u1 - 6.0*delta*g;
                            Set::Scalar u3 = 3.0*ut2_m1 + 10.0*ut2_bnd - 18.0*u1 + 6.0*u2 + 12.0*delta*g;
                            Set::Scalar u4 = -2.0*ut2_m1 - 13.0*ut2_bnd + 24.0*u1 - 12.0*u2 + 4.0*u3 - 12.0*delta*g;
                            M_arr(i,j,k_bnd+1,1)=rho_g1*u1; E_arr(i,j,k_bnd+1)+=0.5*rho_g1*u1*u1;
                            M_arr(i,j,k_bnd+2,1)=rho_g2*u2; E_arr(i,j,k_bnd+2)+=0.5*rho_g2*u2*u2;
                            M_arr(i,j,k_bnd+3,1)=rho_g3*u3; E_arr(i,j,k_bnd+3)+=0.5*rho_g3*u3*u3;
                            M_arr(i,j,k_bnd+4,1)=rho_g4*u4; E_arr(i,j,k_bnd+4)+=0.5*rho_g4*u4*u4;
                        }
                    }); // end ParallelFor Z-HI
                }
            } // end Z-HI boundary
#endif

        } // end MFIter loop

    } // end FillBoundary()

private:
    /// Parse parameters for a single boundary face
    static BoundaryParams ParseFace(IO::ParmParse &pp, std::string face_name);

}; // end class NSCBC4

} // namespace BC

#endif // BC_NSCBC4_H

