#ifndef BC_NSCBC_H
#define BC_NSCBC_H

#include "BC/BC.H"
#include "Solver/EOS/EOS.H"
#include "IO/ParmParse.H"
#include "Set/Set.H"
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>

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
class NSCBC
{
public:
    enum class Type
    {
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
        Set::Scalar sigma = 0.25; ///< Outflow pressure relaxation coefficient

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
    /// @param density_mf    Density field [kg/m³]
    /// @param momentum_mf   Momentum field [kg/(m²·s)]
    /// @param energy_mf     Total energy field [J/m³]
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

        // Capture parameters for GPU
        BoundaryParams params_xlo_local = params_xlo;
        BoundaryParams params_xhi_local = params_xhi;
        BoundaryParams params_ylo_local = params_ylo;
        BoundaryParams params_yhi_local = params_yhi;
#if AMREX_SPACEDIM == 3
        BoundaryParams params_zlo_local = params_zlo;
        BoundaryParams params_zhi_local = params_zhi;
#endif
        Set::Scalar small_local = small;
        Set::Scalar pref_local = pref;

        // Loop over MultiFab boxes
        for (amrex::MFIter mfi(density_mf); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();

            // Get array views
            amrex::Array4<Set::Scalar> const &rho_arr = density_mf.array(mfi);
            amrex::Array4<Set::Scalar> const &momx_arr = momentum_mf.array(mfi, 0);
            amrex::Array4<Set::Scalar> const &momy_arr = momentum_mf.array(mfi, 1);
#if AMREX_SPACEDIM == 3
            amrex::Array4<Set::Scalar> const &momz_arr = momentum_mf.array(mfi, 2);
#endif
            amrex::Array4<Set::Scalar> const &energy_arr = energy_mf.array(mfi);
            amrex::Array4<Set::Scalar> const &eta_arr = eta_mf.array(mfi);

            // Allocate temporary arrays for primitive variables
            // These will be computed from conservative variables
            amrex::Box grown_box = amrex::grow(bx, 4);
            amrex::FArrayBox u_fab(grown_box, 1);
            amrex::FArrayBox v_fab(grown_box, 1);
            amrex::FArrayBox w_fab(grown_box, 1);
            amrex::FArrayBox p_fab(grown_box, 1);
            amrex::FArrayBox T_fab(grown_box, 1);
            amrex::FArrayBox c_fab(grown_box, 1);
            amrex::FArrayBox gamma_fab(grown_box, 1);
            amrex::FArrayBox p0_fab(grown_box, 1);

            amrex::Array4<Set::Scalar> const &u_arr = u_fab.array();
            amrex::Array4<Set::Scalar> const &v_arr = v_fab.array();
            amrex::Array4<Set::Scalar> const &w_arr = w_fab.array();
            amrex::Array4<Set::Scalar> const &p_arr = p_fab.array();
            amrex::Array4<Set::Scalar> const &T_arr = T_fab.array();
            amrex::Array4<Set::Scalar> const &c_arr = c_fab.array();
            amrex::Array4<Set::Scalar> const &gamma_arr = gamma_fab.array();
            amrex::Array4<Set::Scalar> const &p0_arr = p0_fab.array();

            // Step 1: Convert conservative to primitive variables in interior
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Set::Scalar rho = rho_arr(i, j, k);
                Set::Scalar momx = momx_arr(i, j, k);
                Set::Scalar momy = momy_arr(i, j, k);
#if AMREX_SPACEDIM == 3
                Set::Scalar momz = momz_arr(i, j, k);
#else
            Set::Scalar momz = 0.0;
#endif
                Set::Scalar E = energy_arr(i, j, k);
                Set::Scalar eta_val = eta_arr(i, j, k);

                // Velocities
                u_arr(i, j, k) = momx / (rho + small_local);
                v_arr(i, j, k) = momy / (rho + small_local);
                w_arr(i, j, k) = momz / (rho + small_local);

                // Kinetic energy
                Set::Scalar ke = 0.5 * rho * (u_arr(i, j, k) * u_arr(i, j, k) + v_arr(i, j, k) * v_arr(i, j, k) + w_arr(i, j, k) * w_arr(i, j, k));

                // Internal energy
                Set::Scalar ue = E - ke;

                // Mixed EOS properties
                gamma_arr(i, j, k) = Solver::EOS::EOS::MixedGamma(eta_val, eos0, eos1);
                p0_arr(i, j, k) = Solver::EOS::EOS::MixedP0(eta_val, eos0, eos1);

                // Pressure from Tammann EOS
                p_arr(i, j, k) = (gamma_arr(i, j, k) - 1.0) * ue - gamma_arr(i, j, k) * p0_arr(i, j, k) + pref_local;
                p_arr(i, j, k) = std::max(small_local, p_arr(i, j, k));

                // Temperature
                T_arr(i, j, k) = Solver::EOS::EOS::MixedTemperature(rho, p_arr(i, j, k), eta_val, eos0, eos1, pref_local);

                // Sound speed
                c_arr(i, j, k) = Solver::EOS::EOS::SafeSoundSpeed(rho, p_arr(i, j, k), gamma_arr(i, j, k), p0_arr(i, j, k), small_local);
            });

            // Step 2: Apply NSCBC to each boundary face
            // Note: Corners must be filled first (as per Motheau et al. 2018)

            // ====================================================================
            // X-LO BOUNDARY (i = domain.smallEnd(0))
            // ====================================================================
            if (bx.smallEnd(0) == domain.smallEnd(0))
            {
                int idir = 0;   // x-direction
                int isign = -1; // left boundary
                Set::Scalar delta = dx[0];

                amrex::Box ghost_box = amrex::adjCellLo(bx, 0, 1);

                amrex::ParallelFor(ghost_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    // Boundary cell index (last interior cell)
                    int i_bnd = domain.smallEnd(0);

                    // Get boundary state
                    Set::Scalar rho_bnd = rho_arr(i_bnd, j, k);
                    Set::Scalar u_bnd = u_arr(i_bnd, j, k);
                    Set::Scalar v_bnd = v_arr(i_bnd, j, k);
                    Set::Scalar w_bnd = w_arr(i_bnd, j, k);
                    Set::Scalar p_bnd = p_arr(i_bnd, j, k);
                    Set::Scalar T_bnd = T_arr(i_bnd, j, k);
                    Set::Scalar c_bnd = c_arr(i_bnd, j, k);
                    Set::Scalar gamma_bnd = gamma_arr(i_bnd, j, k);
                    Set::Scalar p0_bnd = p0_arr(i_bnd, j, k);

                    // Compute normal derivatives (one-sided, 2nd order)
                    // dQ/dx ~= (-3Q_i + 4Q_{i+1} - Q_{i+2}) / (2dx)
                    Set::Scalar drho_dx = (-3.0 * rho_arr(i_bnd, j, k) + 4.0 * rho_arr(i_bnd + 1, j, k) - rho_arr(i_bnd + 2, j, k)) / (2.0 * delta);
                    Set::Scalar du_dx = (-3.0 * u_arr(i_bnd, j, k) + 4.0 * u_arr(i_bnd + 1, j, k) - u_arr(i_bnd + 2, j, k)) / (2.0 * delta);
                    Set::Scalar dv_dx = (-3.0 * v_arr(i_bnd, j, k) + 4.0 * v_arr(i_bnd + 1, j, k) - v_arr(i_bnd + 2, j, k)) / (2.0 * delta);
                    Set::Scalar dw_dx = (-3.0 * w_arr(i_bnd, j, k) + 4.0 * w_arr(i_bnd + 1, j, k) - w_arr(i_bnd + 2, j, k)) / (2.0 * delta);
                    Set::Scalar dp_dx = (-3.0 * p_arr(i_bnd, j, k) + 4.0 * p_arr(i_bnd + 1, j, k) - p_arr(i_bnd + 2, j, k)) / (2.0 * delta);

                    // Compute tangential derivatives (centered, 2nd order)
                    // Y-direction: dQ/dy ~= (Q_{j+1} - Q_{j-1}) / (2dy)
                    Set::Scalar drho_dy = 0.0, du_dy = 0.0, dv_dy = 0.0, dw_dy = 0.0, dp_dy = 0.0;
                    if (j > domain.smallEnd(1) && j < domain.bigEnd(1))
                    {
                        drho_dy = (rho_arr(i_bnd, j + 1, k) - rho_arr(i_bnd, j - 1, k)) / (2.0 * dx[1]);
                        du_dy = (u_arr(i_bnd, j + 1, k) - u_arr(i_bnd, j - 1, k)) / (2.0 * dx[1]);
                        dv_dy = (v_arr(i_bnd, j + 1, k) - v_arr(i_bnd, j - 1, k)) / (2.0 * dx[1]);
                        dw_dy = (w_arr(i_bnd, j + 1, k) - w_arr(i_bnd, j - 1, k)) / (2.0 * dx[1]);
                        dp_dy = (p_arr(i_bnd, j + 1, k) - p_arr(i_bnd, j - 1, k)) / (2.0 * dx[1]);
                    }

#if AMREX_SPACEDIM == 3
                    // Z-direction: dQ/dz ~= (Q_{k+1} - Q_{k-1}) / (2dz)
                    Set::Scalar drho_dz = 0.0, du_dz = 0.0, dv_dz = 0.0, dw_dz = 0.0, dp_dz = 0.0;
                    if (k > domain.smallEnd(2) && k < domain.bigEnd(2))
                    {
                        drho_dz = (rho_arr(i_bnd, j, k + 1) - rho_arr(i_bnd, j, k - 1)) / (2.0 * dx, [2]);
                        du_dz = (u_arr(i_bnd, j, k + 1) - u_arr(i_bnd, j, k - 1)) / (2.0 * dx, [2]);
                        dv_dz = (v_arr(i_bnd, j, k + 1) - v_arr(i_bnd, j, k - 1)) / (2.0 * dx, [2]);
                        dw_dz = (w_arr(i_bnd, j, k + 1) - w_arr(i_bnd, j, k - 1)) / (2.0 * dx, [2]);
                        dp_dz = (p_arr(i_bnd, j, k + 1) - p_arr(i_bnd, j, k - 1)) / (2.0 * dx, [2]);
                    }
#else
                Set::Scalar drho_dz = 0.0, du_dz = 0.0, dv_dz = 0.0, dw_dz = 0.0, dp_dz = 0.0;
#endif

                    // Compute transverse terms (Section III.B, Motheau et al. 2018)
                    // For x-normal boundary: transverse = y and z directions
                    // T_1 = v*dp/dy + w*dp/dz - rhoc^2(v*drho/dy + w*drho/dz)
                    Set::Scalar T1 = v_bnd * dp_dy + w_bnd * dp_dz - rho_bnd * c_bnd * c_bnd * (v_bnd * drho_dy + w_bnd * drho_dz);
                    Set::Scalar T2 = v_bnd * du_dy + w_bnd * du_dz;
                    Set::Scalar T3 = v_bnd * dv_dy + w_bnd * dv_dz;
                    Set::Scalar T4 = v_bnd * dw_dy + w_bnd * dw_dz;
                    Set::Scalar T5 = v_bnd * dp_dy + w_bnd * dp_dz;

                    // Compute characteristic waves (LODI system)
                    // Wave speeds: A_1 = u - c, A_2 = u, A_3 = u, A_4 = u, A_5 = u + c
                    Set::Scalar lambda1 = u_bnd - c_bnd;
                    Set::Scalar lambda2 = u_bnd;
                    Set::Scalar lambda3 = u_bnd;
                    Set::Scalar lambda4 = u_bnd;
                    Set::Scalar lambda5 = u_bnd + c_bnd;

                    // Interior wave amplitudes (from spatial derivatives)
                    // Equations from Motheau et al. (2018), Section III
                    Set::Scalar L1_int = lambda1 * (dp_dx - rho_bnd * c_bnd * du_dx);
                    Set::Scalar L2_int = lambda2 * (c_bnd * c_bnd * drho_dx - dp_dx);
                    Set::Scalar L3_int = lambda3 * dv_dx;
                    Set::Scalar L4_int = lambda4 * dw_dx;
                    Set::Scalar L5_int = lambda5 * (dp_dx + rho_bnd * c_bnd * du_dx);

                    // Add transverse terms (weighted by Beta)
                    L1_int += params_xlo_local.beta * T1;
                    L2_int += params_xlo_local.beta * T2;
                    L3_int += params_xlo_local.beta * T3;
                    L4_int += params_xlo_local.beta * T4;
                    L5_int += params_xlo_local.beta * T5;

                    // Apply boundary conditions
                    Set::Scalar L1, L2, L3, L4, L5;
                    Set::Scalar M_local = std::abs(u_bnd) / c_bnd; // Local Mach number

                    if (params_xlo_local.type == Type::Inflow)
                    {
                        // SUBSONIC INFLOW
                        // Specify: L2, L3, L4, L5 (incoming characteristics)
                        // Compute: L1 (outgoing characteristic from interior)

                        Set::Scalar K_u = params_xlo_local.relax_u * (1.0 - M_local * M_local) * c_bnd / params_xlo_local.L_ref;
                        Set::Scalar K_T = params_xlo_local.relax_T * (1.0 - M_local * M_local) * c_bnd / params_xlo_local.L_ref;

                        L1 = L1_int; // Outgoing wave from interior
                        L2 = K_T * c_bnd * c_bnd * (T_bnd - params_xlo_local.target_T) / T_bnd;
                        L3 = K_u * (v_bnd - params_xlo_local.target_v);
                        L4 = K_u * (w_bnd - params_xlo_local.target_w);
                        L5 = K_u * (u_bnd - params_xlo_local.target_u);
                    }
                    else if (params_xlo_local.type == Type::Outflow)
                    {
                        // SUBSONIC OUTFLOW
                        // Specify: L1 (incoming characteristic)
                        // Compute: L2, L3, L4, L5 (outgoing from interior)

                        Set::Scalar K_p = params_xlo_local.sigma * (1.0 - M_local * M_local) * c_bnd / params_xlo_local.L_ref;

                        L1 = K_p * (p_bnd - params_xlo_local.target_p);
                        L2 = L2_int;
                        L3 = L3_int;
                        L4 = L4_int;
                        L5 = L5_int;
                    }
                    else if (params_xlo_local.type == Type::SlipWall)
                    {
                        // SLIP WALL
                        L1 = L5_int; // Reflect outgoing acoustic
                        L2 = L2_int;
                        L3 = L3_int;
                        L4 = L4_int;
                        L5 = L5_int;
                    }
                    else
                    { // NoSlipWall
                        // NO-SLIP WALL
                        Set::Scalar K_wall = 0.5 * c_bnd / params_xlo_local.L_ref;

                        L1 = L5_int;
                        L2 = L2_int;
                        L3 = -K_wall * v_bnd;
                        L4 = -K_wall * w_bnd;
                        L5 = L5_int;
                    }

                    // Convert waves back to primitive variable gradients
                    // Equation (26): dQ/dx = S(x)A^-1L
                    Set::Scalar drho_bc = (L5 + L1) / (c_bnd * c_bnd) + L2 / (c_bnd * c_bnd);
                    Set::Scalar du_bc = (L5 - L1) / (2.0 * rho_bnd * c_bnd);
                    Set::Scalar dv_bc = L3;
                    Set::Scalar dw_bc = L4;
                    Set::Scalar dp_bc = (L5 + L1) / 2.0;

                    // Extrapolate to ghost cells using finite differences
                    // Equations (27)-(30) from Motheau et al. (2018)

                    // First ghost cell: Q_{i-1} = Q_i - dx*dQ/dx  ,[Eq. 27]
                    rho_arr(i_bnd - 1, j, k) = rho_bnd - delta * drho_bc;
                    u_arr(i_bnd - 1, j, k) = u_bnd - delta * du_bc;
                    v_arr(i_bnd - 1, j, k) = v_bnd - delta * dv_bc;
                    w_arr(i_bnd - 1, j, k) = w_bnd - delta * dw_bc;
                    p_arr(i_bnd - 1, j, k) = p_bnd - delta * dp_bc;

                    // Second ghost cell: Q_{i-2} = -2Q_{i+1} - 3Q_i + 6Q_{i-1} + 6dx*dQ/dx  ,[Eq. 28]
                    rho_arr(i_bnd - 2, j, k) = -2.0 * rho_arr(i_bnd + 1, j, k) - 3.0 * rho_bnd + 6.0 * rho_arr(i_bnd - 1, j, k) + 6.0 * delta * drho_bc;
                    u_arr(i_bnd - 2, j, k) = -2.0 * u_arr(i_bnd + 1, j, k) - 3.0 * u_bnd + 6.0 * u_arr(i_bnd - 1, j, k) + 6.0 * delta * du_bc;
                    v_arr(i_bnd - 2, j, k) = -2.0 * v_arr(i_bnd + 1, j, k) - 3.0 * v_bnd + 6.0 * v_arr(i_bnd - 1, j, k) + 6.0 * delta * dv_bc;
                    w_arr(i_bnd - 2, j, k) = -2.0 * w_arr(i_bnd + 1, j, k) - 3.0 * w_bnd + 6.0 * w_arr(i_bnd - 1, j, k) + 6.0 * delta * dw_bc;
                    p_arr(i_bnd - 2, j, k) = -2.0 * p_arr(i_bnd + 1, j, k) - 3.0 * p_bnd + 6.0 * p_arr(i_bnd - 1, j, k) + 6.0 * delta * dp_bc;

                    // Third ghost cell: Q_{i-3} = 3Q_{i+1} + 10Q_i - 18Q_{i-1} + 6Q_{i-2} - 12dx*dQ/dx  ,[Eq. 29]
                    rho_arr(i_bnd - 3, j, k) = 3.0 * rho_arr(i_bnd + 1, j, k) + 10.0 * rho_bnd - 18.0 * rho_arr(i_bnd - 1, j, k) + 6.0 * rho_arr(i_bnd - 2, j, k) - 12.0 * delta * drho_bc;
                    u_arr(i_bnd - 3, j, k) = 3.0 * u_arr(i_bnd + 1, j, k) + 10.0 * u_bnd - 18.0 * u_arr(i_bnd - 1, j, k) + 6.0 * u_arr(i_bnd - 2, j, k) - 12.0 * delta * du_bc;
                    v_arr(i_bnd - 3, j, k) = 3.0 * v_arr(i_bnd + 1, j, k) + 10.0 * v_bnd - 18.0 * v_arr(i_bnd - 1, j, k) + 6.0 * v_arr(i_bnd - 2, j, k) - 12.0 * delta * dv_bc;
                    w_arr(i_bnd - 3, j, k) = 3.0 * w_arr(i_bnd + 1, j, k) + 10.0 * w_bnd - 18.0 * w_arr(i_bnd - 1, j, k) + 6.0 * w_arr(i_bnd - 2, j, k) - 12.0 * delta * dw_bc;
                    p_arr(i_bnd - 3, j, k) = 3.0 * p_arr(i_bnd + 1, j, k) + 10.0 * p_bnd - 18.0 * p_arr(i_bnd - 1, j, k) + 6.0 * p_arr(i_bnd - 2, j, k) - 12.0 * delta * dp_bc;

                    // Fourth ghost cell: Q_{i-4} = -2Q_{i+1} - 13Q_i + 24Q_{i-1} - 12Q_{i-2} + 4Q_{i-3} + 12dx*dQ/dx  ,[Eq. 30]
                    rho_arr(i_bnd - 4, j, k) = -2.0 * rho_arr(i_bnd + 1, j, k) - 13.0 * rho_bnd + 24.0 * rho_arr(i_bnd - 1, j, k) - 12.0 * rho_arr(i_bnd - 2, j, k) + 4.0 * rho_arr(i_bnd - 3, j, k) + 12.0 * delta * drho_bc;
                    u_arr(i_bnd - 4, j, k) = -2.0 * u_arr(i_bnd + 1, j, k) - 13.0 * u_bnd + 24.0 * u_arr(i_bnd - 1, j, k) - 12.0 * u_arr(i_bnd - 2, j, k) + 4.0 * u_arr(i_bnd - 3, j, k) + 12.0 * delta * du_bc;
                    v_arr(i_bnd - 4, j, k) = -2.0 * v_arr(i_bnd + 1, j, k) - 13.0 * v_bnd + 24.0 * v_arr(i_bnd - 1, j, k) - 12.0 * v_arr(i_bnd - 2, j, k) + 4.0 * v_arr(i_bnd - 3, j, k) + 12.0 * delta * dv_bc;
                    w_arr(i_bnd - 4, j, k) = -2.0 * w_arr(i_bnd + 1, j, k) - 13.0 * w_bnd + 24.0 * w_arr(i_bnd - 1, j, k) - 12.0 * w_arr(i_bnd - 2, j, k) + 4.0 * w_arr(i_bnd - 3, j, k) + 12.0 * delta * dw_bc;
                    p_arr(i_bnd - 4, j, k) = -2.0 * p_arr(i_bnd + 1, j, k) - 13.0 * p_bnd + 24.0 * p_arr(i_bnd - 1, j, k) - 12.0 * p_arr(i_bnd - 2, j, k) + 4.0 * p_arr(i_bnd - 3, j, k) + 12.0 * delta * dp_bc;

                    // Convert primitive back to conservative in ghost cells
                    for (int ig = i_bnd - 1; ig >= i_bnd - 4; ig--)
                    {
                        Set::Scalar rho_g = std::max(small_local, rho_arr(ig, j, k));
                        Set::Scalar u_g = u_arr(ig, j, k);
                        Set::Scalar v_g = v_arr(ig, j, k);
                        Set::Scalar w_g = w_arr(ig, j, k);
                        Set::Scalar p_g = std::max(small_local, p_arr(ig, j, k));

                        // Conservative variables
                        rho_arr(ig, j, k) = rho_g;
                        momx_arr(ig, j, k) = rho_g * u_g;
                        momy_arr(ig, j, k) = rho_g * v_g;
#if AMREX_SPACEDIM == 3
                        momz_arr(ig, j, k) = rho_g * w_g;
#endif

                        // Total energy: E = rhoe + 1/2rho(u^2 + v^2 + w^2)
                        Set::Scalar ke_g = 0.5 * (u_g * u_g + v_g * v_g + w_g * w_g);
                        Set::Scalar ie_g = (p_g + pref_local + gamma_bnd * p0_bnd) / ((gamma_bnd - 1.0) * rho_g);
                        energy_arr(ig, j, k) = rho_g * (ie_g + ke_g);
                    }
                });
            }

            // ====================================================================
            // X-HI BOUNDARY (i = domain.bigEnd(0))
            // ====================================================================
            if (bx.bigEnd(0) == domain.bigEnd(0))
            {
                int idir = 0;
                int isign = 1;
                Set::Scalar delta = dx[0];

                amrex::Box ghost_box = amrex::adjCellHi(bx, 0, 1);

                amrex::ParallelFor(ghost_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    int i_bnd = domain.bigEnd(0);

                    Set::Scalar rho_bnd = rho_arr(i_bnd, j, k);
                    Set::Scalar u_bnd = u_arr(i_bnd, j, k);
                    Set::Scalar v_bnd = v_arr(i_bnd, j, k);
                    Set::Scalar w_bnd = w_arr(i_bnd, j, k);
                    Set::Scalar p_bnd = p_arr(i_bnd, j, k);
                    Set::Scalar T_bnd = T_arr(i_bnd, j, k);
                    Set::Scalar c_bnd = c_arr(i_bnd, j, k);
                    Set::Scalar gamma_bnd = gamma_arr(i_bnd, j, k);
                    Set::Scalar p0_bnd = p0_arr(i_bnd, j, k);

                    // Normal derivatives (backward difference for right boundary)
                    // dQ/dx ~= (3Q_i - 4Q_{i-1} + Q_{i-2}) / (2dx)
                    Set::Scalar drho_dx = (3.0 * rho_arr(i_bnd, j, k) - 4.0 * rho_arr(i_bnd - 1, j, k) + rho_arr(i_bnd - 2, j, k)) / (2.0 * delta);
                    Set::Scalar du_dx = (3.0 * u_arr(i_bnd, j, k) - 4.0 * u_arr(i_bnd - 1, j, k) + u_arr(i_bnd - 2, j, k)) / (2.0 * delta);
                    Set::Scalar dv_dx = (3.0 * v_arr(i_bnd, j, k) - 4.0 * v_arr(i_bnd - 1, j, k) + v_arr(i_bnd - 2, j, k)) / (2.0 * delta);
                    Set::Scalar dw_dx = (3.0 * w_arr(i_bnd, j, k) - 4.0 * w_arr(i_bnd - 1, j, k) + w_arr(i_bnd - 2, j, k)) / (2.0 * delta);
                    Set::Scalar dp_dx = (3.0 * p_arr(i_bnd, j, k) - 4.0 * p_arr(i_bnd - 1, j, k) + p_arr(i_bnd - 2, j, k)) / (2.0 * delta);

                    // Tangential derivatives (same as x-lo)
                    Set::Scalar drho_dy = 0.0, du_dy = 0.0, dv_dy = 0.0, dw_dy = 0.0, dp_dy = 0.0;
                    if (j > domain.smallEnd(1) && j < domain.bigEnd(1))
                    {
                        drho_dy = (rho_arr(i_bnd, j + 1, k) - rho_arr(i_bnd, j - 1, k)) / (2.0 * dx[1]);
                        du_dy = (u_arr(i_bnd, j + 1, k) - u_arr(i_bnd, j - 1, k)) / (2.0 * dx[1]);
                        dv_dy = (v_arr(i_bnd, j + 1, k) - v_arr(i_bnd, j - 1, k)) / (2.0 * dx[1]);
                        dw_dy = (w_arr(i_bnd, j + 1, k) - w_arr(i_bnd, j - 1, k)) / (2.0 * dx[1]);
                        dp_dy = (p_arr(i_bnd, j + 1, k) - p_arr(i_bnd, j - 1, k)) / (2.0 * dx[1]);
                    }

#if AMREX_SPACEDIM == 3
                    Set::Scalar drho_dz = 0.0, du_dz = 0.0, dv_dz = 0.0, dw_dz = 0.0, dp_dz = 0.0;
                    if (k > domain.smallEnd(2) && k < domain.bigEnd(2))
                    {
                        drho_dz = (rho_arr(i_bnd, j, k + 1) - rho_arr(i_bnd, j, k - 1)) / (2.0 * dx, [2]);
                        du_dz = (u_arr(i_bnd, j, k + 1) - u_arr(i_bnd, j, k - 1)) / (2.0 * dx, [2]);
                        dv_dz = (v_arr(i_bnd, j, k + 1) - v_arr(i_bnd, j, k - 1)) / (2.0 * dx, [2]);
                        dw_dz = (w_arr(i_bnd, j, k + 1) - w_arr(i_bnd, j, k - 1)) / (2.0 * dx, [2]);
                        dp_dz = (p_arr(i_bnd, j, k + 1) - p_arr(i_bnd, j, k - 1)) / (2.0 * dx, [2]);
                    }
#else
                Set::Scalar drho_dz = 0.0, du_dz = 0.0, dv_dz = 0.0, dw_dz = 0.0, dp_dz = 0.0;
#endif

                    // Transverse terms
                    Set::Scalar T1 = v_bnd * dp_dy + w_bnd * dp_dz - rho_bnd * c_bnd * c_bnd * (v_bnd * drho_dy + w_bnd * drho_dz);
                    Set::Scalar T2 = v_bnd * du_dy + w_bnd * du_dz;
                    Set::Scalar T3 = v_bnd * dv_dy + w_bnd * dv_dz;
                    Set::Scalar T4 = v_bnd * dw_dy + w_bnd * dw_dz;
                    Set::Scalar T5 = v_bnd * dp_dy + w_bnd * dp_dz;

                    // Characteristic waves
                    Set::Scalar lambda1 = u_bnd - c_bnd;
                    Set::Scalar lambda2 = u_bnd;
                    Set::Scalar lambda3 = u_bnd;
                    Set::Scalar lambda4 = u_bnd;
                    Set::Scalar lambda5 = u_bnd + c_bnd;

                    Set::Scalar L1_int = lambda1 * (dp_dx - rho_bnd * c_bnd * du_dx);
                    Set::Scalar L2_int = lambda2 * (c_bnd * c_bnd * drho_dx - dp_dx);
                    Set::Scalar L3_int = lambda3 * dv_dx;
                    Set::Scalar L4_int = lambda4 * dw_dx;
                    Set::Scalar L5_int = lambda5 * (dp_dx + rho_bnd * c_bnd * du_dx);

                    L1_int += params_xhi_local.beta * T1;
                    L2_int += params_xhi_local.beta * T2;
                    L3_int += params_xhi_local.beta * T3;
                    L4_int += params_xhi_local.beta * T4;
                    L5_int += params_xhi_local.beta * T5;

                    // Apply boundary conditions
                    Set::Scalar L1, L2, L3, L4, L5;
                    Set::Scalar M_local = std::abs(u_bnd) / c_bnd;

                    if (params_xhi_local.type == Type::Inflow)
                    {
                        Set::Scalar K_u = params_xhi_local.relax_u * (1.0 - M_local * M_local) * c_bnd / params_xhi_local.L_ref;
                        Set::Scalar K_T = params_xhi_local.relax_T * (1.0 - M_local * M_local) * c_bnd / params_xhi_local.L_ref;

                        L1 = K_u * (u_bnd - params_xhi_local.target_u);
                        L2 = K_T * c_bnd * c_bnd * (T_bnd - params_xhi_local.target_T) / T_bnd;
                        L3 = K_u * (v_bnd - params_xhi_local.target_v);
                        L4 = K_u * (w_bnd - params_xhi_local.target_w);
                        L5 = L5_int;
                    }
                    else if (params_xhi_local.type == Type::Outflow)
                    {
                        Set::Scalar K_p = params_xhi_local.sigma * (1.0 - M_local * M_local) * c_bnd / params_xhi_local.L_ref;

                        L1 = L1_int;
                        L2 = L2_int;
                        L3 = L3_int;
                        L4 = L4_int;
                        L5 = K_p * (p_bnd - params_xhi_local.target_p);
                    }
                    else if (params_xhi_local.type == Type::SlipWall)
                    {
                        L1 = L1_int;
                        L2 = L2_int;
                        L3 = L3_int;
                        L4 = L4_int;
                        L5 = L1_int;
                    }
                    else
                    { // NoSlipWall
                        Set::Scalar K_wall = 0.5 * c_bnd / params_xhi_local.L_ref;

                        L1 = L1_int;
                        L2 = L2_int;
                        L3 = -K_wall * v_bnd;
                        L4 = -K_wall * w_bnd;
                        L5 = L1_int;
                    }

                    // Convert waves to gradients ,[Eq. 26]
                    Set::Scalar drho_bc = (L5 + L1) / (c_bnd * c_bnd) + L2 / (c_bnd * c_bnd);
                    Set::Scalar du_bc = (L5 - L1) / (2.0 * rho_bnd * c_bnd);
                    Set::Scalar dv_bc = L3;
                    Set::Scalar dw_bc = L4;
                    Set::Scalar dp_bc = (L5 + L1) / 2.0;

                    // Extrapolate to ghost cells ,[Eqs. 31-34]
                    // First ghost cell: Q_{i+1} = Q_i + dx*dQ/dx  ,[Eq. 31]
                    rho_arr(i_bnd + 1, j, k) = rho_bnd + delta * drho_bc;
                    u_arr(i_bnd + 1, j, k) = u_bnd + delta * du_bc;
                    v_arr(i_bnd + 1, j, k) = v_bnd + delta * dv_bc;
                    w_arr(i_bnd + 1, j, k) = w_bnd + delta * dw_bc;
                    p_arr(i_bnd + 1, j, k) = p_bnd + delta * dp_bc;

                    // Second ghost cell: Q_{i+2} = -2Q_{i-1} - 3Q_i + 6Q_{i+1} - 6dx*dQ/dx  ,[Eq. 32]
                    rho_arr(i_bnd + 2, j, k) = -2.0 * rho_arr(i_bnd - 1, j, k) - 3.0 * rho_bnd + 6.0 * rho_arr(i_bnd + 1, j, k) - 6.0 * delta * drho_bc;
                    u_arr(i_bnd + 2, j, k) = -2.0 * u_arr(i_bnd - 1, j, k) - 3.0 * u_bnd + 6.0 * u_arr(i_bnd + 1, j, k) - 6.0 * delta * du_bc;
                    v_arr(i_bnd + 2, j, k) = -2.0 * v_arr(i_bnd - 1, j, k) - 3.0 * v_bnd + 6.0 * v_arr(i_bnd + 1, j, k) - 6.0 * delta * dv_bc;
                    w_arr(i_bnd + 2, j, k) = -2.0 * w_arr(i_bnd - 1, j, k) - 3.0 * w_bnd + 6.0 * w_arr(i_bnd + 1, j, k) - 6.0 * delta * dw_bc;
                    p_arr(i_bnd + 2, j, k) = -2.0 * p_arr(i_bnd - 1, j, k) - 3.0 * p_bnd + 6.0 * p_arr(i_bnd + 1, j, k) - 6.0 * delta * dp_bc;

                    // Third ghost cell: Q_{i+3} = 3Q_{i-1} + 10Q_i - 18Q_{i+1} + 6Q_{i+2} + 12dx*dQ/dx  ,[Eq. 33]
                    rho_arr(i_bnd + 3, j, k) = 3.0 * rho_arr(i_bnd - 1, j, k) + 10.0 * rho_bnd - 18.0 * rho_arr(i_bnd + 1, j, k) + 6.0 * rho_arr(i_bnd + 2, j, k) + 12.0 * delta * drho_bc;
                    u_arr(i_bnd + 3, j, k) = 3.0 * u_arr(i_bnd - 1, j, k) + 10.0 * u_bnd - 18.0 * u_arr(i_bnd + 1, j, k) + 6.0 * u_arr(i_bnd + 2, j, k) + 12.0 * delta * du_bc;
                    v_arr(i_bnd + 3, j, k) = 3.0 * v_arr(i_bnd - 1, j, k) + 10.0 * v_bnd - 18.0 * v_arr(i_bnd + 1, j, k) + 6.0 * v_arr(i_bnd + 2, j, k) + 12.0 * delta * dv_bc;
                    w_arr(i_bnd + 3, j, k) = 3.0 * w_arr(i_bnd - 1, j, k) + 10.0 * w_bnd - 18.0 * w_arr(i_bnd + 1, j, k) + 6.0 * w_arr(i_bnd + 2, j, k) + 12.0 * delta * dw_bc;
                    p_arr(i_bnd + 3, j, k) = 3.0 * p_arr(i_bnd - 1, j, k) + 10.0 * p_bnd - 18.0 * p_arr(i_bnd + 1, j, k) + 6.0 * p_arr(i_bnd + 2, j, k) + 12.0 * delta * dp_bc;

                    // Fourth ghost cell: Q_{i+4} = -2Q_{i-1} - 13Q_i + 24Q_{i+1} - 12Q_{i+2} + 4Q_{i+3} - 12dx*dQ/dx  ,[Eq. 34]
                    rho_arr(i_bnd + 4, j, k) = -2.0 * rho_arr(i_bnd - 1, j, k) - 13.0 * rho_bnd + 24.0 * rho_arr(i_bnd + 1, j, k) - 12.0 * rho_arr(i_bnd + 2, j, k) + 4.0 * rho_arr(i_bnd + 3, j, k) - 12.0 * delta * drho_bc;
                    u_arr(i_bnd + 4, j, k) = -2.0 * u_arr(i_bnd - 1, j, k) - 13.0 * u_bnd + 24.0 * u_arr(i_bnd + 1, j, k) - 12.0 * u_arr(i_bnd + 2, j, k) + 4.0 * u_arr(i_bnd + 3, j, k) - 12.0 * delta * du_bc;
                    v_arr(i_bnd + 4, j, k) = -2.0 * v_arr(i_bnd - 1, j, k) - 13.0 * v_bnd + 24.0 * v_arr(i_bnd + 1, j, k) - 12.0 * v_arr(i_bnd + 2, j, k) + 4.0 * v_arr(i_bnd + 3, j, k) - 12.0 * delta * dv_bc;
                    w_arr(i_bnd + 4, j, k) = -2.0 * w_arr(i_bnd - 1, j, k) - 13.0 * w_bnd + 24.0 * w_arr(i_bnd + 1, j, k) - 12.0 * w_arr(i_bnd + 2, j, k) + 4.0 * w_arr(i_bnd + 3, j, k) - 12.0 * delta * dw_bc;
                    p_arr(i_bnd + 4, j, k) = -2.0 * p_arr(i_bnd - 1, j, k) - 13.0 * p_bnd + 24.0 * p_arr(i_bnd + 1, j, k) - 12.0 * p_arr(i_bnd + 2, j, k) + 4.0 * p_arr(i_bnd + 3, j, k) - 12.0 * delta * dp_bc;

                    // Convert primitive to conservative
                    for (int ig = i_bnd + 1; ig <= i_bnd + 4; ig++)
                    {
                        Set::Scalar rho_g = std::max(small_local, rho_arr(ig, j, k));
                        Set::Scalar u_g = u_arr(ig, j, k);
                        Set::Scalar v_g = v_arr(ig, j, k);
                        Set::Scalar w_g = w_arr(ig, j, k);
                        Set::Scalar p_g = std::max(small_local, p_arr(ig, j, k));

                        rho_arr(ig, j, k) = rho_g;
                        momx_arr(ig, j, k) = rho_g * u_g;
                        momy_arr(ig, j, k) = rho_g * v_g;
#if AMREX_SPACEDIM == 3
                        momz_arr(ig, j, k) = rho_g * w_g;
#endif

                        Set::Scalar ke_g = 0.5 * (u_g * u_g + v_g * v_g + w_g * w_g);
                        Set::Scalar ie_g = (p_g + pref_local + gamma_bnd * p0_bnd) / ((gamma_bnd - 1.0) * rho_g);
                        energy_arr(ig, j, k) = rho_g * (ie_g + ke_g);
                    }
                });
            }

            // ====================================================================
            // Y-LO BOUNDARY (j = domain.smallEnd(1))
            // ====================================================================
            if (bx.smallEnd(1) == domain.smallEnd(1))
            {
                int idir = 1;   // y-direction
                int isign = -1; // bottom boundary
                Set::Scalar delta = dx[1];

                amrex::Box ghost_box = amrex::adjCellLo(bx, 1, 1);

                amrex::ParallelFor(ghost_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    // Boundary cell index (last interior cell)
                    int j_bnd = domain.smallEnd(1);

                    // Get boundary state
                    Set::Scalar rho_bnd = rho_arr(i, j_bnd, k);
                    Set::Scalar u_bnd = u_arr(i, j_bnd, k);
                    Set::Scalar v_bnd = v_arr(i, j_bnd, k);
                    Set::Scalar w_bnd = w_arr(i, j_bnd, k);
                    Set::Scalar p_bnd = p_arr(i, j_bnd, k);
                    Set::Scalar T_bnd = T_arr(i, j_bnd, k);
                    Set::Scalar c_bnd = c_arr(i, j_bnd, k);
                    Set::Scalar gamma_bnd = gamma_arr(i, j_bnd, k);
                    Set::Scalar p0_bnd = p0_arr(i, j_bnd, k);

                    // Compute normal derivatives (one-sided, 2nd order)
                    // dQ/dy ~= (-3Q_j + 4Q_{j+1} - Q_{j+2}) / (2dy)
                    Set::Scalar drho_dy = (-3.0 * rho_arr(i, j_bnd, k) + 4.0 * rho_arr(i, j_bnd + 1, k) - rho_arr(i, j_bnd + 2, k)) / (2.0 * delta);
                    Set::Scalar du_dy = (-3.0 * u_arr(i, j_bnd, k) + 4.0 * u_arr(i, j_bnd + 1, k) - u_arr(i, j_bnd + 2, k)) / (2.0 * delta);
                    Set::Scalar dv_dy = (-3.0 * v_arr(i, j_bnd, k) + 4.0 * v_arr(i, j_bnd + 1, k) - v_arr(i, j_bnd + 2, k)) / (2.0 * delta);
                    Set::Scalar dw_dy = (-3.0 * w_arr(i, j_bnd, k) + 4.0 * w_arr(i, j_bnd + 1, k) - w_arr(i, j_bnd + 2, k)) / (2.0 * delta);
                    Set::Scalar dp_dy = (-3.0 * p_arr(i, j_bnd, k) + 4.0 * p_arr(i, j_bnd + 1, k) - p_arr(i, j_bnd + 2, k)) / (2.0 * delta);

                    // Compute tangential derivatives (centered, 2nd order)
                    // X-direction: dQ/dx ~= (Q_{i+1} - Q_{i-1}) / (2dx)
                    Set::Scalar drho_dx = 0.0, du_dx = 0.0, dv_dx = 0.0, dw_dx = 0.0, dp_dx = 0.0;
                    if (i > domain.smallEnd(0) && i < domain.bigEnd(0))
                    {
                        drho_dx = (rho_arr(i + 1, j_bnd, k) - rho_arr(i - 1, j_bnd, k)) / (2.0 * dx[0]);
                        du_dx = (u_arr(i + 1, j_bnd, k) - u_arr(i - 1, j_bnd, k)) / (2.0 * dx[0]);
                        dv_dx = (v_arr(i + 1, j_bnd, k) - v_arr(i - 1, j_bnd, k)) / (2.0 * dx[0]);
                        dw_dx = (w_arr(i + 1, j_bnd, k) - w_arr(i - 1, j_bnd, k)) / (2.0 * dx[0]);
                        dp_dx = (p_arr(i + 1, j_bnd, k) - p_arr(i - 1, j_bnd, k)) / (2.0 * dx[0]);
                    }

#if AMREX_SPACEDIM == 3
                    // Z-direction: dQ/dz ~= (Q_{k+1} - Q_{k-1}) / (2dz)
                    Set::Scalar drho_dz = 0.0, du_dz = 0.0, dv_dz = 0.0, dw_dz = 0.0, dp_dz = 0.0;
                    if (k > domain.smallEnd(2) && k < domain.bigEnd(2))
                    {
                        drho_dz = (rho_arr(i, j_bnd, k + 1) - rho_arr(i, j_bnd, k - 1)) / (2.0 * dx[2]);
                        du_dz = (u_arr(i, j_bnd, k + 1) - u_arr(i, j_bnd, k - 1)) / (2.0 * dx[2]);
                        dv_dz = (v_arr(i, j_bnd, k + 1) - v_arr(i, j_bnd, k - 1)) / (2.0 * dx[2]);
                        dw_dz = (w_arr(i, j_bnd, k + 1) - w_arr(i, j_bnd, k - 1)) / (2.0 * dx[2]);
                        dp_dz = (p_arr(i, j_bnd, k + 1) - p_arr(i, j_bnd, k - 1)) / (2.0 * dx[2]);
                    }
#else
                Set::Scalar drho_dz = 0.0, du_dz = 0.0, dv_dz = 0.0, dw_dz = 0.0, dp_dz = 0.0;
#endif

                    // Compute transverse terms (Section III.B, Motheau et al. 2018)
                    // For y-normal boundary: transverse = x and z directions
                    // T_1 = u*dp/dx + w*dp/dz - rhoc^2(u*drho/dx + w*drho/dz)
                    Set::Scalar T1 = u_bnd * dp_dx + w_bnd * dp_dz - rho_bnd * c_bnd * c_bnd * (u_bnd * drho_dx + w_bnd * drho_dz);
                    Set::Scalar T2 = u_bnd * dv_dx + w_bnd * dv_dz;
                    Set::Scalar T3 = u_bnd * du_dx + w_bnd * du_dz;
                    Set::Scalar T4 = u_bnd * dw_dx + w_bnd * dw_dz;
                    Set::Scalar T5 = u_bnd * dp_dx + w_bnd * dp_dz;

                    // Compute characteristic waves (LODI system)
                    // Wave speeds: lambda_1 = v - c, lambda_2 = v, lambda_3 = v, lambda_4 = v, lambda_5 = v + c
                    Set::Scalar lambda1 = v_bnd - c_bnd;
                    Set::Scalar lambda2 = v_bnd;
                    Set::Scalar lambda3 = v_bnd;
                    Set::Scalar lambda4 = v_bnd;
                    Set::Scalar lambda5 = v_bnd + c_bnd;

                    // Interior wave amplitudes (from spatial derivatives)
                    // Equations from Motheau et al. (2018), Section III
                    Set::Scalar L1_int = lambda1 * (dp_dy - rho_bnd * c_bnd * dv_dy);
                    Set::Scalar L2_int = lambda2 * du_dy;
                    Set::Scalar L3_int = lambda3 * (c_bnd * c_bnd * drho_dy - dp_dy);
                    Set::Scalar L4_int = lambda4 * dw_dy;
                    Set::Scalar L5_int = lambda5 * (dp_dy + rho_bnd * c_bnd * dv_dy);

                    // Add transverse terms (weighted by beta)
                    L1_int += params_ylo_local.beta * T1;
                    L2_int += params_ylo_local.beta * T2;
                    L3_int += params_ylo_local.beta * T3;
                    L4_int += params_ylo_local.beta * T4;
                    L5_int += params_ylo_local.beta * T5;

                    // Apply boundary conditions
                    Set::Scalar L1, L2, L3, L4, L5;
                    Set::Scalar M_local = std::abs(v_bnd) / c_bnd; // Local Mach number

                    if (params_ylo_local.type == Type::Inflow)
                    {
                        // SUBSONIC INFLOW
                        Set::Scalar K_u = params_ylo_local.relax_u * (1.0 - M_local * M_local) * c_bnd / params_ylo_local.L_ref;
                        Set::Scalar K_T = params_ylo_local.relax_T * (1.0 - M_local * M_local) * c_bnd / params_ylo_local.L_ref;

                        L1 = L1_int; // Outgoing wave from interior
                        L2 = K_u * (u_bnd - params_ylo_local.target_u);
                        L3 = K_T * c_bnd * c_bnd * (T_bnd - params_ylo_local.target_T) / T_bnd;
                        L4 = K_u * (w_bnd - params_ylo_local.target_w);
                        L5 = K_u * (v_bnd - params_ylo_local.target_v);
                    }
                    else if (params_ylo_local.type == Type::Outflow)
                    {
                        // SUBSONIC OUTFLOW
                        Set::Scalar K_p = params_ylo_local.sigma * (1.0 - M_local * M_local) * c_bnd / params_ylo_local.L_ref;

                        L1 = K_p * (p_bnd - params_ylo_local.target_p);
                        L2 = L2_int;
                        L3 = L3_int;
                        L4 = L4_int;
                        L5 = L5_int;
                    }
                    else if (params_ylo_local.type == Type::SlipWall)
                    {
                        // SLIP WALL
                        L1 = L5_int; // Reflect outgoing acoustic
                        L2 = L2_int;
                        L3 = L3_int;
                        L4 = L4_int;
                        L5 = L5_int;
                    }
                    else
                    { // NoSlipWall
                        // NO-SLIP WALL
                        Set::Scalar K_wall = 0.5 * c_bnd / params_ylo_local.L_ref;

                        L1 = L5_int;
                        L2 = -K_wall * u_bnd;
                        L3 = L3_int;
                        L4 = -K_wall * w_bnd;
                        L5 = L5_int;
                    }

                    // Convert waves back to primitive variable gradients
                    // Equation (26): dQ/dy = S(y)Lambda^-1L
                    Set::Scalar drho_bc = (L5 + L1) / (c_bnd * c_bnd) + L3 / (c_bnd * c_bnd);
                    Set::Scalar du_bc = L2;
                    Set::Scalar dv_bc = (L5 - L1) / (2.0 * rho_bnd * c_bnd);
                    Set::Scalar dw_bc = L4;
                    Set::Scalar dp_bc = (L5 + L1) / 2.0;

                    // Extrapolate to ghost cells using finite differences
                    // Equations (27)-(30) from Motheau et al. (2018)

                    // First ghost cell: Q_{j-1} = Q_j - dy*dQ/dy  [Eq. 27]
                    rho_arr(i, j_bnd - 1, k) = rho_bnd - delta * drho_bc;
                    u_arr(i, j_bnd - 1, k) = u_bnd - delta * du_bc;
                    v_arr(i, j_bnd - 1, k) = v_bnd - delta * dv_bc;
                    w_arr(i, j_bnd - 1, k) = w_bnd - delta * dw_bc;
                    p_arr(i, j_bnd - 1, k) = p_bnd - delta * dp_bc;

                    // Second ghost cell: Q_{j-2} = -2Q_{j+1} - 3Q_j + 6Q_{j-1} + 6dy*dQ/dy  [Eq. 28]
                    rho_arr(i, j_bnd - 2, k) = -2.0 * rho_arr(i, j_bnd + 1, k) - 3.0 * rho_bnd + 6.0 * rho_arr(i, j_bnd - 1, k) + 6.0 * delta * drho_bc;
                    u_arr(i, j_bnd - 2, k) = -2.0 * u_arr(i, j_bnd + 1, k) - 3.0 * u_bnd + 6.0 * u_arr(i, j_bnd - 1, k) + 6.0 * delta * du_bc;
                    v_arr(i, j_bnd - 2, k) = -2.0 * v_arr(i, j_bnd + 1, k) - 3.0 * v_bnd + 6.0 * v_arr(i, j_bnd - 1, k) + 6.0 * delta * dv_bc;
                    w_arr(i, j_bnd - 2, k) = -2.0 * w_arr(i, j_bnd + 1, k) - 3.0 * w_bnd + 6.0 * w_arr(i, j_bnd - 1, k) + 6.0 * delta * dw_bc;
                    p_arr(i, j_bnd - 2, k) = -2.0 * p_arr(i, j_bnd + 1, k) - 3.0 * p_bnd + 6.0 * p_arr(i, j_bnd - 1, k) + 6.0 * delta * dp_bc;

                    // Third ghost cell: Q_{j-3} = 3Q_{j+1} + 10Q_j - 18Q_{j-1} + 6Q_{j-2} - 12dy*dQ/dy  [Eq. 29]
                    rho_arr(i, j_bnd - 3, k) = 3.0 * rho_arr(i, j_bnd + 1, k) + 10.0 * rho_bnd - 18.0 * rho_arr(i, j_bnd - 1, k) + 6.0 * rho_arr(i, j_bnd - 2, k) - 12.0 * delta * drho_bc;
                    u_arr(i, j_bnd - 3, k) = 3.0 * u_arr(i, j_bnd + 1, k) + 10.0 * u_bnd - 18.0 * u_arr(i, j_bnd - 1, k) + 6.0 * u_arr(i, j_bnd - 2, k) - 12.0 * delta * du_bc;
                    v_arr(i, j_bnd - 3, k) = 3.0 * v_arr(i, j_bnd + 1, k) + 10.0 * v_bnd - 18.0 * v_arr(i, j_bnd - 1, k) + 6.0 * v_arr(i, j_bnd - 2, k) - 12.0 * delta * dv_bc;
                    w_arr(i, j_bnd - 3, k) = 3.0 * w_arr(i, j_bnd + 1, k) + 10.0 * w_bnd - 18.0 * w_arr(i, j_bnd - 1, k) + 6.0 * w_arr(i, j_bnd - 2, k) - 12.0 * delta * dw_bc;
                    p_arr(i, j_bnd - 3, k) = 3.0 * p_arr(i, j_bnd + 1, k) + 10.0 * p_bnd - 18.0 * p_arr(i, j_bnd - 1, k) + 6.0 * p_arr(i, j_bnd - 2, k) - 12.0 * delta * dp_bc;

                    // Fourth ghost cell: Q_{j-4} = -2Q_{j+1} - 13Q_j + 24Q_{j-1} - 12Q_{j-2} + 4Q_{j-3} + 12dy*dQ/dy  [Eq. 30]
                    rho_arr(i, j_bnd - 4, k) = -2.0 * rho_arr(i, j_bnd + 1, k) - 13.0 * rho_bnd + 24.0 * rho_arr(i, j_bnd - 1, k) - 12.0 * rho_arr(i, j_bnd - 2, k) + 4.0 * rho_arr(i, j_bnd - 3, k) + 12.0 * delta * drho_bc;
                    u_arr(i, j_bnd - 4, k) = -2.0 * u_arr(i, j_bnd + 1, k) - 13.0 * u_bnd + 24.0 * u_arr(i, j_bnd - 1, k) - 12.0 * u_arr(i, j_bnd - 2, k) + 4.0 * u_arr(i, j_bnd - 3, k) + 12.0 * delta * du_bc;
                    v_arr(i, j_bnd - 4, k) = -2.0 * v_arr(i, j_bnd + 1, k) - 13.0 * v_bnd + 24.0 * v_arr(i, j_bnd - 1, k) - 12.0 * v_arr(i, j_bnd - 2, k) + 4.0 * v_arr(i, j_bnd - 3, k) + 12.0 * delta * dv_bc;
                    w_arr(i, j_bnd - 4, k) = -2.0 * w_arr(i, j_bnd + 1, k) - 13.0 * w_bnd + 24.0 * w_arr(i, j_bnd - 1, k) - 12.0 * w_arr(i, j_bnd - 2, k) + 4.0 * w_arr(i, j_bnd - 3, k) + 12.0 * delta * dw_bc;
                    p_arr(i, j_bnd - 4, k) = -2.0 * p_arr(i, j_bnd + 1, k) - 13.0 * p_bnd + 24.0 * p_arr(i, j_bnd - 1, k) - 12.0 * p_arr(i, j_bnd - 2, k) + 4.0 * p_arr(i, j_bnd - 3, k) + 12.0 * delta * dp_bc;

                    // Convert primitive back to conservative in ghost cells
                    for (int jg = j_bnd - 1; jg >= j_bnd - 4; jg--)
                    {
                        Set::Scalar rho_g = std::max(small_local, rho_arr(i, jg, k));
                        Set::Scalar u_g = u_arr(i, jg, k);
                        Set::Scalar v_g = v_arr(i, jg, k);
                        Set::Scalar w_g = w_arr(i, jg, k);
                        Set::Scalar p_g = std::max(small_local, p_arr(i, jg, k));

                        // Conservative variables
                        rho_arr(i, jg, k) = rho_g;
                        momx_arr(i, jg, k) = rho_g * u_g;
                        momy_arr(i, jg, k) = rho_g * v_g;
#if AMREX_SPACEDIM == 3
                        momz_arr(i, jg, k) = rho_g * w_g;
#endif

                        // Total energy: E = rhoe + 1/2rho(u^2 + v^2 + w^2)
                        Set::Scalar ke_g = 0.5 * (u_g * u_g + v_g * v_g + w_g * w_g);
                        Set::Scalar ie_g = (p_g + pref_local + gamma_bnd * p0_bnd) / ((gamma_bnd - 1.0) * rho_g);
                        energy_arr(i, jg, k) = rho_g * (ie_g + ke_g);
                    }
                });
            }

            // ====================================================================
            // Y-HI BOUNDARY (j = domain.bigEnd(1))
            // ====================================================================
            if (bx.bigEnd(1) == domain.bigEnd(1))
            {
                int idir = 1;
                int isign = 1;
                Set::Scalar delta = dx[1];

                amrex::Box ghost_box = amrex::adjCellHi(bx, 1, 1);

                amrex::ParallelFor(ghost_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    int j_bnd = domain.bigEnd(1);

                    Set::Scalar rho_bnd = rho_arr(i, j_bnd, k);
                    Set::Scalar u_bnd = u_arr(i, j_bnd, k);
                    Set::Scalar v_bnd = v_arr(i, j_bnd, k);
                    Set::Scalar w_bnd = w_arr(i, j_bnd, k);
                    Set::Scalar p_bnd = p_arr(i, j_bnd, k);
                    Set::Scalar T_bnd = T_arr(i, j_bnd, k);
                    Set::Scalar c_bnd = c_arr(i, j_bnd, k);
                    Set::Scalar gamma_bnd = gamma_arr(i, j_bnd, k);
                    Set::Scalar p0_bnd = p0_arr(i, j_bnd, k);

                    // Normal derivatives (backward difference for top boundary)
                    // dQ/dy ~= (3Q_j - 4Q_{j-1} + Q_{j-2}) / (2dy)
                    Set::Scalar drho_dy = (3.0 * rho_arr(i, j_bnd, k) - 4.0 * rho_arr(i, j_bnd - 1, k) + rho_arr(i, j_bnd - 2, k)) / (2.0 * delta);
                    Set::Scalar du_dy = (3.0 * u_arr(i, j_bnd, k) - 4.0 * u_arr(i, j_bnd - 1, k) + u_arr(i, j_bnd - 2, k)) / (2.0 * delta);
                    Set::Scalar dv_dy = (3.0 * v_arr(i, j_bnd, k) - 4.0 * v_arr(i, j_bnd - 1, k) + v_arr(i, j_bnd - 2, k)) / (2.0 * delta);
                    Set::Scalar dw_dy = (3.0 * w_arr(i, j_bnd, k) - 4.0 * w_arr(i, j_bnd - 1, k) + w_arr(i, j_bnd - 2, k)) / (2.0 * delta);
                    Set::Scalar dp_dy = (3.0 * p_arr(i, j_bnd, k) - 4.0 * p_arr(i, j_bnd - 1, k) + p_arr(i, j_bnd - 2, k)) / (2.0 * delta);

                    // Tangential derivatives (same as y-lo)
                    Set::Scalar drho_dx = 0.0, du_dx = 0.0, dv_dx = 0.0, dw_dx = 0.0, dp_dx = 0.0;
                    if (i > domain.smallEnd(0) && i < domain.bigEnd(0))
                    {
                        drho_dx = (rho_arr(i + 1, j_bnd, k) - rho_arr(i - 1, j_bnd, k)) / (2.0 * dx[0]);
                        du_dx = (u_arr(i + 1, j_bnd, k) - u_arr(i - 1, j_bnd, k)) / (2.0 * dx[0]);
                        dv_dx = (v_arr(i + 1, j_bnd, k) - v_arr(i - 1, j_bnd, k)) / (2.0 * dx[0]);
                        dw_dx = (w_arr(i + 1, j_bnd, k) - w_arr(i - 1, j_bnd, k)) / (2.0 * dx[0]);
                        dp_dx = (p_arr(i + 1, j_bnd, k) - p_arr(i - 1, j_bnd, k)) / (2.0 * dx[0]);
                    }

#if AMREX_SPACEDIM == 3
                    Set::Scalar drho_dz = 0.0, du_dz = 0.0, dv_dz = 0.0, dw_dz = 0.0, dp_dz = 0.0;
                    if (k > domain.smallEnd(2) && k < domain.bigEnd(2))
                    {
                        drho_dz = (rho_arr(i, j_bnd, k + 1) - rho_arr(i, j_bnd, k - 1)) / (2.0 * dx[2]);
                        du_dz = (u_arr(i, j_bnd, k + 1) - u_arr(i, j_bnd, k - 1)) / (2.0 * dx[2]);
                        dv_dz = (v_arr(i, j_bnd, k + 1) - v_arr(i, j_bnd, k - 1)) / (2.0 * dx[2]);
                        dw_dz = (w_arr(i, j_bnd, k + 1) - w_arr(i, j_bnd, k - 1)) / (2.0 * dx[2]);
                        dp_dz = (p_arr(i, j_bnd, k + 1) - p_arr(i, j_bnd, k - 1)) / (2.0 * dx[2]);
                    }
#else
                Set::Scalar drho_dz = 0.0, du_dz = 0.0, dv_dz = 0.0, dw_dz = 0.0, dp_dz = 0.0;
#endif

                    // Transverse terms
                    Set::Scalar T1 = u_bnd * dp_dx + w_bnd * dp_dz - rho_bnd * c_bnd * c_bnd * (u_bnd * drho_dx + w_bnd * drho_dz);
                    Set::Scalar T2 = u_bnd * dv_dx + w_bnd * dv_dz;
                    Set::Scalar T3 = u_bnd * du_dx + w_bnd * du_dz;
                    Set::Scalar T4 = u_bnd * dw_dx + w_bnd * dw_dz;
                    Set::Scalar T5 = u_bnd * dp_dx + w_bnd * dp_dz;

                    // Characteristic waves
                    Set::Scalar lambda1 = v_bnd - c_bnd;
                    Set::Scalar lambda2 = v_bnd;
                    Set::Scalar lambda3 = v_bnd;
                    Set::Scalar lambda4 = v_bnd;
                    Set::Scalar lambda5 = v_bnd + c_bnd;

                    Set::Scalar L1_int = lambda1 * (dp_dy - rho_bnd * c_bnd * dv_dy);
                    Set::Scalar L2_int = lambda2 * du_dy;
                    Set::Scalar L3_int = lambda3 * (c_bnd * c_bnd * drho_dy - dp_dy);
                    Set::Scalar L4_int = lambda4 * dw_dy;
                    Set::Scalar L5_int = lambda5 * (dp_dy + rho_bnd * c_bnd * dv_dy);

                    L1_int += params_yhi_local.beta * T1;
                    L2_int += params_yhi_local.beta * T2;
                    L3_int += params_yhi_local.beta * T3;
                    L4_int += params_yhi_local.beta * T4;
                    L5_int += params_yhi_local.beta * T5;

                    // Apply boundary conditions
                    Set::Scalar L1, L2, L3, L4, L5;
                    Set::Scalar M_local = std::abs(v_bnd) / c_bnd;

                    if (params_yhi_local.type == Type::Inflow)
                    {
                        Set::Scalar K_u = params_yhi_local.relax_u * (1.0 - M_local * M_local) * c_bnd / params_yhi_local.L_ref;
                        Set::Scalar K_T = params_yhi_local.relax_T * (1.0 - M_local * M_local) * c_bnd / params_yhi_local.L_ref;

                        L1 = K_u * (v_bnd - params_yhi_local.target_v);
                        L2 = K_u * (u_bnd - params_yhi_local.target_u);
                        L3 = K_T * c_bnd * c_bnd * (T_bnd - params_yhi_local.target_T) / T_bnd;
                        L4 = K_u * (w_bnd - params_yhi_local.target_w);
                        L5 = L5_int;
                    }
                    else if (params_yhi_local.type == Type::Outflow)
                    {
                        Set::Scalar K_p = params_yhi_local.sigma * (1.0 - M_local * M_local) * c_bnd / params_yhi_local.L_ref;

                        L1 = L1_int;
                        L2 = L2_int;
                        L3 = L3_int;
                        L4 = L4_int;
                        L5 = K_p * (p_bnd - params_yhi_local.target_p);
                    }
                    else if (params_yhi_local.type == Type::SlipWall)
                    {
                        L1 = L1_int;
                        L2 = L2_int;
                        L3 = L3_int;
                        L4 = L4_int;
                        L5 = L1_int;
                    }
                    else
                    { // NoSlipWall
                        Set::Scalar K_wall = 0.5 * c_bnd / params_yhi_local.L_ref;

                        L1 = L1_int;
                        L2 = -K_wall * u_bnd;
                        L3 = L3_int;
                        L4 = -K_wall * w_bnd;
                        L5 = L1_int;
                    }

                    // Convert waves to gradients [Eq. 26]
                    Set::Scalar drho_bc = (L5 + L1) / (c_bnd * c_bnd) + L3 / (c_bnd * c_bnd);
                    Set::Scalar du_bc = L2;
                    Set::Scalar dv_bc = (L5 - L1) / (2.0 * rho_bnd * c_bnd);
                    Set::Scalar dw_bc = L4;
                    Set::Scalar dp_bc = (L5 + L1) / 2.0;

                    // Extrapolate to ghost cells [Eqs. 31-34]
                    // First ghost cell: Q_{j+1} = Q_j + dy*dQ/dy  [Eq. 31]
                    rho_arr(i, j_bnd + 1, k) = rho_bnd + delta * drho_bc;
                    u_arr(i, j_bnd + 1, k) = u_bnd + delta * du_bc;
                    v_arr(i, j_bnd + 1, k) = v_bnd + delta * dv_bc;
                    w_arr(i, j_bnd + 1, k) = w_bnd + delta * dw_bc;
                    p_arr(i, j_bnd + 1, k) = p_bnd + delta * dp_bc;

                    // Second ghost cell: Q_{j+2} = -2Q_{j-1} - 3Q_j + 6Q_{j+1} - 6dy*dQ/dy  [Eq. 32]
                    rho_arr(i, j_bnd + 2, k) = -2.0 * rho_arr(i, j_bnd - 1, k) - 3.0 * rho_bnd + 6.0 * rho_arr(i, j_bnd + 1, k) - 6.0 * delta * drho_bc;
                    u_arr(i, j_bnd + 2, k) = -2.0 * u_arr(i, j_bnd - 1, k) - 3.0 * u_bnd + 6.0 * u_arr(i, j_bnd + 1, k) - 6.0 * delta * du_bc;
                    v_arr(i, j_bnd + 2, k) = -2.0 * v_arr(i, j_bnd - 1, k) - 3.0 * v_bnd + 6.0 * v_arr(i, j_bnd + 1, k) - 6.0 * delta * dv_bc;
                    w_arr(i, j_bnd + 2, k) = -2.0 * w_arr(i, j_bnd - 1, k) - 3.0 * w_bnd + 6.0 * w_arr(i, j_bnd + 1, k) - 6.0 * delta * dw_bc;
                    p_arr(i, j_bnd + 2, k) = -2.0 * p_arr(i, j_bnd - 1, k) - 3.0 * p_bnd + 6.0 * p_arr(i, j_bnd + 1, k) - 6.0 * delta * dp_bc;

                    // Third ghost cell: Q_{j+3} = 3Q_{j-1} + 10Q_j - 18Q_{j+1} + 6Q_{j+2} + 12dy*dQ/dy  [Eq. 33]
                    rho_arr(i, j_bnd + 3, k) = 3.0 * rho_arr(i, j_bnd - 1, k) + 10.0 * rho_bnd - 18.0 * rho_arr(i, j_bnd + 1, k) + 6.0 * rho_arr(i, j_bnd + 2, k) + 12.0 * delta * drho_bc;
                    u_arr(i, j_bnd + 3, k) = 3.0 * u_arr(i, j_bnd - 1, k) + 10.0 * u_bnd - 18.0 * u_arr(i, j_bnd + 1, k) + 6.0 * u_arr(i, j_bnd + 2, k) + 12.0 * delta * du_bc;
                    v_arr(i, j_bnd + 3, k) = 3.0 * v_arr(i, j_bnd - 1, k) + 10.0 * v_bnd - 18.0 * v_arr(i, j_bnd + 1, k) + 6.0 * v_arr(i, j_bnd + 2, k) + 12.0 * delta * dv_bc;
                    w_arr(i, j_bnd + 3, k) = 3.0 * w_arr(i, j_bnd - 1, k) + 10.0 * w_bnd - 18.0 * w_arr(i, j_bnd + 1, k) + 6.0 * w_arr(i, j_bnd + 2, k) + 12.0 * delta * dw_bc;
                    p_arr(i, j_bnd + 3, k) = 3.0 * p_arr(i, j_bnd - 1, k) + 10.0 * p_bnd - 18.0 * p_arr(i, j_bnd + 1, k) + 6.0 * p_arr(i, j_bnd + 2, k) + 12.0 * delta * dp_bc;

                    // Fourth ghost cell: Q_{j+4} = -2Q_{j-1} - 13Q_j + 24Q_{j+1} - 12Q_{j+2} + 4Q_{j+3} - 12dy*dQ/dy  [Eq. 34]
                    rho_arr(i, j_bnd + 4, k) = -2.0 * rho_arr(i, j_bnd - 1, k) - 13.0 * rho_bnd + 24.0 * rho_arr(i, j_bnd + 1, k) - 12.0 * rho_arr(i, j_bnd + 2, k) + 4.0 * rho_arr(i, j_bnd + 3, k) - 12.0 * delta * drho_bc;
                    u_arr(i, j_bnd + 4, k) = -2.0 * u_arr(i, j_bnd - 1, k) - 13.0 * u_bnd + 24.0 * u_arr(i, j_bnd + 1, k) - 12.0 * u_arr(i, j_bnd + 2, k) + 4.0 * u_arr(i, j_bnd + 3, k) - 12.0 * delta * du_bc;
                    v_arr(i, j_bnd + 4, k) = -2.0 * v_arr(i, j_bnd - 1, k) - 13.0 * v_bnd + 24.0 * v_arr(i, j_bnd + 1, k) - 12.0 * v_arr(i, j_bnd + 2, k) + 4.0 * v_arr(i, j_bnd + 3, k) - 12.0 * delta * dv_bc;
                    w_arr(i, j_bnd + 4, k) = -2.0 * w_arr(i, j_bnd - 1, k) - 13.0 * w_bnd + 24.0 * w_arr(i, j_bnd + 1, k) - 12.0 * w_arr(i, j_bnd + 2, k) + 4.0 * w_arr(i, j_bnd + 3, k) - 12.0 * delta * dw_bc;
                    p_arr(i, j_bnd + 4, k) = -2.0 * p_arr(i, j_bnd - 1, k) - 13.0 * p_bnd + 24.0 * p_arr(i, j_bnd + 1, k) - 12.0 * p_arr(i, j_bnd + 2, k) + 4.0 * p_arr(i, j_bnd + 3, k) - 12.0 * delta * dp_bc;

                    // Convert primitive to conservative
                    for (int jg = j_bnd + 1; jg <= j_bnd + 4; jg++)
                    {
                        Set::Scalar rho_g = std::max(small_local, rho_arr(i, jg, k));
                        Set::Scalar u_g = u_arr(i, jg, k);
                        Set::Scalar v_g = v_arr(i, jg, k);
                        Set::Scalar w_g = w_arr(i, jg, k);
                        Set::Scalar p_g = std::max(small_local, p_arr(i, jg, k));

                        rho_arr(i, jg, k) = rho_g;
                        momx_arr(i, jg, k) = rho_g * u_g;
                        momy_arr(i, jg, k) = rho_g * v_g;
#if AMREX_SPACEDIM == 3
                        momz_arr(i, jg, k) = rho_g * w_g;
#endif

                        Set::Scalar ke_g = 0.5 * (u_g * u_g + v_g * v_g + w_g * w_g);
                        Set::Scalar ie_g = (p_g + pref_local + gamma_bnd * p0_bnd) / ((gamma_bnd - 1.0) * rho_g);
                        energy_arr(i, jg, k) = rho_g * (ie_g + ke_g);
                    }
                });
            }

#if AMREX_SPACEDIM == 3
            // ====================================================================
            // Z-LO BOUNDARY (k = domain.smallEnd(2))
            // ====================================================================
            if (bx.smallEnd(2) == domain.smallEnd(2))
            {
                int idir = 2;   // z-direction
                int isign = -1; // bottom boundary
                Set::Scalar delta = dx[2];

                amrex::Box ghost_box = amrex::adjCellLo(bx, 2, 1);

                amrex::ParallelFor(ghost_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    // Boundary cell index (last interior cell)
                    int k_bnd = domain.smallEnd(2);

                    // Get boundary state
                    Set::Scalar rho_bnd = rho_arr(i, j, k_bnd);
                    Set::Scalar u_bnd = u_arr(i, j, k_bnd);
                    Set::Scalar v_bnd = v_arr(i, j, k_bnd);
                    Set::Scalar w_bnd = w_arr(i, j, k_bnd);
                    Set::Scalar p_bnd = p_arr(i, j, k_bnd);
                    Set::Scalar T_bnd = T_arr(i, j, k_bnd);
                    Set::Scalar c_bnd = c_arr(i, j, k_bnd);
                    Set::Scalar gamma_bnd = gamma_arr(i, j, k_bnd);
                    Set::Scalar p0_bnd = p0_arr(i, j, k_bnd);

                    // Compute normal derivatives (one-sided, 2nd order)
                    // dQ/dz ~= (-3Q_k + 4Q_{k+1} - Q_{k+2}) / (2dz)
                    Set::Scalar drho_dz = (-3.0 * rho_arr(i, j, k_bnd) + 4.0 * rho_arr(i, j, k_bnd + 1) - rho_arr(i, j, k_bnd + 2)) / (2.0 * delta);
                    Set::Scalar du_dz = (-3.0 * u_arr(i, j, k_bnd) + 4.0 * u_arr(i, j, k_bnd + 1) - u_arr(i, j, k_bnd + 2)) / (2.0 * delta);
                    Set::Scalar dv_dz = (-3.0 * v_arr(i, j, k_bnd) + 4.0 * v_arr(i, j, k_bnd + 1) - v_arr(i, j, k_bnd + 2)) / (2.0 * delta);
                    Set::Scalar dw_dz = (-3.0 * w_arr(i, j, k_bnd) + 4.0 * w_arr(i, j, k_bnd + 1) - w_arr(i, j, k_bnd + 2)) / (2.0 * delta);
                    Set::Scalar dp_dz = (-3.0 * p_arr(i, j, k_bnd) + 4.0 * p_arr(i, j, k_bnd + 1) - p_arr(i, j, k_bnd + 2)) / (2.0 * delta);

                    // Compute tangential derivatives (centered, 2nd order)
                    // X-direction: dQ/dx ~= (Q_{i+1} - Q_{i-1}) / (2dx)
                    Set::Scalar drho_dx = 0.0, du_dx = 0.0, dv_dx = 0.0, dw_dx = 0.0, dp_dx = 0.0;
                    if (i > domain.smallEnd(0) && i < domain.bigEnd(0))
                    {
                        drho_dx = (rho_arr(i + 1, j, k_bnd) - rho_arr(i - 1, j, k_bnd)) / (2.0 * dx[0]);
                        du_dx = (u_arr(i + 1, j, k_bnd) - u_arr(i - 1, j, k_bnd)) / (2.0 * dx[0]);
                        dv_dx = (v_arr(i + 1, j, k_bnd) - v_arr(i - 1, j, k_bnd)) / (2.0 * dx[0]);
                        dw_dx = (w_arr(i + 1, j, k_bnd) - w_arr(i - 1, j, k_bnd)) / (2.0 * dx[0]);
                        dp_dx = (p_arr(i + 1, j, k_bnd) - p_arr(i - 1, j, k_bnd)) / (2.0 * dx[0]);
                    }

                    // Y-direction: dQ/dy ~= (Q_{j+1} - Q_{j-1}) / (2dy)
                    Set::Scalar drho_dy = 0.0, du_dy = 0.0, dv_dy = 0.0, dw_dy = 0.0, dp_dy = 0.0;
                    if (j > domain.smallEnd(1) && j < domain.bigEnd(1))
                    {
                        drho_dy = (rho_arr(i, j + 1, k_bnd) - rho_arr(i, j - 1, k_bnd)) / (2.0 * dx[1]);
                        du_dy = (u_arr(i, j + 1, k_bnd) - u_arr(i, j - 1, k_bnd)) / (2.0 * dx[1]);
                        dv_dy = (v_arr(i, j + 1, k_bnd) - v_arr(i, j - 1, k_bnd)) / (2.0 * dx[1]);
                        dw_dy = (w_arr(i, j + 1, k_bnd) - w_arr(i, j - 1, k_bnd)) / (2.0 * dx[1]);
                        dp_dy = (p_arr(i, j + 1, k_bnd) - p_arr(i, j - 1, k_bnd)) / (2.0 * dx[1]);
                    }

                    // Compute transverse terms (Section III.B, Motheau et al. 2018)
                    // For z-normal boundary: transverse = x and y directions
                    // T_1 = u*dp/dx + v*dp/dy - rhoc^2(u*drho/dx + v*drho/dy)
                    Set::Scalar T1 = u_bnd * dp_dx + v_bnd * dp_dy - rho_bnd * c_bnd * c_bnd * (u_bnd * drho_dx + v_bnd * drho_dy);
                    Set::Scalar T2 = u_bnd * dw_dx + v_bnd * dw_dy;
                    Set::Scalar T3 = u_bnd * du_dx + v_bnd * du_dy;
                    Set::Scalar T4 = u_bnd * dv_dx + v_bnd * dv_dy;
                    Set::Scalar T5 = u_bnd * dp_dx + v_bnd * dp_dy;

                    // Compute characteristic waves (LODI system)
                    // Wave speeds: lambda_1 = w - c, lambda_2 = w, lambda_3 = w, lambda_4 = w, lambda_5 = w + c
                    Set::Scalar lambda1 = w_bnd - c_bnd;
                    Set::Scalar lambda2 = w_bnd;
                    Set::Scalar lambda3 = w_bnd;
                    Set::Scalar lambda4 = w_bnd;
                    Set::Scalar lambda5 = w_bnd + c_bnd;

                    // Interior wave amplitudes (from spatial derivatives)
                    // Equations from Motheau et al. (2018), Section III
                    Set::Scalar L1_int = lambda1 * (dp_dz - rho_bnd * c_bnd * dw_dz);
                    Set::Scalar L2_int = lambda2 * du_dz;
                    Set::Scalar L3_int = lambda3 * dv_dz;
                    Set::Scalar L4_int = lambda4 * (c_bnd * c_bnd * drho_dz - dp_dz);
                    Set::Scalar L5_int = lambda5 * (dp_dz + rho_bnd * c_bnd * dw_dz);

                    // Add transverse terms (weighted by beta)
                    L1_int += params_zlo_local.beta * T1;
                    L2_int += params_zlo_local.beta * T2;
                    L3_int += params_zlo_local.beta * T3;
                    L4_int += params_zlo_local.beta * T4;
                    L5_int += params_zlo_local.beta * T5;

                    // Apply boundary conditions
                    Set::Scalar L1, L2, L3, L4, L5;
                    Set::Scalar M_local = std::abs(w_bnd) / c_bnd; // Local Mach number

                    if (params_zlo_local.type == Type::Inflow)
                    {
                        // SUBSONIC INFLOW
                        Set::Scalar K_u = params_zlo_local.relax_u * (1.0 - M_local * M_local) * c_bnd / params_zlo_local.L_ref;
                        Set::Scalar K_T = params_zlo_local.relax_T * (1.0 - M_local * M_local) * c_bnd / params_zlo_local.L_ref;

                        L1 = L1_int; // Outgoing wave from interior
                        L2 = K_u * (u_bnd - params_zlo_local.target_u);
                        L3 = K_u * (v_bnd - params_zlo_local.target_v);
                        L4 = K_T * c_bnd * c_bnd * (T_bnd - params_zlo_local.target_T) / T_bnd;
                        L5 = K_u * (w_bnd - params_zlo_local.target_w);
                    }
                    else if (params_zlo_local.type == Type::Outflow)
                    {
                        // SUBSONIC OUTFLOW
                        Set::Scalar K_p = params_zlo_local.sigma * (1.0 - M_local * M_local) * c_bnd / params_zlo_local.L_ref;

                        L1 = K_p * (p_bnd - params_zlo_local.target_p);
                        L2 = L2_int;
                        L3 = L3_int;
                        L4 = L4_int;
                        L5 = L5_int;
                    }
                    else if (params_zlo_local.type == Type::SlipWall)
                    {
                        // SLIP WALL
                        L1 = L5_int; // Reflect outgoing acoustic
                        L2 = L2_int;
                        L3 = L3_int;
                        L4 = L4_int;
                        L5 = L5_int;
                    }
                    else
                    { // NoSlipWall
                        // NO-SLIP WALL
                        Set::Scalar K_wall = 0.5 * c_bnd / params_zlo_local.L_ref;

                        L1 = L5_int;
                        L2 = -K_wall * u_bnd;
                        L3 = -K_wall * v_bnd;
                        L4 = L4_int;
                        L5 = L5_int;
                    }

                    // Convert waves back to primitive variable gradients
                    // Equation (26): dQ/dz = S(z)Lambda^-1L
                    Set::Scalar drho_bc = (L5 + L1) / (c_bnd * c_bnd) + L4 / (c_bnd * c_bnd);
                    Set::Scalar du_bc = L2;
                    Set::Scalar dv_bc = L3;
                    Set::Scalar dw_bc = (L5 - L1) / (2.0 * rho_bnd * c_bnd);
                    Set::Scalar dp_bc = (L5 + L1) / 2.0;

                    // Extrapolate to ghost cells using finite differences
                    // Equations (27)-(30) from Motheau et al. (2018)

                    // First ghost cell: Q_{k-1} = Q_k - dz*dQ/dz  [Eq. 27]
                    rho_arr(i, j, k_bnd - 1) = rho_bnd - delta * drho_bc;
                    u_arr(i, j, k_bnd - 1) = u_bnd - delta * du_bc;
                    v_arr(i, j, k_bnd - 1) = v_bnd - delta * dv_bc;
                    w_arr(i, j, k_bnd - 1) = w_bnd - delta * dw_bc;
                    p_arr(i, j, k_bnd - 1) = p_bnd - delta * dp_bc;

                    // Second ghost cell: Q_{k-2} = -2Q_{k+1} - 3Q_k + 6Q_{k-1} + 6dz*dQ/dz  [Eq. 28]
                    rho_arr(i, j, k_bnd - 2) = -2.0 * rho_arr(i, j, k_bnd + 1) - 3.0 * rho_bnd + 6.0 * rho_arr(i, j, k_bnd - 1) + 6.0 * delta * drho_bc;
                    u_arr(i, j, k_bnd - 2) = -2.0 * u_arr(i, j, k_bnd + 1) - 3.0 * u_bnd + 6.0 * u_arr(i, j, k_bnd - 1) + 6.0 * delta * du_bc;
                    v_arr(i, j, k_bnd - 2) = -2.0 * v_arr(i, j, k_bnd + 1) - 3.0 * v_bnd + 6.0 * v_arr(i, j, k_bnd - 1) + 6.0 * delta * dv_bc;
                    w_arr(i, j, k_bnd - 2) = -2.0 * w_arr(i, j, k_bnd + 1) - 3.0 * w_bnd + 6.0 * w_arr(i, j, k_bnd - 1) + 6.0 * delta * dw_bc;
                    p_arr(i, j, k_bnd - 2) = -2.0 * p_arr(i, j, k_bnd + 1) - 3.0 * p_bnd + 6.0 * p_arr(i, j, k_bnd - 1) + 6.0 * delta * dp_bc;

                    // Third ghost cell: Q_{k-3} = 3Q_{k+1} + 10Q_k - 18Q_{k-1} + 6Q_{k-2} - 12dz*dQ/dz  [Eq. 29]
                    rho_arr(i, j, k_bnd - 3) = 3.0 * rho_arr(i, j, k_bnd + 1) + 10.0 * rho_bnd - 18.0 * rho_arr(i, j, k_bnd - 1) + 6.0 * rho_arr(i, j, k_bnd - 2) - 12.0 * delta * drho_bc;
                    u_arr(i, j, k_bnd - 3) = 3.0 * u_arr(i, j, k_bnd + 1) + 10.0 * u_bnd - 18.0 * u_arr(i, j, k_bnd - 1) + 6.0 * u_arr(i, j, k_bnd - 2) - 12.0 * delta * du_bc;
                    v_arr(i, j, k_bnd - 3) = 3.0 * v_arr(i, j, k_bnd + 1) + 10.0 * v_bnd - 18.0 * v_arr(i, j, k_bnd - 1) + 6.0 * v_arr(i, j, k_bnd - 2) - 12.0 * delta * dv_bc;
                    w_arr(i, j, k_bnd - 3) = 3.0 * w_arr(i, j, k_bnd + 1) + 10.0 * w_bnd - 18.0 * w_arr(i, j, k_bnd - 1) + 6.0 * w_arr(i, j, k_bnd - 2) - 12.0 * delta * dw_bc;
                    p_arr(i, j, k_bnd - 3) = 3.0 * p_arr(i, j, k_bnd + 1) + 10.0 * p_bnd - 18.0 * p_arr(i, j, k_bnd - 1) + 6.0 * p_arr(i, j, k_bnd - 2) - 12.0 * delta * dp_bc;

                    // Fourth ghost cell: Q_{k-4} = -2Q_{k+1} - 13Q_k + 24Q_{k-1} - 12Q_{k-2} + 4Q_{k-3} + 12dz*dQ/dz  [Eq. 30]
                    rho_arr(i, j, k_bnd - 4) = -2.0 * rho_arr(i, j, k_bnd + 1) - 13.0 * rho_bnd + 24.0 * rho_arr(i, j, k_bnd - 1) - 12.0 * rho_arr(i, j, k_bnd - 2) + 4.0 * rho_arr(i, j, k_bnd - 3) + 12.0 * delta * drho_bc;
                    u_arr(i, j, k_bnd - 4) = -2.0 * u_arr(i, j, k_bnd + 1) - 13.0 * u_bnd + 24.0 * u_arr(i, j, k_bnd - 1) - 12.0 * u_arr(i, j, k_bnd - 2) + 4.0 * u_arr(i, j, k_bnd - 3) + 12.0 * delta * du_bc;
                    v_arr(i, j, k_bnd - 4) = -2.0 * v_arr(i, j, k_bnd + 1) - 13.0 * v_bnd + 24.0 * v_arr(i, j, k_bnd - 1) - 12.0 * v_arr(i, j, k_bnd - 2) + 4.0 * v_arr(i, j, k_bnd - 3) + 12.0 * delta * dv_bc;
                    w_arr(i, j, k_bnd - 4) = -2.0 * w_arr(i, j, k_bnd + 1) - 13.0 * w_bnd + 24.0 * w_arr(i, j, k_bnd - 1) - 12.0 * w_arr(i, j, k_bnd - 2) + 4.0 * w_arr(i, j, k_bnd - 3) + 12.0 * delta * dw_bc;
                    p_arr(i, j, k_bnd - 4) = -2.0 * p_arr(i, j, k_bnd + 1) - 13.0 * p_bnd + 24.0 * p_arr(i, j, k_bnd - 1) - 12.0 * p_arr(i, j, k_bnd - 2) + 4.0 * p_arr(i, j, k_bnd - 3) + 12.0 * delta * dp_bc;

                    // Convert primitive back to conservative in ghost cells
                    for (int kg = k_bnd - 1; kg >= k_bnd - 4; kg--)
                    {
                        Set::Scalar rho_g = std::max(small_local, rho_arr(i, j, kg));
                        Set::Scalar u_g = u_arr(i, j, kg);
                        Set::Scalar v_g = v_arr(i, j, kg);
                        Set::Scalar w_g = w_arr(i, j, kg);
                        Set::Scalar p_g = std::max(small_local, p_arr(i, j, kg));

                        // Conservative variables
                        rho_arr(i, j, kg) = rho_g;
                        momx_arr(i, j, kg) = rho_g * u_g;
                        momy_arr(i, j, kg) = rho_g * v_g;
                        momz_arr(i, j, kg) = rho_g * w_g;

                        // Total energy: E = rhoe + 1/2rho(u^2 + v^2 + w^2)
                        Set::Scalar ke_g = 0.5 * (u_g * u_g + v_g * v_g + w_g * w_g);
                        Set::Scalar ie_g = (p_g + pref_local + gamma_bnd * p0_bnd) / ((gamma_bnd - 1.0) * rho_g);
                        energy_arr(i, j, kg) = rho_g * (ie_g + ke_g);
                    }
                });
            }

            // ====================================================================
            // Z-HI BOUNDARY (k = domain.bigEnd(2))
            // ====================================================================
            if (bx.bigEnd(2) == domain.bigEnd(2))
            {
                int idir = 2;
                int isign = 1;
                Set::Scalar delta = dx[2];

                amrex::Box ghost_box = amrex::adjCellHi(bx, 2, 1);

                amrex::ParallelFor(ghost_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    int k_bnd = domain.bigEnd(2);

                    Set::Scalar rho_bnd = rho_arr(i, j, k_bnd);
                    Set::Scalar u_bnd = u_arr(i, j, k_bnd);
                    Set::Scalar v_bnd = v_arr(i, j, k_bnd);
                    Set::Scalar w_bnd = w_arr(i, j, k_bnd);
                    Set::Scalar p_bnd = p_arr(i, j, k_bnd);
                    Set::Scalar T_bnd = T_arr(i, j, k_bnd);
                    Set::Scalar c_bnd = c_arr(i, j, k_bnd);
                    Set::Scalar gamma_bnd = gamma_arr(i, j, k_bnd);
                    Set::Scalar p0_bnd = p0_arr(i, j, k_bnd);

                    // Normal derivatives (backward difference for top boundary)
                    // dQ/dz ~= (3Q_k - 4Q_{k-1} + Q_{k-2}) / (2dz)
                    Set::Scalar drho_dz = (3.0 * rho_arr(i, j, k_bnd) - 4.0 * rho_arr(i, j, k_bnd - 1) + rho_arr(i, j, k_bnd - 2)) / (2.0 * delta);
                    Set::Scalar du_dz = (3.0 * u_arr(i, j, k_bnd) - 4.0 * u_arr(i, j, k_bnd - 1) + u_arr(i, j, k_bnd - 2)) / (2.0 * delta);
                    Set::Scalar dv_dz = (3.0 * v_arr(i, j, k_bnd) - 4.0 * v_arr(i, j, k_bnd - 1) + v_arr(i, j, k_bnd - 2)) / (2.0 * delta);
                    Set::Scalar dw_dz = (3.0 * w_arr(i, j, k_bnd) - 4.0 * w_arr(i, j, k_bnd - 1) + w_arr(i, j, k_bnd - 2)) / (2.0 * delta);
                    Set::Scalar dp_dz = (3.0 * p_arr(i, j, k_bnd) - 4.0 * p_arr(i, j, k_bnd - 1) + p_arr(i, j, k_bnd - 2)) / (2.0 * delta);

                    // Tangential derivatives (same as z-lo)
                    Set::Scalar drho_dx = 0.0, du_dx = 0.0, dv_dx = 0.0, dw_dx = 0.0, dp_dx = 0.0;
                    if (i > domain.smallEnd(0) && i < domain.bigEnd(0))
                    {
                        drho_dx = (rho_arr(i + 1, j, k_bnd) - rho_arr(i - 1, j, k_bnd)) / (2.0 * dx[0]);
                        du_dx = (u_arr(i + 1, j, k_bnd) - u_arr(i - 1, j, k_bnd)) / (2.0 * dx[0]);
                        dv_dx = (v_arr(i + 1, j, k_bnd) - v_arr(i - 1, j, k_bnd)) / (2.0 * dx[0]);
                        dw_dx = (w_arr(i + 1, j, k_bnd) - w_arr(i - 1, j, k_bnd)) / (2.0 * dx[0]);
                        dp_dx = (p_arr(i + 1, j, k_bnd) - p_arr(i - 1, j, k_bnd)) / (2.0 * dx[0]);
                    }

                    Set::Scalar drho_dy = 0.0, du_dy = 0.0, dv_dy = 0.0, dw_dy = 0.0, dp_dy = 0.0;
                    if (j > domain.smallEnd(1) && j < domain.bigEnd(1))
                    {
                        drho_dy = (rho_arr(i, j + 1, k_bnd) - rho_arr(i, j - 1, k_bnd)) / (2.0 * dx[1]);
                        du_dy = (u_arr(i, j + 1, k_bnd) - u_arr(i, j - 1, k_bnd)) / (2.0 * dx[1]);
                        dv_dy = (v_arr(i, j + 1, k_bnd) - v_arr(i, j - 1, k_bnd)) / (2.0 * dx[1]);
                        dw_dy = (w_arr(i, j + 1, k_bnd) - w_arr(i, j - 1, k_bnd)) / (2.0 * dx[1]);
                        dp_dy = (p_arr(i, j + 1, k_bnd) - p_arr(i, j - 1, k_bnd)) / (2.0 * dx[1]);
                    }

                    // Transverse terms
                    Set::Scalar T1 = u_bnd * dp_dx + v_bnd * dp_dy - rho_bnd * c_bnd * c_bnd * (u_bnd * drho_dx + v_bnd * drho_dy);
                    Set::Scalar T2 = u_bnd * dw_dx + v_bnd * dw_dy;
                    Set::Scalar T3 = u_bnd * du_dx + v_bnd * du_dy;
                    Set::Scalar T4 = u_bnd * dv_dx + v_bnd * dv_dy;
                    Set::Scalar T5 = u_bnd * dp_dx + v_bnd * dp_dy;

                    // Characteristic waves
                    Set::Scalar lambda1 = w_bnd - c_bnd;
                    Set::Scalar lambda2 = w_bnd;
                    Set::Scalar lambda3 = w_bnd;
                    Set::Scalar lambda4 = w_bnd;
                    Set::Scalar lambda5 = w_bnd + c_bnd;

                    Set::Scalar L1_int = lambda1 * (dp_dz - rho_bnd * c_bnd * dw_dz);
                    Set::Scalar L2_int = lambda2 * du_dz;
                    Set::Scalar L3_int = lambda3 * dv_dz;
                    Set::Scalar L4_int = lambda4 * (c_bnd * c_bnd * drho_dz - dp_dz);
                    Set::Scalar L5_int = lambda5 * (dp_dz + rho_bnd * c_bnd * dw_dz);

                    L1_int += params_zhi_local.beta * T1;
                    L2_int += params_zhi_local.beta * T2;
                    L3_int += params_zhi_local.beta * T3;
                    L4_int += params_zhi_local.beta * T4;
                    L5_int += params_zhi_local.beta * T5;

                    // Apply boundary conditions
                    Set::Scalar L1, L2, L3, L4, L5;
                    Set::Scalar M_local = std::abs(w_bnd) / c_bnd;

                    if (params_zhi_local.type == Type::Inflow)
                    {
                        Set::Scalar K_u = params_zhi_local.relax_u * (1.0 - M_local * M_local) * c_bnd / params_zhi_local.L_ref;
                        Set::Scalar K_T = params_zhi_local.relax_T * (1.0 - M_local * M_local) * c_bnd / params_zhi_local.L_ref;

                        L1 = K_u * (w_bnd - params_zhi_local.target_w);
                        L2 = K_u * (u_bnd - params_zhi_local.target_u);
                        L3 = K_u * (v_bnd - params_zhi_local.target_v);
                        L4 = K_T * c_bnd * c_bnd * (T_bnd - params_zhi_local.target_T) / T_bnd;
                        L5 = L5_int;
                    }
                    else if (params_zhi_local.type == Type::Outflow)
                    {
                        Set::Scalar K_p = params_zhi_local.sigma * (1.0 - M_local * M_local) * c_bnd / params_zhi_local.L_ref;

                        L1 = L1_int;
                        L2 = L2_int;
                        L3 = L3_int;
                        L4 = L4_int;
                        L5 = K_p * (p_bnd - params_zhi_local.target_p);
                    }
                    else if (params_zhi_local.type == Type::SlipWall)
                    {
                        L1 = L1_int;
                        L2 = L2_int;
                        L3 = L3_int;
                        L4 = L4_int;
                        L5 = L1_int;
                    }
                    else
                    { // NoSlipWall
                        Set::Scalar K_wall = 0.5 * c_bnd / params_zhi_local.L_ref;

                        L1 = L1_int;
                        L2 = -K_wall * u_bnd;
                        L3 = -K_wall * v_bnd;
                        L4 = L4_int;
                        L5 = L1_int;
                    }

                    // Convert waves to gradients [Eq. 26]
                    Set::Scalar drho_bc = (L5 + L1) / (c_bnd * c_bnd) + L4 / (c_bnd * c_bnd);
                    Set::Scalar du_bc = L2;
                    Set::Scalar dv_bc = L3;
                    Set::Scalar dw_bc = (L5 - L1) / (2.0 * rho_bnd * c_bnd);
                    Set::Scalar dp_bc = (L5 + L1) / 2.0;

                    // Extrapolate to ghost cells [Eqs. 31-34]
                    // First ghost cell: Q_{k+1} = Q_k + dz*dQ/dz  [Eq. 31]
                    rho_arr(i, j, k_bnd + 1) = rho_bnd + delta * drho_bc;
                    u_arr(i, j, k_bnd + 1) = u_bnd + delta * du_bc;
                    v_arr(i, j, k_bnd + 1) = v_bnd + delta * dv_bc;
                    w_arr(i, j, k_bnd + 1) = w_bnd + delta * dw_bc;
                    p_arr(i, j, k_bnd + 1) = p_bnd + delta * dp_bc;

                    // Second ghost cell: Q_{k+2} = -2Q_{k-1} - 3Q_k + 6Q_{k+1} - 6dz*dQ/dz  [Eq. 32]
                    rho_arr(i, j, k_bnd + 2) = -2.0 * rho_arr(i, j, k_bnd - 1) - 3.0 * rho_bnd + 6.0 * rho_arr(i, j, k_bnd + 1) - 6.0 * delta * drho_bc;
                    u_arr(i, j, k_bnd + 2) = -2.0 * u_arr(i, j, k_bnd - 1) - 3.0 * u_bnd + 6.0 * u_arr(i, j, k_bnd + 1) - 6.0 * delta * du_bc;
                    v_arr(i, j, k_bnd + 2) = -2.0 * v_arr(i, j, k_bnd - 1) - 3.0 * v_bnd + 6.0 * v_arr(i, j, k_bnd + 1) - 6.0 * delta * dv_bc;
                    w_arr(i, j, k_bnd + 2) = -2.0 * w_arr(i, j, k_bnd - 1) - 3.0 * w_bnd + 6.0 * w_arr(i, j, k_bnd + 1) - 6.0 * delta * dw_bc;
                    p_arr(i, j, k_bnd + 2) = -2.0 * p_arr(i, j, k_bnd - 1) - 3.0 * p_bnd + 6.0 * p_arr(i, j, k_bnd + 1) - 6.0 * delta * dp_bc;

                    // Third ghost cell: Q_{k+3} = 3Q_{k-1} + 10Q_k - 18Q_{k+1} + 6Q_{k+2} + 12dz*dQ/dz  [Eq. 33]
                    rho_arr(i, j, k_bnd + 3) = 3.0 * rho_arr(i, j, k_bnd - 1) + 10.0 * rho_bnd - 18.0 * rho_arr(i, j, k_bnd + 1) + 6.0 * rho_arr(i, j, k_bnd + 2) + 12.0 * delta * drho_bc;
                    u_arr(i, j, k_bnd + 3) = 3.0 * u_arr(i, j, k_bnd - 1) + 10.0 * u_bnd - 18.0 * u_arr(i, j, k_bnd + 1) + 6.0 * u_arr(i, j, k_bnd + 2) + 12.0 * delta * du_bc;
                    v_arr(i, j, k_bnd + 3) = 3.0 * v_arr(i, j, k_bnd - 1) + 10.0 * v_bnd - 18.0 * v_arr(i, j, k_bnd + 1) + 6.0 * v_arr(i, j, k_bnd + 2) + 12.0 * delta * dv_bc;
                    w_arr(i, j, k_bnd + 3) = 3.0 * w_arr(i, j, k_bnd - 1) + 10.0 * w_bnd - 18.0 * w_arr(i, j, k_bnd + 1) + 6.0 * w_arr(i, j, k_bnd + 2) + 12.0 * delta * dw_bc;
                    p_arr(i, j, k_bnd + 3) = 3.0 * p_arr(i, j, k_bnd - 1) + 10.0 * p_bnd - 18.0 * p_arr(i, j, k_bnd + 1) + 6.0 * p_arr(i, j, k_bnd + 2) + 12.0 * delta * dp_bc;

                    // Fourth ghost cell: Q_{k+4} = -2Q_{k-1} - 13Q_k + 24Q_{k+1} - 12Q_{k+2} + 4Q_{k+3} - 12dz*dQ/dz  [Eq. 34]
                    rho_arr(i, j, k_bnd + 4) = -2.0 * rho_arr(i, j, k_bnd - 1) - 13.0 * rho_bnd + 24.0 * rho_arr(i, j, k_bnd + 1) - 12.0 * rho_arr(i, j, k_bnd + 2) + 4.0 * rho_arr(i, j, k_bnd + 3) - 12.0 * delta * drho_bc;
                    u_arr(i, j, k_bnd + 4) = -2.0 * u_arr(i, j, k_bnd - 1) - 13.0 * u_bnd + 24.0 * u_arr(i, j, k_bnd + 1) - 12.0 * u_arr(i, j, k_bnd + 2) + 4.0 * u_arr(i, j, k_bnd + 3) - 12.0 * delta * du_bc;
                    v_arr(i, j, k_bnd + 4) = -2.0 * v_arr(i, j, k_bnd - 1) - 13.0 * v_bnd + 24.0 * v_arr(i, j, k_bnd + 1) - 12.0 * v_arr(i, j, k_bnd + 2) + 4.0 * v_arr(i, j, k_bnd + 3) - 12.0 * delta * dv_bc;
                    w_arr(i, j, k_bnd + 4) = -2.0 * w_arr(i, j, k_bnd - 1) - 13.0 * w_bnd + 24.0 * w_arr(i, j, k_bnd + 1) - 12.0 * w_arr(i, j, k_bnd + 2) + 4.0 * w_arr(i, j, k_bnd + 3) - 12.0 * delta * dw_bc;
                    p_arr(i, j, k_bnd + 4) = -2.0 * p_arr(i, j, k_bnd - 1) - 13.0 * p_bnd + 24.0 * p_arr(i, j, k_bnd + 1) - 12.0 * p_arr(i, j, k_bnd + 2) + 4.0 * p_arr(i, j, k_bnd + 3) - 12.0 * delta * dp_bc;

                    // Convert primitive to conservative
                    for (int kg = k_bnd + 1; kg <= k_bnd + 4; kg++)
                    {
                        Set::Scalar rho_g = std::max(small_local, rho_arr(i, j, kg));
                        Set::Scalar u_g = u_arr(i, j, kg);
                        Set::Scalar v_g = v_arr(i, j, kg);
                        Set::Scalar w_g = w_arr(i, j, kg);
                        Set::Scalar p_g = std::max(small_local, p_arr(i, j, kg));

                        rho_arr(i, j, kg) = rho_g;
                        momx_arr(i, j, kg) = rho_g * u_g;
                        momy_arr(i, j, kg) = rho_g * v_g;
                        momz_arr(i, j, kg) = rho_g * w_g;

                        Set::Scalar ke_g = 0.5 * (u_g * u_g + v_g * v_g + w_g * w_g);
                        Set::Scalar ie_g = (p_g + pref_local + gamma_bnd * p0_bnd) / ((gamma_bnd - 1.0) * rho_g);
                        energy_arr(i, j, kg) = rho_g * (ie_g + ke_g);
                    }
                });
            }
#endif // AMREX_SPACEDIM == 3
        }
    }

private:
    /// Parse parameters for a single boundary face
    static BoundaryParams ParseFace(IO::ParmParse &pp, std::string face_name);

    /// Fill ghost cells for a single box
    void FillBox(const amrex::Box &bx,
                 const amrex::Box &domain,
                 amrex::Array4<Set::Scalar> const &rho,
                 amrex::Array4<Set::Scalar> const &momx,
                 amrex::Array4<Set::Scalar> const &momy,
                 amrex::Array4<Set::Scalar> const &momz,
                 amrex::Array4<Set::Scalar> const &energy,
                 amrex::Array4<Set::Scalar> const &eta,
                 const Solver::EOS::EOS &eos0,
                 const Solver::EOS::EOS &eos1,
                 const amrex::GpuArray<Set::Scalar, AMREX_SPACEDIM> &dx,
                 Set::Scalar time,
                 Set::Scalar pref) const;

    /// Compute normal derivative using 2nd-order one-sided stencil
    /// Equation (not explicitly numbered in paper, but standard 2nd-order backward/forward difference)
    AMREX_GPU_DEVICE
    void NormalDerivative(int i, int j, int k, int idir, int isign, Set::Scalar delta, amrex::Array4<Set::Scalar> const &rho, amrex::Array4<Set::Scalar> const &u, amrex::Array4<Set::Scalar> const &v, amrex::Array4<Set::Scalar> const &w, amrex::Array4<Set::Scalar> const &p, Set::Scalar &drho, Set::Scalar &du, Set::Scalar &dv, Set::Scalar &dw, Set::Scalar &dp) const;

    /// Compute tangential derivative using 2nd-order centered stencil
    AMREX_GPU_DEVICE
    void TangentialDerivative(int i, int j, int k, int idir, Set::Scalar delta, amrex::Array4<Set::Scalar> const &rho, amrex::Array4<Set::Scalar> const &u, amrex::Array4<Set::Scalar> const &v, amrex::Array4<Set::Scalar> const &w, amrex::Array4<Set::Scalar> const &p, Set::Scalar &drho, Set::Scalar &du, Set::Scalar &dv, Set::Scalar &dw, Set::Scalar &dp) const;

    /// Compute transverse terms for multi-dimensional coupling
    /// Equations from Section III.B (transverse terms)
    AMREX_GPU_DEVICE
    void ComputeTransverseTerms(int i, int j, int k, int idir, Set::Scalar rho, Set::Scalar u, Set::Scalar v, Set::Scalar w, Set::Scalar p, Set::Scalar gamma, Set::Scalar c, Set::Scalar dpdx, Set::Scalar dudx, Set::Scalar dvdx, Set::Scalar dwdx, Set::Scalar drhodx, Set::Scalar dpdy, Set::Scalar dudy, Set::Scalar dvdy, Set::Scalar dwdy, Set::Scalar drhody, Set::Scalar dpdz, Set::Scalar dudz, Set::Scalar dvdz, Set::Scalar dwdz, Set::Scalar drhodz, Set::Scalar &T1, Set::Scalar &T2, Set::Scalar &T3, Set::Scalar &T4, Set::Scalar &T5) const;

    /// Compute characteristic waves (LODI system)
    /// Equations (10)-(14) for wave definitions
    /// Equations (15)-(19) for boundary conditions
    AMREX_GPU_DEVICE
    void ComputeWaves(int i, int j, int k, int idir, int isign, const BoundaryParams &params, Set::Scalar rho, Set::Scalar u, Set::Scalar v, Set::Scalar w, Set::Scalar p, Set::Scalar T, Set::Scalar gamma, Set::Scalar c, Set::Scalar dp, Set::Scalar du, Set::Scalar dv, Set::Scalar dw, Set::Scalar drho, Set::Scalar T1, Set::Scalar T2, Set::Scalar T3, Set::Scalar T4, Set::Scalar T5, Set::Scalar &L1, Set::Scalar &L2, Set::Scalar &L3, Set::Scalar &L4, Set::Scalar &L5) const;

    /// Update ghost cells using characteristic waves
    /// Equations (27)-(30) for ghost cell extrapolation
    /// Equation (26) for converting waves back to primitive variables
    AMREX_GPU_DEVICE
    void UpdateGhostCells(int i, int j, int k, int idir, int isign, Set::Scalar delta, const BoundaryParams &params, Set::Scalar L1, Set::Scalar L2, Set::Scalar L3, Set::Scalar L4, Set::Scalar L5, Set::Scalar rho_bnd, Set::Scalar c_bnd, amrex::Array4<Set::Scalar> const &rho, amrex::Array4<Set::Scalar> const &u, amrex::Array4<Set::Scalar> const &v, amrex::Array4<Set::Scalar> const &w, amrex::Array4<Set::Scalar> const &p, const amrex::Box &domain) const;

    /// Convert primitive to conservative variables
    AMREX_GPU_DEVICE
    void PrimitiveToConservative(Set::Scalar rho, Set::Scalar u, Set::Scalar v, Set::Scalar w, Set::Scalar p, Set::Scalar gamma, Set::Scalar p0, Set::Scalar &rho_out, Set::Scalar &momx_out, Set::Scalar &momy_out, Set::Scalar &momz_out, Set::Scalar &E_out, Set::Scalar pref) const;
};

} // namespace BC

#endif
