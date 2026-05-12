// 7-equation Navier-Stokes Characteristic Boundary Condition (NSCBC7).
//
// Per-phase LODI for the BN system. Each phase has its own characteristic
// system {u_k - a_k, u_k, u_k, u_k + a_k}; in the diffuse zone where both
// phases co-exist the boundary fill is run twice with a shared target_p
// (mechanical equilibrium at infinity).
//
// Reference (single-phase NSCBC, used per phase):
//   Motheau et al. AIAA J. 55(10):3399-3408 (2017).
//   Poinsot, Lele.  JCP 101:104-129 (1992).
//
// STATUS: Phase-A scaffolding stub. Per-phase parsing is wired so that input
// files can declare nscbc7.<face>.target_p, etc.; the actual boundary fill
// lands in Phase E. Calling FillBoundary at runtime aborts loudly.
//
#ifndef BC_NSCBC7_H
#define BC_NSCBC7_H

#include "BC/BC.H"
#include "IO/ParmParse.H"
#include "Set/Set.H"
#include "Solver/EOS/EOS.H"
#include "Util/Util.H"
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

namespace BC
{

class NSCBC7
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

    /// Per-face, per-phase BC targets. The mechanical-equilibrium assumption
    /// at the boundary means target_p is shared between phases (single
    /// scalar); per-phase velocity / temperature targets remain split so an
    /// inflow can prescribe e.g. liquid-stream u0 vs gas u1 separately.
    struct BoundaryParams
    {
        Type type = Type::Outflow;

        // Shared (mechanical equilibrium at infinity)
        Set::Scalar target_p = 101325.0;
        Set::Scalar beta     = 0.6;
        Set::Scalar sigma    = 0.25;
        Set::Scalar L_ref    = 1.0;

        // Per-phase velocity/temperature targets and relaxation coefficients
        Set::Scalar target_u0 = 0.0, target_v0 = 0.0, target_w0 = 0.0, target_T0 = 300.0;
        Set::Scalar target_u1 = 0.0, target_v1 = 0.0, target_w1 = 0.0, target_T1 = 300.0;
        Set::Scalar relax_u0  = 0.3, relax_v0  = 0.3, relax_w0  = 0.3, relax_T0  = 0.3;
        Set::Scalar relax_u1  = 0.3, relax_v1  = 0.3, relax_w1  = 0.3, relax_T1  = 0.3;

        // Volume fraction at infinity (advected as passive scalar at u_I*)
        Set::Scalar target_alpha0 = 0.5;
    };

private:
    BoundaryParams params_xlo, params_xhi;
    BoundaryParams params_ylo, params_yhi;
    BoundaryParams params_zlo, params_zhi;
    Set::Scalar small = 1.0e-10;

    static BoundaryParams ParseFace(IO::ParmParse &pp, std::string face_name);

public:
    NSCBC7() = default;
    NSCBC7(IO::ParmParse &pp, std::string /*prefix*/ = "nscbc7") { Parse(*this, pp); }

    static void Parse(NSCBC7 &value, IO::ParmParse &pp)
    {
        pp.query_default("nscbc7.small", value.small, 1.0e-10);
        value.params_xlo = ParseFace(pp, "nscbc7.xlo");
        value.params_xhi = ParseFace(pp, "nscbc7.xhi");
        value.params_ylo = ParseFace(pp, "nscbc7.ylo");
        value.params_yhi = ParseFace(pp, "nscbc7.yhi");
#if AMREX_SPACEDIM == 3
        value.params_zlo = ParseFace(pp, "nscbc7.zlo");
        value.params_zhi = ParseFace(pp, "nscbc7.zhi");
#endif
    }

    /// Phase-E entry point. Phase-A signature placeholder so the integrator
    /// can be wired in advance. Argument list mirrors the 7-eqn conservatives
    /// + per-phase EOS-derived fields.
    template <typename EOS0, typename EOS1>
    void FillBoundary(amrex::MultiFab & /*alpha_rho_0_mf*/,
                      amrex::MultiFab & /*alpha_rho_M0_mf*/,
                      amrex::MultiFab & /*alpha_rho_E0_mf*/,
                      amrex::MultiFab & /*alpha_rho_1_mf*/,
                      amrex::MultiFab & /*alpha_rho_M1_mf*/,
                      amrex::MultiFab & /*alpha_rho_E1_mf*/,
                      amrex::MultiFab & /*alpha0_mf*/,
                      amrex::MultiFab & /*pressure_mf*/,
                      const EOS0 & /*eos0*/,
                      const EOS1 & /*eos1*/,
                      const amrex::Geometry & /*geom*/,
                      Set::Scalar /*time*/,
                      Set::Scalar /*pref*/)
    {
        Util::Abort(INFO,
            "NSCBC7::FillBoundary called, but per-phase LODI fill is a "
            "Phase-E item and not yet implemented. Either keep "
            "equation_count=5 (so this path is unreachable), or finish "
            "Phase E before exercising this code.");
    }
};

inline NSCBC7::BoundaryParams NSCBC7::ParseFace(IO::ParmParse &pp, std::string face_name)
{
    BoundaryParams params;

    std::string type_str = "none";
    std::vector<std::string> type_vec;
    if (pp.queryarr((face_name + ".type").c_str(), type_vec) && !type_vec.empty())
    {
        type_str = type_vec[0];
    }

    if      (type_str == "inflow")     params.type = Type::Inflow;
    else if (type_str == "outflow")    params.type = Type::Outflow;
    else if (type_str == "slipwall")   params.type = Type::SlipWall;
    else if (type_str == "noslipwall") params.type = Type::NoSlipWall;
    else                               params.type = Type::None;

    // Shared boundary-state targets
    pp.query((face_name + ".target_p").c_str(),      params.target_p);
    pp.query((face_name + ".beta").c_str(),          params.beta);
    pp.query((face_name + ".sigma").c_str(),         params.sigma);
    pp.query((face_name + ".L_ref").c_str(),         params.L_ref);
    pp.query((face_name + ".target_alpha0").c_str(), params.target_alpha0);

    // Per-phase targets
    pp.query((face_name + ".target_u0").c_str(), params.target_u0);
    pp.query((face_name + ".target_v0").c_str(), params.target_v0);
    pp.query((face_name + ".target_w0").c_str(), params.target_w0);
    pp.query((face_name + ".target_T0").c_str(), params.target_T0);
    pp.query((face_name + ".target_u1").c_str(), params.target_u1);
    pp.query((face_name + ".target_v1").c_str(), params.target_v1);
    pp.query((face_name + ".target_w1").c_str(), params.target_w1);
    pp.query((face_name + ".target_T1").c_str(), params.target_T1);

    // Per-phase relaxation
    pp.query((face_name + ".relax_u0").c_str(), params.relax_u0);
    pp.query((face_name + ".relax_v0").c_str(), params.relax_v0);
    pp.query((face_name + ".relax_w0").c_str(), params.relax_w0);
    pp.query((face_name + ".relax_T0").c_str(), params.relax_T0);
    pp.query((face_name + ".relax_u1").c_str(), params.relax_u1);
    pp.query((face_name + ".relax_v1").c_str(), params.relax_v1);
    pp.query((face_name + ".relax_w1").c_str(), params.relax_w1);
    pp.query((face_name + ".relax_T1").c_str(), params.relax_T1);

    return params;
}

} // namespace BC

#endif
