#include "BC/NSCBC.H"
#include "Util/Util.H"
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

// SEE THIS PAPER:
// https://hal.science/hal-01699049/file/2018_AIAA_CFD_Motheau.pdf
// Motheau, et. al (2018)
//
// Implementation of the Navier-Stokes Characteristic Boundary Condition (NSCBC)
//
// STABILITY MODIFICATIONS (matching professor's working implementation):
//   - Uses 1st-order FD instead of 2nd-order to eliminate amplification
//   - Uses simple linear extrapolation for 2nd ghost cell (2× vs 6× amplification)
//   - Writes gamma, pi, pressure to ghost cells for Riemann solver
//   - Extensive NaN guards and fallback to boundary state

namespace BC
{

// ============================================================================
// CONSTRUCTOR AND PARSING
// ============================================================================

NSCBC::NSCBC(IO::ParmParse &pp, std::string /*prefix*/)
{
    Parse(*this, pp);
}


void NSCBC::Parse(NSCBC &value, IO::ParmParse &pp)
{
    // Parse global parameters
    pp.query_default("nscbc.small", value.small, 1.0e-10);

    // Parse boundary parameters for each face
    value.params_xlo = ParseFace(pp, "nscbc.xlo");
    value.params_xhi = ParseFace(pp, "nscbc.xhi");
    value.params_ylo = ParseFace(pp, "nscbc.ylo");
    value.params_yhi = ParseFace(pp, "nscbc.yhi");
#if AMREX_SPACEDIM == 3
    value.params_zlo = ParseFace(pp, "nscbc.zlo");
    value.params_zhi = ParseFace(pp, "nscbc.zhi");
#endif
}

NSCBC::BoundaryParams NSCBC::ParseFace(IO::ParmParse &pp, std::string face_name)
{
    BoundaryParams params;

    // Parse boundary type - use queryarr for string
    std::string type_str = "none";
    std::vector<std::string> type_vec;
    if (pp.queryarr((face_name + ".type").c_str(), type_vec) && !type_vec.empty())
    {
        type_str = type_vec[0];
    }

    if (type_str == "inflow")
    {
        params.type = Type::Inflow;
    }
    else if (type_str == "outflow")
    {
        params.type = Type::Outflow;
    }
    else if (type_str == "slipwall")
    {
        params.type = Type::SlipWall;
    }
    else if (type_str == "noslipwall")
    {
        params.type = Type::NoSlipWall;
    }
    else
    {
        params.type = Type::None;
    }

    // Parse target values (with defaults) - use .c_str()
    pp.query((face_name + ".target_u").c_str(), params.target_u);
    pp.query((face_name + ".target_v").c_str(), params.target_v);
    pp.query((face_name + ".target_w").c_str(), params.target_w);
    pp.query((face_name + ".target_T").c_str(), params.target_T);
    pp.query((face_name + ".target_p").c_str(), params.target_p);
    pp.query((face_name + ".drive_amp").c_str(),   params.drive_amp);
    pp.query((face_name + ".drive_omega").c_str(), params.drive_omega);
    pp.query((face_name + ".drive_phase").c_str(), params.drive_phase);

    // Parse relaxation coefficients
    pp.query((face_name + ".relax_u").c_str(), params.relax_u);
    pp.query((face_name + ".relax_v").c_str(), params.relax_v);
    pp.query((face_name + ".relax_w").c_str(), params.relax_w);
    pp.query((face_name + ".relax_T").c_str(), params.relax_T);

    // Parse transverse term weight
    pp.query((face_name + ".beta").c_str(), params.beta);

    // Parse outflow parameters
    pp.query((face_name + ".sigma").c_str(), params.sigma);

    // Parse reference length
    pp.query((face_name + ".L_ref").c_str(), params.L_ref);

    return params;
}


} // namespace BC
