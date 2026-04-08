#include "BC/NSCBC.H"
#include "Util/Util.H"
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

// SEE THIS PAPER: 
// https://hal.science/hal-01699049/file/2018_AIAA_CFD_Motheau.pdf
// Motheau, et. al (2018)
// 
// Implementation of the Navier-Stokes Characteristic Boundry Condition (NSCBC)
//

namespace BC
{

// ============================================================================
// CONSTRUCTOR AND PARSING
// ============================================================================

NSCBC::NSCBC(IO::ParmParse &pp, std::string prefix)
{
    Parse(*this, pp);
}

void NSCBC::Parse(NSCBC &value, IO::ParmParse &pp)
{
    // Parse boundary parameters
    value.params_xlo = ParseFace(pp, "nscbc.xlo");
    value.params_xhi = ParseFace(pp, "nscbc.xhi");
    value.params_ylo = ParseFace(pp, "nscbc.ylo");
    value.params_yhi = ParseFace(pp, "nscbc.yhi");
#if AMREX_SPACEDIM == 3
    value.params_zlo = ParseFace(pp, "nscbc.zlo");
    value.params_zhi = ParseFace(pp, "nscbc.zhi");
#endif

    pp.query("nscbc.small", value.small);
}

NSCBC::BoundaryParams NSCBC::ParseFace(IO::ParmParse &pp, std::string face_name)
{
    BoundaryParams params;

    // Parse type
    std::string type_str;
    pp.query((face_name + ".type").c_str(), type_str);

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

    // Parse target values
    pp.query((face_name + ".target_u").c_str(), params.target_u);
    pp.query((face_name + ".target_v").c_str(), params.target_v);
    pp.query((face_name + ".target_w").c_str(), params.target_w);
    pp.query((face_name + ".target_T").c_str(), params.target_T);
    pp.query((face_name + ".target_p").c_str(), params.target_p);

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
