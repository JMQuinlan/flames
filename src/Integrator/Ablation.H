#ifndef INTEGRATOR_ABLATION_H
#define INTEGRATOR_ABLATION_H
#include "Set/Base.H"
#include "BC/BC.H"
namespace Integrator {
// SURFACE ABLATION for Hydro2 -- default OFF (ablation.on = 0).
//
// Heat-of-ablation ("Q*") model, the standard engineering closure for a receding
// thermal-protection surface: the wall is held at the ablation temperature T_w and
// the heat conducted into it from the gas is consumed by removing solid,
//     rho_s Q* v_abl = q_w,      q_w = k dT/dn |gas side      (q_w > 0 only),
// Q* = effective heat of ablation (sensible + latent) per unit mass of solid.
// In this first version NO BLOWING: the ablated mass is removed, not injected into the gas.
//
// [2026-10-06] IN-DEPTH CONDUCTION (ablation.conduction = 1, default 0 = the model above,
// bit-identical).  The solid carries its own temperature T_s (field "ablation_Ts"),
//     rho_s cp_s dT_s/dt = div(k_s grad T_s),
// advanced explicitly once per step.  At every fluid|solid face the surface temperature
// follows from flux continuity between the two one-sided gradients,
//     T_w* = (h_g T_g + h_s T_s) / (h_g + h_s),    h = k / (distance cell centre - surface),
// and T_wall becomes the ABLATION temperature: below it nothing is removed (q_g = q_s);
// once T_w* reaches it the surface is held at T_wall and the excess removes solid,
//     rho_s Q* v_abl = q_g - q_s,   q_g = h_g (T_g - T_wall),  q_s = h_s (T_wall - T_s).
// (Q* is then the heat of ablation proper; the sensible heat goes into T_s.)
// Exact checks: two half-spaces in contact (T_w = (e_g T_g + e_s T_s)/(e_g + e_s),
// e = sqrt(k rho c)) and the two-phase Stefan problem (tests/FlowAblation/input_1D_Contact,
// input_1D_Stefan2).
//
// NON-DEFORMING (embedded, solid.*) bodies: the indicator phi (1 fluid, 0 solid)
// recedes by the level-set equation  d(phi)/dt = v_abl |grad phi|, with v_abl
// extended off the wall cells; cells uncovered by the surface are filled exactly
// as for solid.moving.  Needs the sharp wall (solid.wall_flux = 1), heat
// conduction in the gas (thermal.conduction = 1) and the isothermal wall
// (solid.T_wall, set here to ablation.T_wall).
//
// Exact check: gas half-space at T_inf against a wall at T_w (no flow, constant
// properties): surface position s = 2 lam sqrt(alpha t) with
//     lam exp(lam^2) (1 + erf lam) = rho cp (T_inf - T_w) / (sqrt(pi) rho_s Q*)
// (one-phase Stefan similarity solution; tests/FlowAblation/input_1D_Stefan).
class Ablation {
public:
    int          on = 0;
    Set::Scalar  T_wall = -1.0;     // wall (ablation) temperature
    Set::Scalar  Qstar  = -1.0;     // effective heat of ablation per unit mass of solid
    Set::Scalar  rho_s  = -1.0;     // solid density
    Set::Scalar  accel  = 1.0;      // recession speed multiplier (ablation is slow against the flow)
    int          extend = 8;        // passes extending v_abl off the wall cells
    Set::Scalar  t_start = 0.0;     // no recession before this time (lets the flow establish)
    Set::Field<Set::Scalar> q_mf;   // heat flux into the wall per unit TRUE surface area, on wall-adjacent fluid cells
    Set::Field<Set::Scalar> v_mf;   // extended recession speed (plot / scratch)
    // in-depth conduction in the solid (ablation.conduction)
    int          conduction = 0;
    Set::Scalar  k_s  = -1.0;       // solid thermal conductivity
    Set::Scalar  cp_s = -1.0;       // solid specific heat
    Set::Scalar  T_solid0 = -1.0;   // initial solid temperature (default T_wall)
    Set::Field<Set::Scalar> Ts_mf;  // solid temperature (meaningful where phi < 0.5)
    BC::BC<Set::Scalar> *Ts_bc = nullptr;

    // One fluid|solid face: surface temperature from flux continuity, capped at the ablation
    // temperature.  hg, hs = k / distance on each side; Tg, Ts the one-sided reference values.
    static AMREX_FORCE_INLINE Set::Scalar WallTemperature(Set::Scalar hg, Set::Scalar Tg, Set::Scalar hs, Set::Scalar Ts, Set::Scalar Tabl)
    {
        return std::min((hg * Tg + hs * Ts) / (hg + hs), Tabl);
    }
};
}
#endif
