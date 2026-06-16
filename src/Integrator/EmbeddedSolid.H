#ifndef INTEGRATOR_EMBEDDEDSOLID_H
#define INTEGRATOR_EMBEDDEDSOLID_H
#include <algorithm>
#include "Set/Base.H"
#include "IC/IC.H"
#include "BC/BC.H"
namespace Integrator {
// Yang (2023) diffuse-domain embedded solid.  Owns the indicator field phi, its
// gradient, the prescribed quiescent target state, the wall parameters, and the
// per-cell wall physics (the (1-phi)/kappa momentum Brinkman penalty + the
// solid-skip test).  Hydro2 holds one instance and drives the field loops.
class EmbeddedSolid {
public:
    // parameters
    int          apply = 0;                  // was apply_embedded_solid
    Set::Scalar  brinkman = 0.0;             // was solid_brinkman (kappa=1/brinkman; <=0 -> auto wall)
    Set::Scalar  relax_skip = 0.0;           // was solid_relax_skip (skip relax/sharpen/CFL where phi<this; =0.5)
    Set::Scalar  refinement_criterion = NAN; // was phi_refinement_criterion
    // indicator + gradient
    Set::Field<Set::Scalar> phi_mf;
    Set::Field<Set::Scalar> phi_old_mf;
    Set::Field<Set::Scalar> grad_phi_mf;
    IC::IC<Set::Scalar>* phi_ic = nullptr;
    BC::BC<Set::Scalar>* phi_bc = nullptr;
    // Prescribed quiescent solid target.  The 6-eqn conservative state is
    // two-phase, so the target fills BOTH phase slots (density0/1, energy0/1) --
    // but these are DERIVED, not input: InitEmbeddedSolidTarget() splits a
    // SINGLE solid density + pressure by the local eta (stiffened-gas EOS for
    // the per-phase energy).  The input is therefore single-phase.
    Set::Field<Set::Scalar> density0_mf, density1_mf, momentum_mf, energy0_mf, energy1_mf;
    IC::IC<Set::Scalar> *density_ic=nullptr, *pressure_ic=nullptr, *momentum_ic=nullptr;

    ~EmbeddedSolid() {
        delete phi_ic; delete phi_bc;
        delete density_ic; delete pressure_ic; delete momentum_ic;
    }

    // per-cell wall physics (inline; called inside Hydro2's loops)
    AMREX_GPU_HOST_DEVICE static Set::Scalar clampPhi(Set::Scalar p) { return std::min(std::max(p, 0.0), 1.0); }
    // true in the geometric solid (relaxation/sharpening/CFL must skip there)
    AMREX_GPU_HOST_DEVICE bool isSolid(Set::Scalar phi) const { return phi < relax_skip; }
    // momentum-only Brinkman penalty source for one velocity component
    AMREX_GPU_HOST_DEVICE Set::Scalar momentumPenalty(Set::Scalar M_d, Set::Scalar Ms_d, Set::Scalar phi_c) const {
        return -brinkman * (1.0 - phi_c) * (M_d - Ms_d);
    }
};
}
#endif
