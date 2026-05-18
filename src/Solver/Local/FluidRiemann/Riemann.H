#ifndef SOLVER_LOCAL_FLUIDRIEMANN_H
#define SOLVER_LOCAL_FLUIDRIEMANN_H

#include "Set/Base.H"
#include "Set/Set.H"

namespace Solver
{
namespace Local
{
namespace FluidRiemann
{

struct State
{
    // --------------------------------------------------------------
    // Conservative variables
    // --------------------------------------------------------------
    amrex::Array4<const Set::Scalar> rho_arr;
    amrex::Array4<const Set::Scalar> M_arr;      // momentum has 2 comps
    amrex::Array4<const Set::Scalar> E_arr;

    // Thermodynamic fields
    amrex::Array4<const Set::Scalar> gamma_arr;
    amrex::Array4<const Set::Scalar> pi_arr;   // Tammann EOS stiffness (pi = pi)
    amrex::Array4<const Set::Scalar> T_arr;

    // Coordinates and direction
    int i, j, k;
    int dir;   // 0 = X normal, 1 = Y normal

    // --------------------------------------------------------------
    // Stored primitive/conservative scalars
    // --------------------------------------------------------------
    Set::Scalar rho;
    Set::Scalar M_normal;
    Set::Scalar M_tangent;
    Set::Scalar E;
    Set::Scalar gam;
    Set::Scalar pi;
    Set::Scalar T;

    // Volume fraction of the phase this State represents.
    // Default 1.0: rho/M/E are interpreted as single-phase (intrinsic
    // == partial). For two-phase callers, set alpha = α_k_face after
    // construction; rho/M/E are then interpreted as partial densities
    // (α_k · intrinsic) and the EOS recovers intrinsic via /alpha.
    Set::Scalar alpha = Set::Scalar(1.0);

    // --------------------------------------------------------------
    // Default
    // --------------------------------------------------------------
    AMREX_GPU_HOST_DEVICE
    State()
        : rho(0), M_normal(0), M_tangent(0), E(0),
          gam(NAN), pi(NAN), T(NAN),
          i(0), j(0), k(0), dir(0)
    {}


    // ===============================================================
    // CPU‑only constructor for Riemann test problems
    // ===============================================================
    #ifdef AMREX_USE_GPU
    AMREX_CPU_HOST    // ensures it's HOST‑only when GPU is enabled
    #endif
    State(Set::Scalar a_rho,
        Set::Scalar a_Mn,
        Set::Scalar a_Mt,
        Set::Scalar a_E,
        Set::Scalar a_gam,
        Set::Scalar a_pi,
        Set::Scalar a_T)
    {
        rho       = a_rho;
        M_normal  = a_Mn;
        M_tangent = a_Mt;
        E         = a_E;
        gam       = a_gam;
        pi        = a_pi;
        T         = a_T;

        // Dummy indices for CPU test constructor
        i = j = k = 0;
        dir = 0;
    }


    // --------------------------------------------------------------
    // Main constructor: Array4 everywhere (CORRECT)
    // --------------------------------------------------------------
    AMREX_GPU_HOST_DEVICE
    State(const amrex::Array4<const Set::Scalar>& rhoA,
          const amrex::Array4<const Set::Scalar>& MA,
          const amrex::Array4<const Set::Scalar>& EA,
          const amrex::Array4<const Set::Scalar>& gA,
          const amrex::Array4<const Set::Scalar>& piA,
          const amrex::Array4<const Set::Scalar>& TA,
          int ii, int jj, int kk, int idir)
        :
          rho_arr(rhoA),
          M_arr(MA),
          E_arr(EA),
          gamma_arr(gA),
          pi_arr(piA),
          T_arr(TA),
          i(ii), j(jj), k(kk),
          dir(idir)
    {
        load_state();
    }

    // --------------------------------------------------------------
    // GPU-capable constructor with SCALAR gamma/pi (single-phase EOS).
    // Used by the per-phase Euler path where each phase carries a
    // constant gamma, pi and the Array4 indirection is unnecessary.
    // --------------------------------------------------------------
    AMREX_GPU_HOST_DEVICE
    State(const amrex::Array4<const Set::Scalar>& rhoA,
          const amrex::Array4<const Set::Scalar>& MA,
          const amrex::Array4<const Set::Scalar>& EA,
          Set::Scalar a_gam,
          Set::Scalar a_pi,
          int ii, int jj, int kk, int idir)
        :
          rho_arr(rhoA),
          M_arr(MA),
          E_arr(EA),
          // gamma_arr / pi_arr / T_arr left default-constructed; never read
          i(ii), j(jj), k(kk),
          dir(idir)
    {
        rho = rho_arr(i,j,k);
        E   = E_arr(i,j,k);
        gam = a_gam;
        pi  = a_pi;
        T   = Set::Scalar(0.0);

        Set::Scalar mx = M_arr(i,j,k,0);
        Set::Scalar my = M_arr(i,j,k,1);
        if (dir == 0) { M_normal = mx; M_tangent = my; }
        else          { M_normal = my; M_tangent = mx; }
    }

    // --------------------------------------------------------------
    // Load fields from the Array4s
    // --------------------------------------------------------------
    AMREX_GPU_HOST_DEVICE
    void load_state()
    {
        rho = rho_arr(i,j,k);
        E   = E_arr(i,j,k);

        gam = gamma_arr(i,j,k);
        pi  = pi_arr(i,j,k);
        T   = T_arr(i,j,k);

        // Momentum components
        Set::Scalar mx = M_arr(i,j,k,0);
        Set::Scalar my = M_arr(i,j,k,1);

        // Select normal/tangent directions
        if (dir == 0)        // X-normal face
        {
            M_normal  = mx;
            M_tangent = my;
        }
        else                 // Y-normal face
        {
            M_normal  = my;
            M_tangent = mx;
        }
    }

    // --------------------------------------------------------------
    // Printing utilities
    // --------------------------------------------------------------
    friend std::ostream&
    operator<<(std::ostream& os, const State& s)
    {
        os << "rho=" << s.rho << ", "
           << "Mn="  << s.M_normal << ", "
           << "Mt="  << s.M_tangent << ", "
           << "E="   << s.E << ", "
           << "gam=" << s.gam << ", "
           << "pi="  << s.pi << ", "
           << "T="   << s.T  << ", "
           << "alpha=" << s.alpha;
        return os;
    }

    // --------------------------------------------------------------
    // Vector arithmetic keeps working unchanged
    // --------------------------------------------------------------
    AMREX_GPU_HOST_DEVICE
    void operator+=(const State& b)
    {
        rho += b.rho;
        M_normal += b.M_normal;
        M_tangent += b.M_tangent;
        E   += b.E;
        gam += b.gam;
        pi  += b.pi;
        T   += b.T;
        alpha += b.alpha;
    }

    AMREX_GPU_HOST_DEVICE
    void operator-=(const State& b)
    {
        rho -= b.rho;
        M_normal -= b.M_normal;
        M_tangent -= b.M_tangent;
        E   -= b.E;
        gam -= b.gam;
        pi  -= b.pi;
        T   -= b.T;
        alpha -= b.alpha;
    }

    AMREX_GPU_HOST_DEVICE
    void operator*=(Set::Scalar a)
    {
        rho *= a;
        M_normal *= a;
        M_tangent *= a;
        E *= a;
        gam *= a;
        pi *= a;
        T *= a;
        alpha *= a;
    }

    AMREX_GPU_HOST_DEVICE
    void operator/=(Set::Scalar a)
    {
        rho /= a;
        M_normal /= a;
        M_tangent /= a;
        E /= a;
        gam /= a;
        pi /= a;
        T /= a;
        alpha /= a;
    }
};

struct Flux
{
    Set::Scalar mass;
    Set::Scalar momentum_normal;
    Set::Scalar momentum_tangent;
    Set::Scalar energy;

    // Auxiliary thermodynamic quantities (not conserved)
    Set::Scalar gam;
    Set::Scalar pi;
    Set::Scalar T;

    // ----------------------------------------------------------
    // Default constructor
    // ----------------------------------------------------------
    AMREX_GPU_HOST_DEVICE
    Flux()
        : mass(0.0), momentum_normal(0.0), momentum_tangent(0.0),
          energy(0.0), gam(NAN), pi(NAN), T(NAN)
    {}

    // ----------------------------------------------------------
    // Constructor from components
    // ----------------------------------------------------------
    AMREX_GPU_HOST_DEVICE
    Flux(Set::Scalar m, Set::Scalar Mn, Set::Scalar Mt, Set::Scalar E,
         Set::Scalar g = NAN, Set::Scalar pi_ = NAN, Set::Scalar T_ = NAN)
        : mass(m), momentum_normal(Mn), momentum_tangent(Mt), energy(E),
          gam(g), pi(pi_), T(T_)
    {}

    // ----------------------------------------------------------
    // Construct from a State
    // ----------------------------------------------------------
    AMREX_GPU_HOST_DEVICE
    Flux(const State& s)
        : mass(s.rho),
          momentum_normal(s.M_normal),
          momentum_tangent(s.M_tangent),
          energy(s.E),
          gam(s.gam),
          pi(s.pi),
          T(s.T)
    {}

    // ----------------------------------------------------------
    // Printing (debugging only)
    // ----------------------------------------------------------
    friend std::ostream& operator<<(std::ostream& os, const Flux& f)
    {
        os << "mass=" << f.mass << ", "
           << "Mn="   << f.momentum_normal << ", "
           << "Mt="   << f.momentum_tangent << ", "
           << "E="    << f.energy << ", "
           << "gam="  << f.gam << ", "
           << "pi="   << f.pi << ", "
           << "T="    << f.T;
        return os;
    }

    // ----------------------------------------------------------
    // +=  (conservative)
    // ----------------------------------------------------------
    AMREX_GPU_HOST_DEVICE
    void operator+=(const Flux& b)
    {
        mass             += b.mass;
        momentum_normal  += b.momentum_normal;
        momentum_tangent += b.momentum_tangent;
        energy           += b.energy;

        // Non-conserved values: best to forward RHS source terms
        gam = b.gam;
        pi = b.pi;
        T   = b.T;
    }

    // ----------------------------------------------------------
    // -=
    // ----------------------------------------------------------
    AMREX_GPU_HOST_DEVICE
    void operator-=(const Flux& b)
    {
        mass             -= b.mass;
        momentum_normal  -= b.momentum_normal;
        momentum_tangent -= b.momentum_tangent;
        energy           -= b.energy;

        gam = b.gam;
        pi  = b.pi;
        T   = b.T;
    }

    // ----------------------------------------------------------
    // *= scalar
    // ----------------------------------------------------------
    AMREX_GPU_HOST_DEVICE
    void operator*=(Set::Scalar a)
    {
        mass             *= a;
        momentum_normal  *= a;
        momentum_tangent *= a;
        energy           *= a;

        // pass-through
        gam *= 1;
        pi  *= 1;
        T   *= 1;
    }

    // ----------------------------------------------------------
    // /= scalar
    // ----------------------------------------------------------
    AMREX_GPU_HOST_DEVICE
    void operator/=(Set::Scalar a)
    {
        mass             /= a;
        momentum_normal  /= a;
        momentum_tangent /= a;
        energy           /= a;

        gam *= 1;
        pi  *= 1;
        T   *= 1;
    }

    // ----------------------------------------------------------
    // Binary ops
    // ----------------------------------------------------------
    friend AMREX_GPU_HOST_DEVICE
    Flux operator+(Flux a, const Flux& b)
    { a += b; return a; }

    friend AMREX_GPU_HOST_DEVICE
    Flux operator-(Flux a, const Flux& b)
   { a -= b; return a; }

    friend AMREX_GPU_HOST_DEVICE
    Flux operator*(const Flux& a, Set::Scalar b)
    {
        Flux f = a;
        f.mass             *= b;
        f.momentum_normal  *= b;
        f.momentum_tangent *= b;
        f.energy           *= b;
        return f;
    }

    friend AMREX_GPU_HOST_DEVICE
    Flux operator*(Set::Scalar b, const Flux& a)
    { return a * b; }

    friend AMREX_GPU_HOST_DEVICE
    Flux operator/(const Flux& a, Set::Scalar b)
    {
        Flux f = a;
        f.mass             /= b;
        f.momentum_normal  /= b;
        f.momentum_tangent /= b;
        f.energy           /= b;
        return f;
    }
};

// --------------------------------------------------------------
// Korteweg face data — optional input to Riemann solvers that
// support flux-form capillary stress. Caller passes nullptr to
// disable. eta_face / dn_eta / dt_eta / lap_eta are face-evaluated
// (caller chooses stencil); kappa is the gradient-energy
// coefficient already weighted by the per-phase mass fraction
// (kappa = Y_k_face · κ_global). The two phases' Korteweg flux
// contributions then sum to the full Σ_K flux on the mixture.
// --------------------------------------------------------------
struct KortewegFaceData
{
    Set::Scalar eta_face;   // η at the face
    Set::Scalar dn_eta;     // ∂_n η  (face-normal direction)
    Set::Scalar dt_eta;     // ∂_t η  (face-tangential direction)
    Set::Scalar lap_eta;    // ∇²η at the face
    Set::Scalar kappa;      // gradient-energy coefficient × phase weight
    // Bulk-thermo trace correction (well-balanced compensation):
    //   Σ_K_wb_nn = Σ_K_nn − q_eta.
    // Caller computes q_eta = Y_k_face · λ·[ηW'(η) − W(η)] at the face;
    // the per-phase mass-fraction weight Y_k goes here too so the two
    // phases' contributions sum to the full mixture trace correction.
    // Default 0 ⇒ no correction (legacy / pure-gradient-stress mode).
    Set::Scalar q_eta = Set::Scalar(0.0);
    // When set to 1, the Riemann solver modifies momentum but NOT energy.
    // The caller is responsible for the Korteweg energy coupling (e.g.
    // via the non-conservative form (div Σ_K)·u + Σ_K:∇u at cell
    // centers, which avoids the S_star amplification of face residuals).
    int skip_energy = 0;
};

class FluidRiemann
{
public:
    // Legacy single-phase Riemann solve. Each derived solver overrides.
    virtual Flux Solve(State lo, State hi, Set::Scalar p_ref, Set::Scalar small, int Spec_Vol) = 0;

    // Extended solve: passive scalars (advect with the contact wave) and
    // optional Korteweg face data (added to momentum/energy flux as a
    // flux-form capillary stress). Default implementation in the base
    // class dispatches to legacy Solve, zeros passive fluxes, and ignores
    // kfd — derived solvers that support the new path override this.
    //
    //   n_passive : number of passive conservative scalars at this face
    //   q_lo, q_hi: [n_passive] L/R conservative values (e.g., α_k ρ_k η)
    //   q_flux    : [n_passive] output flux array (caller-allocated)
    //   kfd       : nullptr ⇒ no Korteweg contribution
    virtual Flux Solve(State lo, State hi, Set::Scalar p_ref, Set::Scalar small,
                       int n_passive,
                       Set::Scalar const* /*q_lo*/, Set::Scalar const* /*q_hi*/,
                       Set::Scalar* q_flux,
                       KortewegFaceData const* /*kfd*/,
                       int Spec_Vol)
    {
        for (int s = 0; s < n_passive; ++s) q_flux[s] = Set::Scalar(0.0);
        return Solve(lo, hi, p_ref, small, Spec_Vol);
    }

    virtual ~FluidRiemann() = default;
};

}
}
}

#endif
