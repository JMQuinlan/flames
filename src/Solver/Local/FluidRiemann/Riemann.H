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
    amrex::Array4<const Set::Scalar> Tamann_p_0_arr;
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
    Set::Scalar Tamann_p_0;
    Set::Scalar T;

    // --------------------------------------------------------------
    // Default
    // --------------------------------------------------------------
    AMREX_GPU_HOST_DEVICE
    State()
        : rho(0), M_normal(0), M_tangent(0), E(0),
          gam(NAN), Tamann_p_0(NAN), T(NAN),
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
        Set::Scalar a_Tamann_p_0,
        Set::Scalar a_T)
    {
        rho       = a_rho;
        M_normal  = a_Mn;
        M_tangent = a_Mt;
        E         = a_E;
        gam       = a_gam;
        Tamann_p_0        = a_Tamann_p_0;
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
          const amrex::Array4<const Set::Scalar>& Tamann_p_0A,
          const amrex::Array4<const Set::Scalar>& TA,
          int ii, int jj, int kk, int idir)
        :
          rho_arr(rhoA),
          M_arr(MA),
          E_arr(EA),
          gamma_arr(gA),
          Tamann_p_0_arr(Tamann_p_0A),
          T_arr(TA),
          i(ii), j(jj), k(kk),
          dir(idir)
    {
        load_state();
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
        Tamann_p_0  = Tamann_p_0_arr(i,j,k);
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
           << "Tamann_p_0="  << s.Tamann_p_0 << ", "
           << "T="   << s.T;
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
        Tamann_p_0  += b.Tamann_p_0;
        T   += b.T;
    }

    AMREX_GPU_HOST_DEVICE
    void operator-=(const State& b)
    {
        rho -= b.rho;
        M_normal -= b.M_normal;
        M_tangent -= b.M_tangent;
        E   -= b.E;
        gam -= b.gam;
        Tamann_p_0  -= b.Tamann_p_0;
        T   -= b.T;
    }

    AMREX_GPU_HOST_DEVICE
    void operator*=(Set::Scalar a)
    {
        rho *= a;
        M_normal *= a;
        M_tangent *= a;
        E *= a;
        gam *= a;
        Tamann_p_0 *= a;
        T *= a;
    }

    AMREX_GPU_HOST_DEVICE
    void operator/=(Set::Scalar a)
    {
        rho /= a;
        M_normal /= a;
        M_tangent /= a;
        E /= a;
        gam /= a;
        Tamann_p_0 /= a;
        T /= a;
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
    Set::Scalar Tamann_p_0;
    Set::Scalar T;

    // ----------------------------------------------------------
    // Default constructor
    // ----------------------------------------------------------
    AMREX_GPU_HOST_DEVICE
    Flux()
        : mass(0.0), momentum_normal(0.0), momentum_tangent(0.0),
          energy(0.0), gam(NAN), Tamann_p_0(NAN), T(NAN)
    {}

    // ----------------------------------------------------------
    // Constructor from components
    // ----------------------------------------------------------
    AMREX_GPU_HOST_DEVICE
    Flux(Set::Scalar m, Set::Scalar Mn, Set::Scalar Mt, Set::Scalar E,
         Set::Scalar g = NAN, Set::Scalar Tamann_p_0_ = NAN, Set::Scalar T_ = NAN)
        : mass(m), momentum_normal(Mn), momentum_tangent(Mt), energy(E),
          gam(g), Tamann_p_0(Tamann_p_0_), T(T_)
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
          Tamann_p_0(s.Tamann_p_0),
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
           << "Tamann_p_0="   << f.Tamann_p_0 << ", "
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
        Tamann_p_0  = b.Tamann_p_0;
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
        Tamann_p_0  = b.Tamann_p_0;
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
        Tamann_p_0  *= 1;
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
        Tamann_p_0  *= 1;
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

class FluidRiemann
{
public:
    virtual Flux Solve(State lo, State hi, Set::Scalar p_ref, Set::Scalar small, int Spec_Vol) = 0;
    virtual ~FluidRiemann() = default;
};

}
}
}

#endif
