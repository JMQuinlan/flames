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
    Set::Scalar rho = NAN;
    Set::Scalar M_normal = NAN;
    Set::Scalar M_tangent = NAN;
    Set::Scalar E = NAN;
    Set::Scalar gam = NAN;
    Set::Scalar p0 = NAN;
    Set::Scalar T = NAN;

    // Default constructor
    State()
    {
        rho = 0.0;
        M_normal = 0.0;
        M_tangent = 0.0;
        E = 0.0;
        gam = NAN;
        p0 = NAN;
        T = NAN;
    }

    // Constructor with optional temperature
    State(Set::Scalar a_rho, Set::Scalar a_M_normal, Set::Scalar a_M_tangent, Set::Scalar a_E, Set::Scalar a_gam = NAN, Set::Scalar a_p0 = NAN, Set::Scalar a_T = NAN)
        : rho(a_rho), M_normal(a_M_normal), M_tangent(a_M_tangent), E(a_E), gam(a_gam), p0(a_p0), T(a_T) {}

    // Constructor without temperature_mf
    State(Set::Patch<const Set::Scalar> density_mf, Set::Patch<const Set::Scalar> momentum_mf, Set::Patch<const Set::Scalar> energy_mf, int i, int j, int k, int direction)
    {
        rho = density_mf(i, j, k);
        if (direction == 0)
        {
            M_normal = momentum_mf(i, j, k, 0);
            M_tangent = momentum_mf(i, j, k, 1);
        }
        else if (direction == 1)
        {
            M_normal = momentum_mf(i, j, k, 1);
            M_tangent = momentum_mf(i, j, k, 0);
        }
        else
        {
            Util::Abort(INFO, "Not supported yet");
        }
        E = energy_mf(i, j, k);
        gam = NAN;
        p0 = NAN;
        T = NAN;
    }

    // Original constructor with temperature_mf
    State(Set::Patch<const Set::Scalar> density_mf, Set::Patch<const Set::Scalar> momentum_mf, Set::Patch<const Set::Scalar> energy_mf, Set::Patch<const Set::Scalar> gam_mf, Set::Patch<const Set::Scalar> p0_mf, Set::Patch<const Set::Scalar> temperature_mf, int i, int j, int k, int direction)
    {
        rho = density_mf(i, j, k);
        if (direction == 0)
        {
            M_normal = momentum_mf(i, j, k, 0);
            M_tangent = momentum_mf(i, j, k, 1);
        }
        else if (direction == 1)
        {
            M_normal = momentum_mf(i, j, k, 1);
            M_tangent = momentum_mf(i, j, k, 0);
        }
        else
        {
            Util::Abort(INFO, "Not supported yet");
        }
        E = energy_mf(i, j, k);
        gam = gam_mf(i, j, k);
        p0 = p0_mf(i, j, k);
        T = temperature_mf(i, j, k);
    }

    friend std::ostream &
    operator<<(std::ostream &os, const State &state)
    {
        os << "rho=" << state.rho << ", ";
        os << "Mn=" << state.M_normal << ", ";
        os << "Mt=" << state.M_tangent << ", ";
        os << "E=" << state.E << ", ";
        os << "gam=" << state.gam << ", ";
        os << "p0=" << state.p0 << ", ";
        os << "T=" << state.T << ", ";
        return os;
    }

    void
    operator+=(const State &a)
    {
        rho += a.rho;
        M_normal += a.M_normal;
        M_tangent += a.M_tangent;
        E += a.E;
        gam += a.gam;
        p0 += a.p0;
        T += a.T;
    };

    void
    operator-=(const State &a)
    {
        rho -= a.rho;
        M_normal -= a.M_normal;
        M_tangent -= a.M_tangent;
        E -= a.E;
        gam -= a.gam;
        p0 -= a.p0;
        T -= a.T;
    };

    void
    operator*=(const Set::Scalar alpha)
    {
        rho *= alpha;
        M_normal *= alpha;
        M_tangent *= alpha;
        E *= alpha;
        gam *= alpha;
        p0 *= alpha;
        T *= alpha;
    };

    void
    operator/=(const Set::Scalar alpha)
    {
        rho /= alpha;
        M_normal /= alpha;
        M_tangent /= alpha;
        E /= alpha;
        gam /= alpha;
        p0 /= alpha;
        T /= alpha;
    };

    friend State
    operator+(const State &a, const State &b)
    {
        return State(a.rho + b.rho, a.M_normal + b.M_normal, a.M_tangent + b.M_tangent, a.E + b.E, a.gam + b.gam, a.p0 + b.p0, a.T + b.T);
    }

    friend State
    operator-(const State &a, const State &b)
    {
        return State(a.rho - b.rho, a.M_normal - b.M_normal, a.M_tangent - b.M_tangent, a.E - b.E, a.gam - b.gam, a.p0 - b.p0, a.T - b.T);
    }

    friend State
    operator*(const Set::Scalar alpha, const State &b)
    {
        return State(b.rho * alpha, b.M_normal * alpha, b.M_tangent * alpha, b.E * alpha, b.gam * alpha, b.p0 * alpha, b.T * alpha);
    }

    friend State
    operator*(const State &b, const Set::Scalar alpha)
    {
        return State(b.rho * alpha, b.M_normal * alpha, b.M_tangent * alpha, b.E * alpha, b.gam * alpha, b.p0 * alpha, b.T * alpha);
    }

    friend State
    operator/(const State &b, const Set::Scalar alpha)
    {
        return State(b.rho / alpha, b.M_normal / alpha, b.M_tangent / alpha, b.E / alpha, b.gam / alpha, b.p0 / alpha, b.T / alpha);
    }
};

struct Flux
{
    Set::Scalar mass;
    Set::Scalar momentum_normal;
    Set::Scalar momentum_tangent;
    Set::Scalar energy;
    Set::Scalar Gam;
    Set::Scalar P0;
    Set::Scalar temperature;
    Set::Scalar u_interface; // NEW: HLLC interface velocity

    Flux() : mass(0.0), momentum_normal(0.0), momentum_tangent(0.0), energy(0.0), Gam(0.0), P0(0.0), temperature(0.0), u_interface(0.0) {}

    // Constructor without u_interface (for backward compatibility)
    Flux(Set::Scalar a_mass, Set::Scalar a_momentum_normal, Set::Scalar a_momentum_tangent, Set::Scalar a_energy, Set::Scalar a_Gam = NAN, Set::Scalar a_P0 = NAN, Set::Scalar a_temperature = NAN)
        : mass(a_mass), momentum_normal(a_momentum_normal),
          momentum_tangent(a_momentum_tangent), energy(a_energy), Gam(a_Gam), P0(a_P0), temperature(a_temperature), u_interface(0.0) {}

    // Constructor with u_interface
    Flux(Set::Scalar a_mass, Set::Scalar a_momentum_normal, Set::Scalar a_momentum_tangent, Set::Scalar a_energy, Set::Scalar a_Gam, Set::Scalar a_P0, Set::Scalar a_temperature, Set::Scalar a_u_interface)
        : mass(a_mass), momentum_normal(a_momentum_normal),
          momentum_tangent(a_momentum_tangent), energy(a_energy),
          Gam(a_Gam), P0(a_P0), temperature(a_temperature),
          u_interface(a_u_interface) {}

    Flux(const State &in) : mass(in.rho),
                            momentum_normal(in.M_normal),
                            momentum_tangent(in.M_tangent),
                            energy(in.E),
                            Gam(in.gam),
                            P0(in.p0),
                            temperature(in.T),
                            u_interface(0.0) {}

    friend std::ostream &
    operator<<(std::ostream &os, const Flux &flux)
    {
        os << "mass=" << flux.mass << ", ";
        os << "Mn=" << flux.momentum_normal << ", ";
        os << "Mt=" << flux.momentum_tangent << ", ";
        os << "E=" << flux.energy << ", ";
        os << "gam=" << flux.Gam << ", ";
        os << "p0=" << flux.P0 << ", ";
        os << "T=" << flux.temperature << ", ";
        os << "u_int=" << flux.u_interface; // NEW
        return os;
    }

    void
    operator+=(const Flux &a)
    {
        mass += a.mass;
        momentum_normal += a.momentum_normal;
        momentum_tangent += a.momentum_tangent;
        energy += a.energy;
        Gam += a.Gam;
        P0 += a.P0;
        temperature += a.temperature;
        u_interface += a.u_interface; // NEW
    }

    void
    operator-=(const Flux &a)
    {
        mass -= a.mass;
        momentum_normal -= a.momentum_normal;
        momentum_tangent -= a.momentum_tangent;
        energy -= a.energy;
        Gam -= a.Gam;
        P0 -= a.P0;
        temperature -= a.temperature;
        u_interface -= a.u_interface; // NEW
    }

    void
    operator*=(const Set::Scalar alpha)
    {
        mass *= alpha;
        momentum_normal *= alpha;
        momentum_tangent *= alpha;
        energy *= alpha;
        Gam *= alpha; // FIXED (was Gam *= Gam)
        P0 *= alpha;  // FIXED (was P0 *= P0)
        temperature *= alpha;
        u_interface *= alpha; // NEW
    }

    void
    operator/=(const Set::Scalar alpha)
    {
        mass /= alpha;
        momentum_normal /= alpha;
        momentum_tangent /= alpha;
        energy /= alpha;
        Gam /= alpha;
        P0 /= alpha;
        temperature /= alpha;
        u_interface /= alpha; // NEW
    }

    friend Flux
    operator+(const Flux &a, const Flux &b)
    {
        return Flux(a.mass + b.mass, a.momentum_normal + b.momentum_normal, a.momentum_tangent + b.momentum_tangent, a.energy + b.energy, a.Gam + b.Gam, a.P0 + b.P0, a.temperature + b.temperature,
                    a.u_interface + b.u_interface); // NEW
    }

    friend Flux
    operator-(const Flux &a, const Flux &b)
    {
        return Flux(a.mass - b.mass, a.momentum_normal - b.momentum_normal, a.momentum_tangent - b.momentum_tangent, a.energy - b.energy, a.Gam - b.Gam, a.P0 - b.P0, a.temperature - b.temperature,
                    a.u_interface - b.u_interface); // NEW
    }

    friend Flux
    operator*(const Flux &a, const Set::Scalar beta)
    {
        return Flux(a.mass * beta, a.momentum_normal * beta, a.momentum_tangent * beta, a.energy * beta, a.Gam * beta, a.P0 * beta, a.temperature * beta, a.u_interface * beta); // NEW
    }

    friend Flux
    operator*(const Set::Scalar beta, const Flux &a)
    {
        return Flux(a.mass * beta, a.momentum_normal * beta, a.momentum_tangent * beta, a.energy * beta, a.Gam * beta, a.P0 * beta, a.temperature * beta, a.u_interface * beta); // NEW
    }

    friend Flux
    operator/(const Flux &a, const Set::Scalar beta)
    {
        return Flux(a.mass / beta, a.momentum_normal / beta, a.momentum_tangent / beta, a.energy / beta, a.Gam / beta, a.P0 / beta, a.temperature / beta, a.u_interface / beta); // NEW
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
