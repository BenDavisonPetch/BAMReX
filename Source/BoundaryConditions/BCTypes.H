#ifndef BCTypes_H_
#define BCTypes_H_

#include <AMReX.H>
#include <string>

/**
 * BC type definitions
 */
namespace BAMReXBCType
{
enum bctype
{
    transmissive,
    reflective,
    periodic,
    gresho,
    pressure_outlet,
    coaxial_jet,
    supersonic_jet
};

/**
 * Helper functions
 */
inline bctype enum_from_string(const std::string &name)
{
    if (name == "transmissive")
        return transmissive;
    else if (name == "reflective")
        return reflective;
    else if (name == "periodic")
        return periodic;
    else if (name == "gresho")
        return gresho;
    else if (name == "pressure_outlet")
        return pressure_outlet;
    else if (name == "coaxial_jet")
        return coaxial_jet;
    else if (name == "supersonic_jet")
        return supersonic_jet;
    else
    {
        amrex::Abort("Invalid BC type specified");
        return transmissive;
    }
}
}

namespace RigidBodyBCType
{
enum rigidbodybctype
{
    reflective,
    no_slip,
    foextrap
};

inline rigidbodybctype enum_from_string(const std::string &name)
{
    if (name == "reflective")
        return reflective;
    else if (name == "no_slip")
        return no_slip;
    else
    {
        amrex::Abort("Invalid rigid body BC type specified");
        return reflective;
    }
}
}

#endif