#ifndef NumericalMethods_H_
#define NumericalMethods_H_

#include <AMReX.H>

#include <string>

namespace NumericalMethods
{
enum Method
{
    muscl_hancock,
    hllc,
    imex,
    rcm,
    slic,
    force,
    lax_friedrichs
};

inline Method enum_from_string(const std::string &name)
{
    if (name == "muscl_hancock")
        return muscl_hancock;
    else if (name == "hllc")
        return hllc;
    else if (name == "imex")
        return imex;
    else if (name == "rcm")
        return rcm;
    else if (name == "slic")
        return slic;
    else if (name == "force")
        return force;
    else if (name == "lax_friedrichs")
        return lax_friedrichs;
    else
    {
        amrex::Abort("Invalid numerical method \"" + name
                     + "\". Valid options are "
                       "muscl_hancock, hllc, and imex");
        return muscl_hancock;
    }
}
}

//! Methods for handling viscous terms
namespace ViscMethods
{
enum Method
{
    //! No viscous terms
    none,
    //! Via operator splitting, done explicitly, no time subcycling (assumes
    //! dt_diff > dt_conv)
    op_split_explicit
};

inline Method enum_from_string(const std::string &name)
{
    if (name == "none")
        return none;
    else if (name == "op_split_explicit")
        return op_split_explicit;
    else
    {
        amrex::Abort("Invalid viscosity method \"" + name
                     + "\". Valid options are none and op_split_explicit.");
        return none;
    }
}
}

#endif