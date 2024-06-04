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
    rcm
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
    else
    {
        amrex::Abort("Invalid numerical method! Valid options are "
                     "muscl_hancock, hllc, and imex");
        return muscl_hancock;
    }
}
}

#endif