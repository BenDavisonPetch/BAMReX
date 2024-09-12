#ifndef VarNames_H_
#define VarNames_H_

#include "Index.H"
#include <AMReX_BLassert.H>
#include <string>

inline unsigned int var_name_to_idx(const std::string &var_name)
{
    if (var_name == "density")
        return UInd::RHO;
    else if (var_name == "mom_x")
        return UInd::MX;
    else if (var_name == "mom_y")
        return UInd::MY;
    else if (var_name == "mom_z")
        return UInd::MZ;
    else if (var_name == "energy")
        return UInd::E;
    else if (var_name == "u_x")
        return QInd::U;
    else if (var_name == "u_y")
        return QInd::V;
    else if (var_name == "u_z")
        return QInd::W;
    else if (var_name == "p")
        return QInd::P;
    else if (var_name == "T")
        return QInd::T;
    else if (var_name == "eta")
        return QInd::ETA;
    else if (var_name == "zeta")
        return QInd::ZETA;
    else if (var_name == "kappa")
        return QInd::KAPPA;
    amrex::Abort("Invalid variable name " + var_name);
    return 0;
}

inline unsigned int var_is_primv(const std::string &var_name)
{
    if (var_name == "density" || var_name == "mom_x" || var_name == "mom_y"
        || var_name == "mom_z" || var_name == "energy")
        return false;
    else if (var_name == "u_x" || var_name == "u_y" || var_name == "u_z"
             || var_name == "p" || var_name == "T" || var_name == "eta"
             || var_name == "zeta" || var_name == "kappa")
        return true;
    else
    {
        amrex::Abort("Invalid variable name " + var_name);
        return true;
    }
}

#endif