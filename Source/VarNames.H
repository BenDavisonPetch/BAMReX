#ifndef VarNames_H_
#define VarNames_H_

#include <string>

#include <AMReX_BLassert.H>

inline unsigned int var_name_to_idx(const std::string &var_name)
{
    if (var_name == "density")
        return 0;
    else if (var_name == "mom_x")
        return 1;
    else if (var_name == "mom_y")
        return 2;
    else if (var_name == "mom_z")
        return 3;
    else if (var_name == "energy")
        return 1 + AMREX_SPACEDIM;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(false,
                                     "Invalid variable name " + var_name);
    return 0;
}

#endif