#ifndef IndexMacros_H_
#define IndexMacros_H_

#include <AMReX.H>

//! Conservative indices
struct UInd
{
    static constexpr int RHO = 0;
    static constexpr int MX  = 1;
    static constexpr int MY  = 2;
    static constexpr int MZ  = 3;
    static constexpr int E   = (1 + AMREX_SPACEDIM);

    static constexpr int size = E + 1;
};

//! Primitive indices
struct QInd
{
    static constexpr int RHO = 0;
    static constexpr int U   = 1;
    static constexpr int V   = 2;
    static constexpr int W   = 3;
    static constexpr int P   = (1 + AMREX_SPACEDIM);
    //! Extra parts for viscous computations
    static constexpr int T     = (2 + AMREX_SPACEDIM);
    static constexpr int ETA   = (3 + AMREX_SPACEDIM);
    static constexpr int ZETA  = (4 + AMREX_SPACEDIM);
    static constexpr int KAPPA = (5 + AMREX_SPACEDIM);

    static constexpr int size  = P + 1;
    static constexpr int vsize = KAPPA + 1;
};

#endif