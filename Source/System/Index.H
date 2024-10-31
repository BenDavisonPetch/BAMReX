#ifndef IndexMacros_H_
#define IndexMacros_H_

#include <AMReX.H>

//! Conservative indices for fluid system
struct UInd
{
    static constexpr int RHO = 0;
    static constexpr int MX  = 1;
    static constexpr int MY  = 2;
    static constexpr int MZ  = 3;
    static constexpr int E   = (1 + AMREX_SPACEDIM);

    static constexpr int size = E + 1;

    static inline constexpr int Mdir(int dir)
    {
        switch (dir)
        {
        case 0: return MX;
        case 1: return MY;
        case 2: return MZ;
        };
        return -1;
    }
};

//! Primitive indices for fluid system
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

    static inline constexpr int Udir(int dir)
    {
        switch (dir)
        {
        case 0: return U;
        case 1: return V;
        case 2: return W;
        };
        return -1;
    }
};

#endif