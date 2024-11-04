#ifndef PlasmaIndex_H_
#define PlasmaIndex_H_

#include <AMReX.H>
#include <AMReX_BLassert.H>

//! Conservative indices for plasma system
struct PUInd
{
    static constexpr int RHO = 0;
    static constexpr int MX  = 1;
    static constexpr int MY  = 2;
    static constexpr int MZ  = 3;
    static constexpr int E   = 4;
    static constexpr int BX  = 5;
    static constexpr int BY  = 6;
    static constexpr int BZ  = 7;

    static constexpr int size = BZ + 1;

    static inline constexpr int Mdir(int dir)
    {
        switch (dir)
        {
        case 0: return MX;
        case 1: return MY;
        case 2: return MZ;
        };
        AMREX_ASSERT(false);
        return -1;
    }

    static inline constexpr int Bdir(int dir)
    {
        switch (dir)
        {
        case 0: return BX;
        case 1: return BY;
        case 2: return BZ;
        };
        AMREX_ASSERT(false);
        return -1;
    }
};

//! Primitive indices for plasma system
struct PQInd
{
    static constexpr int RHO  = 0;
    static constexpr int U    = 1;
    static constexpr int V    = 2;
    static constexpr int W    = 3;
    static constexpr int P    = 4;
    static constexpr int PTOT = 5;
    static constexpr int BX   = 6;
    static constexpr int BY   = 7;
    static constexpr int BZ   = 8;
    //! Extra parts for viscous computations
    static constexpr int T     = 9;
    static constexpr int ETA   = 10;
    static constexpr int ZETA  = 11;
    static constexpr int KAPPA = 12;

    static constexpr int size  = BZ + 1;
    static constexpr int vsize = KAPPA + 1;
    // Used to return primitive variables without having to copy over B
    static constexpr int size_noB = PTOT + 1;

    static inline constexpr int Udir(int dir)
    {
        switch (dir)
        {
        case 0: return U;
        case 1: return V;
        case 2: return W;
        };
        AMREX_ASSERT(false);
        return -1;
    }

    static inline constexpr int Bdir(int dir)
    {
        switch (dir)
        {
        case 0: return BX;
        case 1: return BY;
        case 2: return BZ;
        };
        return -1;
    }
};

#endif