#ifndef Multidim_H_
#define Multidim_H_

#include <AMReX.H>

#define MULTIDIM__(x, dim) x##_##dim##D
#define MULTIDIM_(x, dim)  MULTIDIM__(x, dim)
#define MULTIDIM(x)        MULTIDIM_(x, AMREX_SPACEDIM)

#endif