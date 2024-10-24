#ifndef CalcLimitingVal_H_
#define CalcLimitingVal_H_

#include "System/Euler/Euler.H"
#include "System/Parm.H"

#include <AMReX_Array4.H>

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
calc_limiting_var(const amrex::Array4<const amrex::Real> &U,
                  const amrex::Array4<const amrex::Real> &Q, const Parm &parm,
                  int i, int j, int k, int n) noexcept
{
    if (parm.limit_on_primv[n])
    {
        return Q(i, j, k, parm.limiter_indices[n]);
    }
    else
    {
        return U(i, j, k, parm.limiter_indices[n]);
    }
}

#endif