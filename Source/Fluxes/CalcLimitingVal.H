#ifndef CalcLimitingVal_H_
#define CalcLimitingVal_H_

#include "Euler/Euler.H"
#include "Parm.H"

#include <AMReX_Array4.H>

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
calc_limiting_var(const amrex::Array4<const amrex::Real> &U, const Parm &parm,
                  int i, int j, int k, int n) noexcept
{
    if (parm.limit_on_primv[n])
    {
        const auto primv
            = primv_from_consv(U, parm.adiabatic, parm.epsilon, i, j, k);
        return primv[parm.limiter_indices[n]];
    }
    else
    {
        return U(i, j, k, parm.limiter_indices[n]);
    }
}

#endif