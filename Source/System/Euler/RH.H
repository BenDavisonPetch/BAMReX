/**
 * Routines for calculating post-shock conditions
 */
#ifndef RH_H_
#define RH_H_

#include <AMReX_Array.H>
#include <AMReX_REAL.H>

#include "System/Parm.H"

/**
 * Calculates post-shock conditions for a shock defined with Mach number
 * relative to stationary ambient conditions. The returned GpuArray is in 1D
 * primitive variables (i.e. density, velocity, pressure)
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::GpuArray<amrex::Real, 3>
post_shock(amrex::Real M, amrex::Real rho_amb, amrex::Real p_amb,
           const Parm &parm)
{
    using namespace amrex;
    GpuArray<Real, 3> Q;

    const Real c_amb = sqrt(p_amb * parm.adia[0] / rho_amb);
    Q[0] = rho_amb * (parm.adia[1] * M * M) / (2 + parm.adia[2] * M * M);
    Q[1] = c_amb * 2 * (M * M - 1) / (parm.adia[1] * M);
    Q[2] = p_amb * (1 + 2 * parm.adia[7] * (M * M - 1));
    return Q;
}

#endif