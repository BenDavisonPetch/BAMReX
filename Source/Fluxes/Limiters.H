#ifndef Limiters_H_
#define Limiters_H_

#include <cmath>

#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE amrex::Real minbee(const amrex::Real r)
{
    return r <= 0 ? 0 : (r > 1 ? std::fmin(2. / (1 + r), 1.) : r);
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE amrex::Real van_leer(const amrex::Real r)
{
    return r <= 0 ? 0 : std::fmin(2. * r / (1 + r), 2. / (1 + r));
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE amrex::Real van_albada(const amrex::Real r)
{
    return r <= 0 ? 0 : std::fmin(r * (1 + r) / (1 + r * r), 2. / (1 + r));
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE amrex::Real superbee(const amrex::Real r)
{
    if (r <= 0)
        return 0;
    else if (0 < r && r <= 0.5)
        return 2 * r;
    else if (0.5 < r && r <= 1)
        return 1;
    else
        return std::fmin(r, std::fmin(2. / (1 + r), 2));
}

#endif