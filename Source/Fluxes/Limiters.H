#ifndef Limiters_H_
#define Limiters_H_

#include <AMReX_BLassert.H>
#include <cmath>
#include <string>

#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

enum Limiter
{
    minbee_limiter,
    van_leer_limiter,
    van_albada_limiter,
    superbee_limiter
};

inline Limiter limiter_from_name(const std::string &name);

template <Limiter limiter>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
                                    slope_limit(const amrex::Real r);

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real minbee(const amrex::Real r);
AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real van_leer(const amrex::Real r);
AMREX_GPU_DEVICE AMREX_FORCE_INLINE             amrex::Real
                                    van_albada(const amrex::Real r);
AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real superbee(const amrex::Real r);

//
// Implementations
//

inline Limiter limiter_from_name(const std::string &name)
{
    if (name == "minbee")
        return minbee_limiter;
    else if (name == "van_leer")
        return van_leer_limiter;
    else if (name == "van_albada")
        return van_albada_limiter;
    else if (name == "superbee")
        return superbee_limiter;
    else
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(false,
                                         "Invalid slope limiter: " + name);
        return minbee_limiter;
    }
}

template <Limiter limiter>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
                                    slope_limit(const amrex::Real r)
{
    switch (limiter)
    {
    case minbee_limiter: return minbee(r);
    case van_leer_limiter: return van_leer(r);
    case van_albada_limiter: return van_albada(r);
    case superbee_limiter: return superbee(r);
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real minbee(const amrex::Real r)
{
    return r <= 0 ? 0 : (r > 1 ? fmin(2. / (1 + r), 1.) : r);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real van_leer(const amrex::Real r)
{
    return r <= 0 ? 0 : fmin(2. * r / (1 + r), 2. / (1 + r));
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real van_albada(const amrex::Real r)
{
    return r <= 0 ? 0 : fmin(r * (1 + r) / (1 + r * r), 2. / (1 + r));
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real superbee(const amrex::Real r)
{
    return (r <= 0) ? 0
                    : ((0 < r && r <= 0.5)
                           ? 2 * r
                           : ((0.5 < r && r <= 1)
                                  ? 1
                                  : fmin(r, fmin(2. / (1 + r), 2.))));
    // if (r <= 0)
    //     return 0;
    // else if (0 < r && r <= 0.5)
    //     return 2 * r;
    // else if (0.5 < r && r <= 1)
    //     return 1;
    // else
    //     return std::fmin(r, std::fmin(2. / (1 + r), 2));
}

#endif