#ifndef TAGGING_K_H_
#define TAGGING_K_H_

#include "GFM/GFMFlag.H"
#include <AMReX_Array4.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

// Tag regions of high phi.
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void state_error(int i, int j, int k, amrex::Array4<char> const &tag,
                 amrex::Array4<amrex::Real const> const &state,
                 amrex::Real const phi_threshold, char const tagval) noexcept
{
    if (state(i, j, k) >= phi_threshold)
    {
        tag(i, j, k) = tagval;
    }
}

// Tag regions of high phi gradient.
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void grad_error(int i, int j, int k, amrex::Array4<char> const &tag,
                amrex::Array4<amrex::Real const> const &state,
                amrex::Real const grad_threshold, char const tagval) noexcept
{
    amrex::Real ax = std::abs(state(i - 1, j, k) - state(i, j, k));
    ax = amrex::max(ax, std::abs(state(i, j, k) - state(i + 1, j, k)));

    amrex::Real ay;
#if AMREX_SPACEDIM < 2
    ay = 0.0;
#else
    ay = std::abs(state(i, j - 1, k) - state(i, j, k));
    ay = amrex::max(ay, std::abs(state(i, j, k) - state(i, j + 1, k)));
#endif

    amrex::Real az;
#if AMREX_SPACEDIM < 3
    az = 0.0;
#else
    az = std::abs(state(i, j, k - 1) - state(i, j, k));
    az = amrex::max(az, std::abs(state(i, j, k) - state(i, j, k + 1)));
#endif
    if (amrex::max(ax, ay, az) >= grad_threshold * std::abs(state(i, j, k)))
    {
        tag(i, j, k) = tagval;
    }
}

// Tag cells at the interface
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void interface_error(int i, int j, int k, amrex::Array4<char> const &tag,
                     amrex::Array4<amrex::Real const> const &LS,
                     amrex::Array4<int const> const         &gfm_flags,
                     amrex::Real dist_tol, char const tagval) noexcept
{
    if (at_interface(gfm_flags(i, j, k)) || fabs(LS(i, j, k)) <= dist_tol)
        tag(i, j, k) = tagval;
}

// Untag ghost cells not at the interface
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void ghost_error(int i, int j, int k, amrex::Array4<char> const &tag,
                 amrex::Array4<amrex::Real const> const & /*LS*/,
                 amrex::Array4<int const> const &gfm_flags,
                 amrex::Real /*dist_tol*/, char const clearval) noexcept
{
    if (is_ghost(gfm_flags(i, j, k)) && !at_interface(gfm_flags(i, j, k)))
        tag(i, j, k) = clearval;
}

#endif /* TAGGING_K_H_ */
