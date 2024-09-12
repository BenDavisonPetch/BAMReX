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
                 amrex::Real const phi_threshold, unsigned int var_idx,
                 char const tagval) noexcept
{
    if (state(i, j, k, var_idx) >= phi_threshold)
    {
        tag(i, j, k) = tagval;
    }
}

// Tag regions of high phi gradient.
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void grad_error(int i, int j, int k, amrex::Array4<char> const &tag,
                amrex::Array4<amrex::Real const> const &state,
                amrex::Real const grad_threshold, unsigned int var_idx,
                char const tagval) noexcept
{
    amrex::Real ax
        = std::abs(state(i - 1, j, k, var_idx) - state(i, j, k, var_idx));
    ax = amrex::max(
        ax, std::abs(state(i, j, k, var_idx) - state(i + 1, j, k, var_idx)));

    amrex::Real ay;
#if AMREX_SPACEDIM < 2
    ay = 0.0;
#else
    ay = std::abs(state(i, j - 1, k, var_idx) - state(i, j, k, var_idx));
    ay = amrex::max(
        ay, std::abs(state(i, j, k, var_idx) - state(i, j + 1, k, var_idx)));
#endif

    amrex::Real az;
#if AMREX_SPACEDIM < 3
    az = 0.0;
#else
    az = std::abs(state(i, j, k - 1, var_idx) - state(i, j, k, var_idx));
    az = amrex::max(
        az, std::abs(state(i, j, k, var_idx) - state(i, j, k + 1, var_idx)));
#endif
    constexpr amrex::Real VAL_MAG_MIN = 1e-14;
    if (amrex::max(ax, ay, az)
        >= grad_threshold
               * amrex::max(std::abs(state(i, j, k, var_idx)), VAL_MAG_MIN))
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
