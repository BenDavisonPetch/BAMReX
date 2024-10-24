#ifndef Flux_Helpers_H_
#define Flux_Helpers_H_

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_GpuQualifiers.H>

#include "Fluxes/Limiters.H"
#include "System/System.H"

/**
 * @brief Uses the slope-limiting method to linearly reconstruct and locally
 * half-time-step update variables.
 *
 * \tparam dir The direction in which to reconstruct.
 * \param gbx The box over which to calculate the locally updated variables.
 * \param time The current time.
 * \param half_stepped_L An amrex::Array4<Real> used to return the locally
 * updated values on the low-index side of each cell.
 * \param half_stepped_R An amrex::Array4<Real> used to return the locally
 * updated values on the high-index side of each cell.
 * \param consv_values The value to reconstruction and half-time-step.
 * \param dx_arr The array of dx/dy/dz values.
 * \param dt The full time step.
 * \param system
 * \param h_parm pointer to host-allocated parameters
 * \param d_parm pointer to device-allocated parameters
 *
 * \details The consv_values should be defined on a box grown by at least 1
 * index compared to \c gbx. Note that \c gbx itself would normally be grown
 * by 1 cell compared to the box of real cells when using this routine in e.g.
 * SLIC or MUSCL-Hancock
 */
template <int dir>
AMREX_GPU_HOST void reconstruct_and_half_time_step(
    const amrex::Box &gbx, const amrex::Real time,
    const amrex::Array4<amrex::Real>                &half_stepped_L,
    const amrex::Array4<amrex::Real>                &half_stepped_R,
    const amrex::Array4<const amrex::Real>          &consv_values,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr, amrex::Real dt,
    const System *system, const Parm *h_parm, const Parm *d_parm);

template <Limiter limiter, int dir>
AMREX_GPU_HOST void reconstruct_and_half_time_step(
    const amrex::Box &gbx, const amrex::Real time,
    const amrex::Array4<amrex::Real> &half_stepped_L,
    const amrex::Array4<amrex::Real> &half_stepped_R,
    const amrex::Array4<const amrex::Real>          &consv_values,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr, amrex::Real dt,
    const System *system, const Parm *h_parm, const Parm *d_parm);

#endif