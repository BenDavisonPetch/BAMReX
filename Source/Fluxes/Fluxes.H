#ifndef Fluxes_H_
#define Fluxes_H_

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_GpuQualifiers.H>

#include "System/Euler/Euler.H" // This include determines which material everything uses

/**
 * \param dir the direction in which to calculate the flux
 * \param time the current time
 * \param bx the Box object to iterate over (obtained from an MFIter),
 * covering the real cells in the update
 * \param flux the flux array to write to. Represents the flux in direction
 * \a dir
 * \param consv_values the variable values in conservative form
 * \param dx_arr
 * \param dt
 * \param system
 * \param h_parm pointer to host-allocated parameters
 * \param d_parm pointer to device-allocated parameters
 *
 * Implemented in Fluxes/Force.cpp
 */
template <int dir>
AMREX_GPU_HOST void
compute_force_flux(amrex::Real time, const amrex::Box &bx,
                   const amrex::Array4<amrex::Real>       &flux,
                   const amrex::Array4<const amrex::Real> &consv_values,
                   amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr,
                   amrex::Real dt, const System *system, const Parm *h_parm,
                   const Parm *d_parm);

/**
 * Allows for calculation of the FORCE flux using a different set of values for
 * left and right conservative variables
 *
 * \param dir the direction in which to calculate the flux
 * \param time the current time
 * \param bx the Box object to iterate over (obtained from an MFIter)
 * \param flux the flux array to write to. Represents the flux in direction
 * \a dir
 * \param consv_values_L the variable values in conservative form to use for
 * index i (in x dir)
 * \param consv_values_R the variable values in conservative form to use for
 * index i+1 (in x dir)
 * \param dx_arr
 * \param dt
 * \param system
 * \param h_parm pointer to host-allocated parameters
 * \param d_parm pointer to device-allocated parameters
 *
 * Implemented in Fluxes/Force.cpp
 */
template <int dir>
AMREX_GPU_HOST void
compute_force_flux_LR(amrex::Real time, const amrex::Box &bx,
                      const amrex::Array4<amrex::Real>       &flux,
                      const amrex::Array4<const amrex::Real> &consv_values_L,
                      const amrex::Array4<const amrex::Real> &consv_values_R,
                      amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr,
                      amrex::Real dt, const System *system, const Parm *h_parm,
                      const Parm *d_parm);

template <int dir>
AMREX_GPU_HOST void
compute_LF_flux(amrex::Real time, const amrex::Box &bx,
                const amrex::Array4<amrex::Real>                &flux,
                const amrex::Array4<const amrex::Real>          &consv_values,
                amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr,
                amrex::Real dt, const System *system, const Parm *h_parm,
                const Parm *d_parm);

template <int dir>
AMREX_GPU_HOST void
compute_SLIC_flux(amrex::Real time, const amrex::Box &bx,
                  const amrex::Array4<amrex::Real>       &flux,
                  const amrex::Array4<const amrex::Real> &consv_values,
                  amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr,
                  amrex::Real dt, const System *system, const Parm *h_parm,
                  const Parm *d_parm);

/**
 * Allows for calculation of the HLLC flux using a different set of values for
 * left and right conservative variables
 *
 * \param dir the direction in which to calculate the flux
 * \param time the current time
 * \param bx the Box object covering the real cells in the update
 * \param flux the flux array to write to. Represents the flux in direction
 * \c dir
 * \param consv_values_L the variable values in conservative form to use for
 * index i (in x dir)
 * \param consv_values_R the variable values in conservative form to use for
 * index i+1 (in x dir)
 * \param dx_arr
 * \param dt
 * \param system
 * \param h_parm pointer to host-allocated parameters
 * \param d_parm pointer to device-allocated parameters
 *
 * Implemented in Fluxes/HLLC.cpp
 *
 * @note When using this function in a slope limiting scheme, you may have to
 * pass your reconstructed states labelled left into \c consv_values_R and vice
 * versa.
 */
template <int dir>
AMREX_GPU_HOST void
compute_HLLC_flux_LR(amrex::Real time, const amrex::Box &bx,
                     const amrex::Array4<amrex::Real>       &flux,
                     const amrex::Array4<const amrex::Real> &consv_values_L,
                     const amrex::Array4<const amrex::Real> &consv_values_R,
                     amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr,
                     amrex::Real dt, const System *system, const Parm *h_parm,
                     const Parm *d_parm);

/**
 * @brief For convenience. Calls \c compute_HLLC_flux_LR
 */
template <int dir>
AMREX_GPU_HOST AMREX_FORCE_INLINE void
compute_HLLC_flux(amrex::Real time, const amrex::Box &bx,
                  const amrex::Array4<amrex::Real>       &flux,
                  const amrex::Array4<const amrex::Real> &consv_values,
                  amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr,
                  amrex::Real dt, const System *system, const Parm *h_parm,
                  const Parm *d_parm)
{

    compute_HLLC_flux_LR<dir>(time, bx, flux, consv_values, consv_values,
                              dx_arr, dt, system, h_parm, d_parm);
}

AMREX_GPU_HOST AMREX_FORCE_INLINE void
compute_HLLC_flux(int dir, amrex::Real time, const amrex::Box &bx,
                  const amrex::Array4<amrex::Real>       &flux,
                  const amrex::Array4<const amrex::Real> &consv_values,
                  amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr,
                  amrex::Real dt, const System *system, const Parm *h_parm,
                  const Parm *d_parm)
{
    if (dir == 0)
        compute_HLLC_flux<0>(time, bx, flux, consv_values, dx_arr, dt, system,
                             h_parm, d_parm);
#if AMREX_SPACEDIM >= 2
    else if (dir == 1)
        compute_HLLC_flux<1>(time, bx, flux, consv_values, dx_arr, dt, system,
                             h_parm, d_parm);
#endif
#if AMREX_SPACEDIM == 3
    else
        compute_HLLC_flux<2>(time, bx, flux, consv_values, dx_arr, dt, system,
                             h_parm, d_parm);
#endif
}

template <int dir>
AMREX_GPU_HOST void compute_MUSCL_hancock_flux(
    amrex::Real time, const amrex::Box &bx,
    const amrex::Array4<amrex::Real>                &flux,
    const amrex::Array4<const amrex::Real>          &consv_values,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr, amrex::Real dt,
    const System *system, const Parm *h_parm, const Parm *d_parm);

AMREX_FORCE_INLINE AMREX_GPU_HOST void compute_MUSCL_hancock_flux(
    int dir, amrex::Real time, const amrex::Box &bx,
    const amrex::Array4<amrex::Real>                &flux,
    const amrex::Array4<const amrex::Real>          &consv_values,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr, amrex::Real dt,
    const System *system, const Parm *h_parm, const Parm *d_parm)
{
    if (dir == 0)
        compute_MUSCL_hancock_flux<0>(time, bx, flux, consv_values, dx_arr, dt,
                                      system, h_parm, d_parm);
#if AMREX_SPACEDIM >= 2
    else if (dir == 1)
        compute_MUSCL_hancock_flux<1>(time, bx, flux, consv_values, dx_arr, dt,
                                      system, h_parm, d_parm);
#endif
#if AMREX_SPACEDIM == 3
    else
        compute_MUSCL_hancock_flux<2>(time, bx, flux, consv_values, dx_arr, dt,
                                      system, h_parm, d_parm);
#endif
}

#endif