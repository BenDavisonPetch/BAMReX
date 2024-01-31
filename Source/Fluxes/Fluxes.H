#ifndef Fluxes_H_
#define Fluxes_H_

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_GpuQualifiers.H>

#include "Euler/Euler.H" // This include determines which material everything uses

/**
 * \param dir the direction in which to calculate the flux
 * \param time the current time
 * \param bx the Box object to iterate over (obtained from an MFIter)
 * \param flux the flux array to write to. Represents the flux in direction
 * \a dir
 * \param consv_values the variable values in conservative form
 * \param dx_arr
 * \param dt
 *
 * Implemented in Fluxes/Force.cpp
 */
AMREX_GPU_HOST
void compute_force_flux(
    const int dir, amrex::Real time, const amrex::Box &bx,
    const amrex::Array4<amrex::Real>                &flux,
    const amrex::Array4<const amrex::Real>          &consv_values,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr, amrex::Real dt);

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
 *
 * Implemented in Fluxes/Force.cpp
 */
AMREX_GPU_HOST
void compute_force_flux_LR(
    const int dir, amrex::Real time, const amrex::Box &bx,
    const amrex::Array4<amrex::Real>                &flux,
    const amrex::Array4<const amrex::Real>          &consv_values_L,
    const amrex::Array4<const amrex::Real>          &consv_values_R,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr, amrex::Real dt);

AMREX_GPU_HOST
void compute_LF_flux(const int dir, amrex::Real time, const amrex::Box &bx,
                     const amrex::Array4<amrex::Real>       &flux,
                     const amrex::Array4<const amrex::Real> &consv_values,
                     amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr,
                     amrex::Real                                      dt);

AMREX_GPU_HOST
void compute_SLIC_flux(const int dir, amrex::Real time, const amrex::Box &bx,
                       const amrex::Array4<amrex::Real>       &flux,
                       const amrex::Array4<const amrex::Real> &consv_values,
                       amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr,
                       amrex::Real                                      dt);

#endif