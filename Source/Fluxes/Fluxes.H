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
 * \param primv_values the variable values in primitive form
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

#endif