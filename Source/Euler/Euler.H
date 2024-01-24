#ifndef Euler_H_
#define Euler_H_

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_GpuQualifiers.H>

/**
 * \brief Computes primitive values for ideal gas Euler equations
 *
 * \param gbx should cover the region you want to compute primitive values on
 * (which may have to include ghost cells)
 */
AMREX_GPU_HOST
void compute_primitive_values(
    amrex::Real time, const amrex::Box &gbx,
    const amrex::Array4<amrex::Real>       &primv_values,
    const amrex::Array4<const amrex::Real> &consv_values);

/**
 * \brief Computes the flux function values (for a given direction) across the
 * whole box given the primitive and conserved forms of the values.
 *
 * \param gbx should cover the region you want to compute the flux function
 * values on (which may have to include ghost cells)
 */
AMREX_GPU_HOST
void compute_flux_function(
    const int dir, amrex::Real time, const amrex::Box &gbx,
    const amrex::Array4<amrex::Real>       &flux_func,
    const amrex::Array4<const amrex::Real> &primv_values,
    const amrex::Array4<const amrex::Real> &consv_values);

#endif