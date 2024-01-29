#ifndef Euler_H_
#define Euler_H_

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MultiFab.H>

#define MAX_STATE_SIZE AMREX_SPACEDIM + 2

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

AMREX_GPU_HOST
amrex::Real max_wave_speed(amrex::Real time, const amrex::MultiFab &S_new);

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, MAX_STATE_SIZE> conservative_from_primitive(
    const amrex::GpuArray<const amrex::Real, MAX_STATE_SIZE> &primv_vals,
    const amrex::Real                                         adiabatic)
{
    amrex::GpuArray<amrex::Real, MAX_STATE_SIZE> consv_vals;
    // density
    consv_vals[0] = primv_vals[0];
    // momentum
    AMREX_D_TERM(consv_vals[1] = primv_vals[0] * primv_vals[1];
                 , consv_vals[2] = primv_vals[0] * primv_vals[2];
                 , consv_vals[3] = primv_Vals[0] * primv_vals[3];)

    const double pressure            = primv_vals[1 + AMREX_SPACEDIM];
    const double vol_internal_energy = pressure / (adiabatic - 1);

    // Calculate u squared
    double u_sq = AMREX_D_TERM(primv_vals[1] * primv_vals[1],
                               +primv_vals[2] * primv_vals[2],
                               +primv_vals[3] * primv_vals[3]);

    // energy
    consv_vals[1 + AMREX_SPACEDIM]
        = vol_internal_energy + 0.5 * primv_vals[0] * u_sq;

    return consv_vals;
}

#endif