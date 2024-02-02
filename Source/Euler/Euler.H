#ifndef Euler_H_
#define Euler_H_

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MultiFab.H>

#include "AmrLevelAdv.H"

#define MAX_STATE_SIZE AMREX_SPACEDIM + 2

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
    const amrex::Array4<const amrex::Real> &consv_values);

AMREX_GPU_HOST
amrex::Real max_wave_speed(amrex::Real time, const amrex::MultiFab &S_new);

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, MAX_STATE_SIZE> consv_from_primv(
    const amrex::GpuArray<const amrex::Real, MAX_STATE_SIZE> &primv_vals,
    const amrex::Real                                         adiabatic)
{
    amrex::GpuArray<amrex::Real, MAX_STATE_SIZE> consv_vals;
    // density
    consv_vals[0] = primv_vals[0];
    // momentum
    AMREX_D_TERM(consv_vals[1] = primv_vals[0] * primv_vals[1];
                 , consv_vals[2] = primv_vals[0] * primv_vals[2];
                 , consv_vals[3] = primv_vals[0] * primv_vals[3];)

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

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, MAX_STATE_SIZE>
primv_from_consv(const amrex::Array4<const amrex::Real> &consv_values,
                 const int i, const int j, const int k)
{
    AMREX_ASSERT(2 + AMREX_SPACEDIM <= MAX_STATE_SIZE);

    const amrex::Real adiabatic = AmrLevelAdv::d_prob_parm->adiabatic;

    amrex::GpuArray<amrex::Real, MAX_STATE_SIZE> primv_values;
    // density
    primv_values[0] = consv_values(i, j, k, 0);
    AMREX_D_TERM(primv_values[1]
                 = consv_values(i, j, k, 1) / consv_values(i, j, k, 0);
                 , primv_values[2]
                   = consv_values(i, j, k, 2) / consv_values(i, j, k, 0);
                 , primv_values[3]
                   = consv_values(i, j, k, 3) / consv_values(i, j, k, 0);)

    // Calculate u squared
    double u_sq = AMREX_D_TERM(primv_values[1] * primv_values[1],
                               +primv_values[2] * primv_values[2],
                               +primv_values[3] * primv_values[3]);

    // pressure
    primv_values[1 + AMREX_SPACEDIM]
        = (adiabatic - 1)
          * (consv_values(i, j, k, 1 + AMREX_SPACEDIM)
             - 0.5 * consv_values(i, j, k, 0) * u_sq);

    return primv_values;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, MAX_STATE_SIZE>
euler_flux(const int dir, const amrex::Array4<const amrex::Real> &consv_values,
           const int i, const int j, const int k)
{
    AMREX_ASSERT(2 + AMREX_SPACEDIM <= MAX_STATE_SIZE);

    const amrex::Real adiabatic = AmrLevelAdv::d_prob_parm->adiabatic;

    const amrex::GpuArray<amrex::Real, MAX_STATE_SIZE> primv_values
        = primv_from_consv(consv_values, i, j, k);
    amrex::GpuArray<amrex::Real, MAX_STATE_SIZE> euler_flux;

    // density flux
    const double pressure = primv_values[1 + AMREX_SPACEDIM];
    const double energy   = consv_values(i, j, k, 1 + AMREX_SPACEDIM);

    // density flux
    euler_flux[0] = consv_values(i, j, k, 1 + dir);

    // momentum flux
    AMREX_D_TERM(
        euler_flux[1] = consv_values(i, j, k, 1 + dir) * primv_values[1];
        , euler_flux[2] = consv_values(i, j, k, 1 + dir) * primv_values[2];
        , euler_flux[3] = consv_values(i, j, k, 1 + dir) * primv_values[3];)

    euler_flux[1 + dir] += pressure;

    // energy flux
    euler_flux[1 + AMREX_SPACEDIM]
        = (energy + pressure) * primv_values[1 + dir];

    return euler_flux;
}

#endif