#ifndef Euler_H_
#define Euler_H_

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MultiFab.H>

#include "AmrLevelAdv.H"
#include "Euler/NComp.H"

/**
 * \brief Computes the flux function values (for a given direction) across the
 * whole box given the primitive and conserved forms of the values.
 *
 * \param gbx should cover the region you want to compute the flux function
 * values on (which may have to include ghost cells)
 */
template <int dir>
AMREX_GPU_HOST void
compute_flux_function(amrex::Real time, const amrex::Box &gbx,
                      const amrex::Array4<amrex::Real>       &flux_func,
                      const amrex::Array4<const amrex::Real> &consv_values,
                      const amrex::Real adiabatic, const amrex::Real epsilon);

AMREX_GPU_HOST
amrex::Real max_wave_speed(amrex::Real time, const amrex::MultiFab &S_new);

AMREX_GPU_HOST
amrex::Real max_speed(const amrex::MultiFab &state);

/**
 * \brief Calculates the maximum value of (u / dx + v / dy + w / dz) on the
 * domain
 */
AMREX_GPU_HOST
amrex::Real
max_dx_scaled_speed(const amrex::MultiFab                              &state,
                    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx);

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, EULER_NCOMP>
consv_from_primv(const amrex::GpuArray<amrex::Real, EULER_NCOMP> &primv_vals,
                 const amrex::Real adiabatic, const amrex::Real epsilon);

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, EULER_NCOMP>
primv_from_consv(const amrex::Array4<const amrex::Real> &consv_values,
                 const amrex::Real adiabatic, const amrex::Real epsilon,
                 const int i, const int j, const int k);

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, EULER_NCOMP>
primv_from_consv(const amrex::Array4<const amrex::Real> &consv_values,
                 const amrex::Real adiabatic, const amrex::Real epsilon,
                 const amrex::IntVect &iv);

template <int dir>
AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE amrex::GpuArray<amrex::Real, EULER_NCOMP>
    euler_flux(const amrex::Array4<const amrex::Real> &consv_values,
               const amrex::Real adiabatic, const amrex::Real epsilon,
               const int i, const int j, const int k);

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real
sound_speed(const amrex::GpuArray<amrex::Real, EULER_NCOMP> &primv_values,
            const amrex::Real adiabatic, const amrex::Real epsilon);

/**
 * @brief Gives, for the Euler equations with ideal gas EoS, the Riemann
 * problem function
 * \f[
 *  f(p, \bm{Q}_L, \bm{Q}_R) = f_L(p, \bm{Q}_L) + f_R(p, \bm{Q}_R) + \Delta u
 *  \Delta u = u_R - u_L
 * \f]
 */
template <int dir>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real euler_riemann_function(
    const amrex::Real                                pressure,
    const amrex::GpuArray<amrex::Real, EULER_NCOMP> &primv_L,
    const amrex::GpuArray<amrex::Real, EULER_NCOMP> &primv_R,
    const amrex::Real adiabatic_L, const amrex::Real adiabatic_R,
    const amrex::Real epsilon);

/**
 * @brief Gives either f_L or f_R (depending on primitive values given) for the
 * Riemann problem equation
 * \f[
 *  f(p, \bm{Q}_L, \bm{Q}_R) = f_L(p, \bm{Q}_L) + f_R(p, \bm{Q}_R) + \Delta u
 *  \Delta u = u_R - u_L
 * \f]
 */
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real euler_riemann_function_one_side(
    const amrex::Real                                pressure,
    const amrex::GpuArray<amrex::Real, EULER_NCOMP> &primv,
    const amrex::Real                                adiabatic);

/**
 * @brief Gives, for the Euler equations with ideal gas EoS, the derivative
 * of the Riemann problem function
 * \f[
 *  f'(p, \bm{Q}_L, \bm{Q}_R) = f'_L(p, \bm{Q}_L) + f'_R(p, \bm{Q}_R)
 * \f]
 */
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real euler_riemann_function_deriv(
    const amrex::Real                                pressure,
    const amrex::GpuArray<amrex::Real, EULER_NCOMP> &primv_L,
    const amrex::GpuArray<amrex::Real, EULER_NCOMP> &primv_R,
    const amrex::Real adiabatic_L, const amrex::Real adiabatic_R);

/**
 * @brief Gives either f'_L or f'_R (depending on primitive values given) for
 * the Riemann problem equation \f[ f(p, \bm{Q}_L, \bm{Q}_R) = f_L(p, \bm{Q}_L)
 * + f_R(p, \bm{Q}_R) + \Delta u \Delta u = u_R - u_L \f]
 */
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real euler_riemann_function_one_side_deriv(
    const amrex::Real                                pressure,
    const amrex::GpuArray<amrex::Real, EULER_NCOMP> &primv,
    const amrex::Real                                adiabatic);

//
// Implementations
//

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, EULER_NCOMP>
consv_from_primv(const amrex::GpuArray<amrex::Real, EULER_NCOMP> &primv_vals,
                 const amrex::Real adiabatic, const amrex::Real epsilon)
{
    amrex::GpuArray<amrex::Real, EULER_NCOMP> consv_vals;
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
        = vol_internal_energy + 0.5 * epsilon * primv_vals[0] * u_sq;

    return consv_vals;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, EULER_NCOMP>
primv_from_consv(const amrex::Array4<const amrex::Real> &consv_values,
                 const amrex::Real adiabatic, const amrex::Real epsilon,
                 const int i, const int j, const int k)
{
    AMREX_ASSERT(2 + AMREX_SPACEDIM <= EULER_NCOMP);

    amrex::GpuArray<amrex::Real, EULER_NCOMP> primv_values;
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
             - 0.5 * epsilon * consv_values(i, j, k, 0) * u_sq);

    return primv_values;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, EULER_NCOMP>
primv_from_consv(const amrex::Array4<const amrex::Real> &consv_values,
                 const amrex::Real adiabatic, const amrex::Real epsilon,
                 const amrex::IntVect &iv)
{
    return primv_from_consv(consv_values, adiabatic, epsilon, iv.dim3().x,
                            iv.dim3().y, iv.dim3().z);
}

template <int dir>
AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE amrex::GpuArray<amrex::Real, EULER_NCOMP>
    euler_flux(const amrex::Array4<const amrex::Real> &consv_values,
               const amrex::Real adiabatic, const amrex::Real epsilon,
               const int i, const int j, const int k)
{
    AMREX_ASSERT(2 + AMREX_SPACEDIM <= EULER_NCOMP);

    const amrex::GpuArray<amrex::Real, EULER_NCOMP> primv_values
        = primv_from_consv(consv_values, adiabatic, epsilon, i, j, k);
    amrex::GpuArray<amrex::Real, EULER_NCOMP> euler_flux;

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

    euler_flux[1 + dir] += pressure / epsilon;

    // energy flux
    euler_flux[1 + AMREX_SPACEDIM]
        = (energy + pressure) * primv_values[1 + dir];

    return euler_flux;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real
sound_speed(const amrex::GpuArray<amrex::Real, EULER_NCOMP> &primv_values,
            const amrex::Real adiabatic, const amrex::Real epsilon)
{
    return std::sqrt(adiabatic * primv_values[1 + AMREX_SPACEDIM]
                     / (primv_values[0] * epsilon));
}

template <int dir>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real euler_riemann_function(
    const amrex::Real                                pressure,
    const amrex::GpuArray<amrex::Real, EULER_NCOMP> &primv_L,
    const amrex::GpuArray<amrex::Real, EULER_NCOMP> &primv_R,
    const amrex::Real adiabatic_L, const amrex::Real adiabatic_R,
    const amrex::Real epsilon)
{
    return euler_riemann_function_one_side(pressure, primv_L, adiabatic_L)
           + euler_riemann_function_one_side(pressure, primv_R, adiabatic_R)
           + sqrt(epsilon) * (primv_R[1 + dir] - primv_L[1 + dir]);
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real euler_riemann_function_one_side(
    const amrex::Real                                pressure,
    const amrex::GpuArray<amrex::Real, EULER_NCOMP> &primv,
    const amrex::Real                                adiabatic)
{
    const amrex::Real p_K = primv[1 + AMREX_SPACEDIM];

    if (pressure > p_K)
    {
        // shock
        const amrex::Real A_K = 2 / ((adiabatic + 1) * primv[0]);
        const amrex::Real B_K = ((adiabatic - 1) / (adiabatic + 1)) * p_K;
        return (pressure - p_K) * std::sqrt(A_K / ((pressure + B_K)));
    }
    else
    {
        // rarefaction
        const amrex::Real c_K = sound_speed(primv, adiabatic, 1);
        return 2 * c_K / (adiabatic - 1)
               * (std::pow(pressure / p_K, (adiabatic - 1) / (2 * adiabatic))
                  - 1);
    }
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real euler_riemann_function_deriv(
    const amrex::Real                                pressure,
    const amrex::GpuArray<amrex::Real, EULER_NCOMP> &primv_L,
    const amrex::GpuArray<amrex::Real, EULER_NCOMP> &primv_R,
    const amrex::Real adiabatic_L, const amrex::Real adiabatic_R)
{
    return euler_riemann_function_one_side_deriv(pressure, primv_L,
                                                 adiabatic_L)
           + euler_riemann_function_one_side_deriv(pressure, primv_R,
                                                   adiabatic_R);
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real euler_riemann_function_one_side_deriv(
    const amrex::Real                                pressure,
    const amrex::GpuArray<amrex::Real, EULER_NCOMP> &primv,
    const amrex::Real                                adiabatic)
{
    const amrex::Real p_K = primv[1 + AMREX_SPACEDIM];

    if (pressure > p_K)
    {
        // shock
        const amrex::Real A_K = 2 / ((adiabatic + 1) * primv[0]);
        const amrex::Real B_K = ((adiabatic - 1) / (adiabatic + 1)) * p_K;
        return std::sqrt(A_K / ((pressure + B_K)))
               * (1 - (pressure - p_K) / (2 * (B_K + pressure)));
    }
    else
    {
        // rarefaction
        const amrex::Real c_K = sound_speed(primv, adiabatic, 1);
        return std::pow(pressure / p_K, -(adiabatic + 1) / (2 * (adiabatic)))
               / (primv[0] * c_K);
    }
}

#endif