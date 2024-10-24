#include "Euler.H"
#include "AmrLevelAdv.H"

#include <AMReX_BLassert.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFParallelFor.H>

using namespace amrex;

template <int dir>
AMREX_GPU_HOST void
compute_flux_function(amrex::Real /*time*/, const amrex::Box &bx,
                      const amrex::Array4<amrex::Real>       &flux_func,
                      const amrex::Array4<const amrex::Real> &consv_values,
                      const amrex::Real adiabatic, const amrex::Real epsilon)
{
    AMREX_ASSERT(dir < AMREX_SPACEDIM);
    ParallelFor(
        bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            const auto primv_values
                = primv_from_consv(consv_values, adiabatic, epsilon, i, j, k);
            const double pressure = primv_values[1 + AMREX_SPACEDIM];
            const double energy   = consv_values(i, j, k, 1 + AMREX_SPACEDIM);

            // density flux
            flux_func(i, j, k, 0) = consv_values(i, j, k, 1 + dir);

            // momentum flux
            AMREX_D_TERM(flux_func(i, j, k, 1)
                         = consv_values(i, j, k, 1 + dir) * primv_values[1];
                         , flux_func(i, j, k, 2)
                           = consv_values(i, j, k, 1 + dir) * primv_values[2];
                         , flux_func(i, j, k, 3)
                           = consv_values(i, j, k, 1 + dir) * primv_values[3];)

            flux_func(i, j, k, 1 + dir) += pressure / epsilon;

            // energy flux
            flux_func(i, j, k, 1 + AMREX_SPACEDIM)
                = (energy + pressure) * primv_values[1 + dir];
        });
}

AMREX_GPU_HOST
amrex::Real max_wave_speed(amrex::Real /*time*/, const amrex::MultiFab &S_new,
                           amrex::Real adiabatic, amrex::Real epsilon)
{
    BL_PROFILE("max_wave_speed()");

    auto const &ma         = S_new.const_arrays();
    auto        wave_speed = ParReduce(
               TypeList<ReduceOpMax>{}, TypeList<Real>{}, S_new,
               IntVect(0), // no ghost cells
               [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k)
                   noexcept -> GpuTuple<Real>
               {
            const Array4<const Real> &consv = ma[box_no];
            const auto                primv_arr
                = primv_from_consv(consv, adiabatic, epsilon, i, j, k);
            const Real velocity = std::sqrt(AMREX_D_TERM(
                       primv_arr[1] * primv_arr[1], +primv_arr[2] * primv_arr[2],
                       +primv_arr[3] * primv_arr[3]));
            return sqrt(adiabatic * primv_arr[1 + AMREX_SPACEDIM]
                               / (primv_arr[0] * epsilon))
                   + velocity;
        });

    BL_PROFILE_VAR("wave_wave_speed::ReduceRealMax", preduce);
    ParallelDescriptor::ReduceRealMax(wave_speed);
    BL_PROFILE_VAR_STOP(preduce);
    return wave_speed;
}

AMREX_GPU_HOST
amrex::Real max_speed(const amrex::MultiFab &state)
{
    BL_PROFILE("max_speed()");
    auto const &ma         = state.const_arrays();
    auto        wave_speed = ParReduce(
               TypeList<ReduceOpMax>{}, TypeList<Real>{}, state,
               IntVect(0), // no ghost cells
               [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k)
                   noexcept -> GpuTuple<Real>
               {
            const Array4<const Real> &consv = ma[box_no];
            const Real                u_sq
                = (AMREX_D_TERM(consv(i, j, k, 1) * consv(i, j, k, 1),
                                       +consv(i, j, k, 2) * consv(i, j, k, 2),
                                       +consv(i, j, k, 3) * consv(i, j, k, 3)))
                  / (consv(i, j, k, 0) * consv(i, j, k, 0));
            return { sqrt(u_sq) };
        });

    BL_PROFILE_VAR("max_speed::ReduceRealMax", preduce);
    ParallelDescriptor::ReduceRealMax(wave_speed);
    BL_PROFILE_VAR_STOP(preduce);
    return wave_speed;
}

/**
 * \brief Calculates the maximum value of (|u| / dx + |v| / dy + |w| / dz) on
 * the domain
 */
AMREX_GPU_HOST
amrex::Real max_dx_scaled_speed(const amrex::MultiFab                &state,
                                const GpuArray<Real, AMREX_SPACEDIM> &dx)
{
    BL_PROFILE("max_dx_scaled_speed()");
    auto const &ma = state.const_arrays();
    auto        wave_speed
        = ParReduce(TypeList<ReduceOpMax>{}, TypeList<Real>{}, state,
                    IntVect(0), // no ghost cells
                    [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k)
                        noexcept -> GpuTuple<Real>
                    {
                        const Array4<const Real> &consv = ma[box_no];
                        return (AMREX_D_TERM(fabs(consv(i, j, k, 1)) / dx[0],
                                             +fabs(consv(i, j, k, 2)) / dx[1],
                                             +fabs(consv(i, j, k, 3)) / dx[2]))
                               / consv(i, j, k, 0);
                    });

    BL_PROFILE_VAR("max_dx_scaled_speed::ReduceRealMax", preduce);
    ParallelDescriptor::ReduceRealMax(wave_speed);
    BL_PROFILE_VAR_STOP(preduce);
    return wave_speed;
}

template AMREX_GPU_HOST void
compute_flux_function<0>(amrex::Real, const amrex::Box &,
                         const amrex::Array4<amrex::Real> &,
                         const amrex::Array4<const amrex::Real> &,
                         const amrex::Real, const amrex::Real);

#if AMREX_SPACEDIM >= 2
template AMREX_GPU_HOST void
compute_flux_function<1>(amrex::Real, const amrex::Box &,
                         const amrex::Array4<amrex::Real> &,
                         const amrex::Array4<const amrex::Real> &,
                         const amrex::Real, const amrex::Real);
#endif
#if AMREX_SPACEDIM >= 3
template AMREX_GPU_HOST void
compute_flux_function<2>(amrex::Real, const amrex::Box &,
                         const amrex::Array4<amrex::Real> &,
                         const amrex::Array4<const amrex::Real> &,
                         const amrex::Real, const amrex::Real);
#endif