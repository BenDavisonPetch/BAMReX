#include "Euler.H"
#include "AmrLevelAdv.H"

#include <AMReX_BLassert.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFParallelFor.H>

using namespace amrex;

AMREX_GPU_HOST
void compute_flux_function(
    const int dir, amrex::Real /*time*/, const amrex::Box &bx,
    const amrex::Array4<amrex::Real>       &flux_func,
    const amrex::Array4<const amrex::Real> &consv_values,
    const amrex::Real                       adiabatic)
{
    AMREX_ASSERT(dir < AMREX_SPACEDIM);
    ParallelFor(
        bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            const auto primv_values
                = primv_from_consv(consv_values, adiabatic, i, j, k);
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

            flux_func(i, j, k, 1 + dir) += pressure;

            // energy flux
            flux_func(i, j, k, 1 + AMREX_SPACEDIM)
                = (energy + pressure) * primv_values[1 + dir];
        });
}

AMREX_GPU_HOST
amrex::Real max_wave_speed(amrex::Real time, const amrex::MultiFab &S_new)
{
    const Real adiabatic  = AmrLevelAdv::d_prob_parm->adiabatic;
    Real       wave_speed = 0;

    for (MFIter mfi(S_new, true); mfi.isValid(); ++mfi)
    {
        const Box  &bx  = mfi.tilebox();
        const auto &arr = S_new.array(mfi);

        ParallelFor(bx,
                    [=, &wave_speed] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        const auto primv_arr
                            = primv_from_consv(arr, adiabatic, i, j, k);
                        const Real velocity = std::sqrt(
                            AMREX_D_TERM(primv_arr[1] * primv_arr[1],
                                         +primv_arr[2] * primv_arr[2],
                                         +primv_arr[3] * primv_arr[3]));
                        wave_speed = std::max(
                            wave_speed,
                            std::sqrt(adiabatic * primv_arr[1 + AMREX_SPACEDIM]
                                      / primv_arr[0])
                                + velocity);
                    });
    }

    ParallelDescriptor::ReduceRealMax(wave_speed);

    return wave_speed;
}