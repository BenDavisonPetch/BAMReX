#include "Euler.H"
#include "AmrLevelAdv.H"

#include <AMReX_BLassert.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFParallelFor.H>

using namespace amrex;

AMREX_GPU_HOST
void compute_primitive_values(
    amrex::Real /*time*/, const amrex::Box &bx,
    const amrex::Array4<amrex::Real>       &primv_values,
    const amrex::Array4<const amrex::Real> &consv_values)
{
    double adiabatic = AmrLevelAdv::d_prob_parm->adiabatic;
    ParallelFor(
        bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            // density
            primv_values(i, j, k, 0) = consv_values(i, j, k, 0);
            // velocities
            AMREX_D_TERM(
                primv_values(i, j, k, 1)
                = consv_values(i, j, k, 1) / consv_values(i, j, k, 0);
                , primv_values(i, j, k, 2)
                  = consv_values(i, j, k, 2) / consv_values(i, j, k, 0);
                , primv_values(i, j, k, 3)
                  = consv_values(i, j, k, 3) / consv_values(i, j, k, 0);)

            // Calculate u squared
            double u_sq = AMREX_D_TERM(
                primv_values(i, j, k, 1) * primv_values(i, j, k, 1),
                +primv_values(i, j, k, 2) * primv_values(i, j, k, 2),
                +primv_values(i, j, k, 3) * primv_values(i, j, k, 3));

            // pressure
            primv_values(i, j, k, 1 + AMREX_SPACEDIM)
                = (adiabatic - 1)
                  * (consv_values(i, j, k, 1 + AMREX_SPACEDIM)
                     - 0.5 * consv_values(i, j, k, 0) * u_sq);
        });
}

AMREX_GPU_HOST
void compute_flux_function(
    const int dir, amrex::Real /*time*/, const amrex::Box &bx,
    const amrex::Array4<amrex::Real>       &flux_func,
    const amrex::Array4<const amrex::Real> &primv_values,
    const amrex::Array4<const amrex::Real> &consv_values)
{
    AMREX_ASSERT(dir < AMREX_SPACEDIM);
    ParallelFor(
        bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            const double pressure = primv_values(i, j, k, 1 + AMREX_SPACEDIM);
            const double energy   = consv_values(i, j, k, 1 + AMREX_SPACEDIM);

            // density flux
            flux_func(i, j, k, 0) = consv_values(i, j, k, 1 + dir);

            // momentum flux
            AMREX_D_TERM(
                flux_func(i, j, k, 1)
                = consv_values(i, j, k, 1 + dir) * primv_values(i, j, k, 1);
                , flux_func(i, j, k, 2)
                  = consv_values(i, j, k, 1 + dir) * primv_values(i, j, k, 2);
                , flux_func(i, j, k, 3)
                  = consv_values(i, j, k, 1 + dir) * primv_values(i, j, k, 3);)

            flux_func(i, j, k, 1 + dir) += pressure;

            // energy flux
            flux_func(i, j, k, 1 + AMREX_SPACEDIM)
                = (energy + pressure) * primv_values(i, j, k, 1 + dir);
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

        FArrayBox primvfab;
        primvfab.resize(bx, AmrLevelAdv::NUM_STATE);
        // Use Elixir to ensure memory doesn't get deallocated before GPU
        // kernels can access data
        Elixir primveli = primvfab.elixir();

        const auto &primv_arr = primvfab.array(0, AmrLevelAdv::NUM_STATE);
        compute_primitive_values(time, bx, primv_arr, arr);

        ParallelFor(bx,
                    [=, &wave_speed] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        const Real velocity = std::sqrt(AMREX_D_TERM(
                            primv_arr(i, j, k, 1) * primv_arr(i, j, k, 1),
                            +primv_arr(i, j, k, 2) * primv_arr(i, j, k, 2),
                            +primv_arr(i, j, k, 3) * primv_arr(i, j, k, 3)));
                        wave_speed          = std::max(
                                     wave_speed,
                                     std::sqrt(adiabatic
                                               * primv_arr(i, j, k, 1 + AMREX_SPACEDIM)
                                               / primv_arr(i, j, k, 0)) + velocity);
                    });
    }

    ParallelDescriptor::ReduceRealMax(wave_speed);

    return wave_speed;
}