#include "Fluxes/Flux_Helpers.H"
#include "AmrLevelAdv.H"
#include "Fluxes/Fluxes.H"
#include "Fluxes/Limiters.H"

#include <AMReX_Box.H>
#include <AMReX_Extension.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

using namespace amrex;

template <int dir>
AMREX_GPU_HOST void reconstruct_and_half_time_step(
    const amrex::Box &gbx, const amrex::Real time,
    const amrex::Array4<amrex::Real> &half_stepped_L,
    const amrex::Array4<amrex::Real> &half_stepped_R,
    const amrex::Real adiabatic_L, const amrex::Real adiabatic_R,
    const amrex::Real                                epsilon,
    const amrex::Array4<const amrex::Real>          &consv_values,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr, amrex::Real dt)
{
    switch (AmrLevelAdv::h_prob_parm->limiter)
    {
    case (minbee_limiter):
        reconstruct_and_half_time_step<minbee_limiter, dir>(
            gbx, time, half_stepped_L, half_stepped_R, adiabatic_L,
            adiabatic_R, epsilon, consv_values, dx_arr, dt);
        break;
    case (superbee_limiter):
        reconstruct_and_half_time_step<superbee_limiter, dir>(
            gbx, time, half_stepped_L, half_stepped_R, adiabatic_L,
            adiabatic_R, epsilon, consv_values, dx_arr, dt);
        break;
    case (van_albada_limiter):
        reconstruct_and_half_time_step<van_albada_limiter, dir>(
            gbx, time, half_stepped_L, half_stepped_R, adiabatic_L,
            adiabatic_R, epsilon, consv_values, dx_arr, dt);
        break;
    case (van_leer_limiter):
        reconstruct_and_half_time_step<van_leer_limiter, dir>(
            gbx, time, half_stepped_L, half_stepped_R, adiabatic_L,
            adiabatic_R, epsilon, consv_values, dx_arr, dt);
        break;
    default:
        AMREX_ASSERT(false); // internal error
        break;
    }
}

template <Limiter limiter, int dir>
AMREX_GPU_HOST void reconstruct_and_half_time_step(
    const amrex::Box &gbx, const amrex::Real time,
    const amrex::Array4<amrex::Real> &half_stepped_L,
    const amrex::Array4<amrex::Real> &half_stepped_R,
    const amrex::Real adiabatic_L, const amrex::Real adiabatic_R,
    const amrex::Real                                epsilon,
    const amrex::Array4<const amrex::Real>          &consv_values,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr, amrex::Real dt)
{
    const Real &dx = dx_arr[dir];
    // Temporary fab
    FArrayBox tmpfab;
    // We size the temporary fab to cover the same indices as convs_values,
    // i.e. including ghost cells
    const int NSTATE = AmrLevelAdv::NUM_STATE;
    tmpfab.resize(gbx, NSTATE * 2);
    Elixir tmpeli = tmpfab.elixir();

    Array4<Real> flux_func_L = tmpfab.array(0, NSTATE);
    Array4<Real> flux_func_R = tmpfab.array(NSTATE, NSTATE);

    int i_off = (dir == 0) ? 1 : 0;
    int j_off = (dir == 1) ? 1 : 0;
    int k_off = (dir == 2) ? 1 : 0;

    const auto &limiter_indices = AmrLevelAdv::h_prob_parm->limiter_indices;

    //
    // Reconstruct
    //
    ParallelFor(gbx, NSTATE,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                {
                    // Calculate slope ratio
                    // This is technically recomputed for every component of
                    // the variables vector but the alternative isn't great for
                    // cache locality

                    // limiter variable index
                    int        iq = limiter_indices[n];
                    const Real numerator
                        = consv_values(i, j, k, iq)
                          - consv_values(i - i_off, j - j_off, k - k_off, iq);
                    const Real denom
                        = consv_values(i + i_off, j + j_off, k + k_off, iq)
                          - consv_values(i, j, k, iq);
                    Real limiter_val;
                    if (numerator == 0. && denom == 0.)
                        limiter_val = slope_limit<limiter>(1);
                    else if (denom == 0)
                        limiter_val = 0;
                    else
                        limiter_val = slope_limit<limiter>(numerator / denom);

                    // get delta
                    const Real omega = 0;
                    const Real delta_left
                        = consv_values(i, j, k, n)
                          - consv_values(i - i_off, j - j_off, k - k_off, n);
                    const Real delta_right
                        = consv_values(i + i_off, j + j_off, k + k_off, n)
                          - consv_values(i, j, k, n);
                    const Real delta = 0.5 * (1 + omega) * delta_left
                                       + 0.5 * (1 - omega) * delta_right;

                    half_stepped_L(i, j, k, n)
                        = consv_values(i, j, k, n) - 0.5 * limiter_val * delta;
                    half_stepped_R(i, j, k, n)
                        = consv_values(i, j, k, n) + 0.5 * limiter_val * delta;
                });

    //
    // Half time-step update
    //
    compute_flux_function<dir>(time, gbx, flux_func_L, half_stepped_L,
                               adiabatic_L, epsilon);
    compute_flux_function<dir>(time, gbx, flux_func_R, half_stepped_R,
                               adiabatic_R, epsilon);

    double hdtdx = 0.5 * dt / dx;

    ParallelFor(
        gbx, NSTATE,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
        {
            half_stepped_L(i, j, k, n)
                -= hdtdx * (flux_func_R(i, j, k, n) - flux_func_L(i, j, k, n));
            half_stepped_R(i, j, k, n)
                -= hdtdx * (flux_func_R(i, j, k, n) - flux_func_L(i, j, k, n));
        });
}

template AMREX_GPU_HOST void reconstruct_and_half_time_step<0>(
    const amrex::Box &, const amrex::Real, const amrex::Array4<amrex::Real> &,
    const amrex::Array4<amrex::Real> &, const amrex::Real, const amrex::Real,
    const amrex::Real, const amrex::Array4<const amrex::Real> &,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &, amrex::Real);

#if AMREX_SPACEDIM >= 2
template AMREX_GPU_HOST void reconstruct_and_half_time_step<1>(
    const amrex::Box &, const amrex::Real, const amrex::Array4<amrex::Real> &,
    const amrex::Array4<amrex::Real> &, const amrex::Real, const amrex::Real,
    const amrex::Real, const amrex::Array4<const amrex::Real> &,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &, amrex::Real);
#endif
#if AMREX_SPACEDIM >= 3
template AMREX_GPU_HOST void reconstruct_and_half_time_step<2>(
    const amrex::Box &, const amrex::Real, const amrex::Array4<amrex::Real> &,
    const amrex::Array4<amrex::Real> &, const amrex::Real, const amrex::Real,
    const amrex::Real, const amrex::Array4<const amrex::Real> &,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &, amrex::Real);
#endif