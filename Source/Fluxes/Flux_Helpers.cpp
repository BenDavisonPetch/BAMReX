#include "Fluxes/Flux_Helpers.H"
#include "AmrLevelAdv.H"
#include "Fluxes/CalcLimitingVal.H"
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
    const amrex::Array4<amrex::Real>                &half_stepped_L,
    const amrex::Array4<amrex::Real>                &half_stepped_R,
    const amrex::Array4<const amrex::Real>          &consv_values,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr, amrex::Real dt,
    const System *system, const Parm *h_parm, const Parm *d_parm)
{
    switch (AmrLevelAdv::h_parm->limiter)
    {
    case (Limiter::minbee):
        reconstruct_and_half_time_step<Limiter::minbee, dir>(
            gbx, time, half_stepped_L, half_stepped_R, consv_values, dx_arr,
            dt, system, h_parm, d_parm);
        break;
    case (Limiter::superbee):
        reconstruct_and_half_time_step<Limiter::superbee, dir>(
            gbx, time, half_stepped_L, half_stepped_R, consv_values, dx_arr,
            dt, system, h_parm, d_parm);
        break;
    case (Limiter::van_albada):
        reconstruct_and_half_time_step<Limiter::van_albada, dir>(
            gbx, time, half_stepped_L, half_stepped_R, consv_values, dx_arr,
            dt, system, h_parm, d_parm);
        break;
    case (Limiter::van_leer):
        reconstruct_and_half_time_step<Limiter::van_leer, dir>(
            gbx, time, half_stepped_L, half_stepped_R, consv_values, dx_arr,
            dt, system, h_parm, d_parm);
        break;
    default:
        AMREX_ASSERT(false); // internal error
        break;
    }
}

template <Limiter limiter, int dir>
AMREX_GPU_HOST void reconstruct_and_half_time_step(
    const amrex::Box &gbx, const amrex::Real time,
    const amrex::Array4<amrex::Real>                &half_stepped_L,
    const amrex::Array4<amrex::Real>                &half_stepped_R,
    const amrex::Array4<const amrex::Real>          &consv_values,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr, amrex::Real dt,
    const System *system, const Parm *h_parm, const Parm *d_parm)
{
    BL_PROFILE("reconstruct_and_half_time_step()");
    const Real &dx   = dx_arr[dir];
    const Box  &gbx2 = grow(gbx, 1);
    // Temporary fab
    FArrayBox tmpfab;
    // We size the temporary fab to cover the same indices as convs_values,
    // i.e. including ghost cells
    const int NSTATE = system->StateSize();
    if (h_parm->need_primitive_for_limiting)
        tmpfab.resize(gbx2, NSTATE * 3);
    else
        tmpfab.resize(gbx, NSTATE * 2);
    Elixir tmpeli = tmpfab.elixir();

    Array4<Real> flux_func_L = tmpfab.array(0, NSTATE);
    Array4<Real> flux_func_R = tmpfab.array(NSTATE, NSTATE);

    Array4<Real> Q;
    if (h_parm->need_primitive_for_limiting)
    {
        Q = tmpfab.array(NSTATE * 2, NSTATE);
        system->UtoQ(gbx2, Q, consv_values, h_parm, d_parm);
    }

    int i_off = (dir == 0) ? 1 : 0;
    int j_off = (dir == 1) ? 1 : 0;
    int k_off = (dir == 2) ? 1 : 0;

    //
    // Reconstruct
    //
    ParallelFor(
        gbx, NSTATE,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
        {
            // limiter variable index
            const Real val_C
                = calc_limiting_var(consv_values, Q, *d_parm, i, j, k, n);
            const Real val_L = calc_limiting_var(
                consv_values, Q, *d_parm, i - i_off, j - j_off, k - k_off, n);
            const Real val_R = calc_limiting_var(
                consv_values, Q, *d_parm, i + i_off, j + j_off, k + k_off, n);
            const Real numerator = val_C - val_L;
            const Real denom     = val_R - val_C;
            Real       limiter_val;
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
    system->FillFluxes(dir, time, gbx, flux_func_L, half_stepped_L, h_parm,
                       d_parm);
    system->FillFluxes(dir, time, gbx, flux_func_R, half_stepped_R, h_parm,
                       d_parm);

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
    const amrex::Array4<amrex::Real> &,
    const amrex::Array4<const amrex::Real> &,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &, amrex::Real,
    const System *, const Parm *, const Parm *);

#if AMREX_SPACEDIM >= 2
template AMREX_GPU_HOST void reconstruct_and_half_time_step<1>(
    const amrex::Box &, const amrex::Real, const amrex::Array4<amrex::Real> &,
    const amrex::Array4<amrex::Real> &,
    const amrex::Array4<const amrex::Real> &,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &, amrex::Real,
    const System *, const Parm *, const Parm *);
#endif
#if AMREX_SPACEDIM >= 3
template AMREX_GPU_HOST void reconstruct_and_half_time_step<2>(
    const amrex::Box &, const amrex::Real, const amrex::Array4<amrex::Real> &,
    const amrex::Array4<amrex::Real> &,
    const amrex::Array4<const amrex::Real> &,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &, amrex::Real,
    const System *, const Parm *, const Parm *);
#endif