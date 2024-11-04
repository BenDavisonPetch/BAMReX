#include "Fluxes/HLLC.H"
#include "AmrLevelAdv.H"
#include "Fluxes/Flux_Helpers.H"
#include "Fluxes/Fluxes.H"
#include "Fluxes/Limiters.H"
#include "System/Plasma/Plasma_K.H"

#include <AMReX_Arena.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_Extension.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

using namespace amrex;

// This way of having different routines for euler and plasma is messy but it
// avoids some serious complications with polymorphism and GPU code
template <int dir>
AMREX_GPU_HOST void compute_HLLC_flux_LR_euler(
    amrex::Real time, const amrex::Box &bx,
    const amrex::Array4<amrex::Real>                &flux,
    const amrex::Array4<const amrex::Real>          &consv_values_L,
    const amrex::Array4<const amrex::Real>          &consv_values_R,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr, amrex::Real dt,
    const System *system, const Parm *h_parm, const Parm *d_parm);

template <int dir>
AMREX_GPU_HOST void compute_HLLC_flux_LR_plasma(
    amrex::Real time, const amrex::Box &bx,
    const amrex::Array4<amrex::Real>                &flux,
    const amrex::Array4<const amrex::Real>          &consv_values_L,
    const amrex::Array4<const amrex::Real>          &consv_values_R,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr, amrex::Real dt,
    const System *system, const Parm *h_parm, const Parm *d_parm);

template <int dir>
AMREX_GPU_HOST void
compute_HLLC_flux_LR(amrex::Real time, const amrex::Box &bx,
                     const amrex::Array4<amrex::Real>       &flux,
                     const amrex::Array4<const amrex::Real> &consv_values_L,
                     const amrex::Array4<const amrex::Real> &consv_values_R,
                     amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr,
                     amrex::Real dt, const System *system, const Parm *h_parm,
                     const Parm *d_parm)
{
    BL_PROFILE("compute_HLLC_flux_LR()");
    switch (system->GetSystemType())
    {
    case (SystemType::fluid):
        compute_HLLC_flux_LR_euler<dir>(time, bx, flux, consv_values_L,
                                        consv_values_R, dx_arr, dt, system,
                                        h_parm, d_parm);
        return;
    case (SystemType::plasma):
        compute_HLLC_flux_LR_plasma<dir>(time, bx, flux, consv_values_L,
                                         consv_values_R, dx_arr, dt, system,
                                         h_parm, d_parm);
        return;
    }
}

template <int dir>
AMREX_GPU_HOST void compute_HLLC_flux_LR_euler(
    amrex::Real /*time*/, const amrex::Box &bx,
    const amrex::Array4<amrex::Real>       &flux,
    const amrex::Array4<const amrex::Real> &consv_values_L,
    const amrex::Array4<const amrex::Real> &consv_values_R,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const & /*dx_arr*/,
    amrex::Real /*dt*/, const System *system, const Parm *h_parm,
    const Parm *d_parm)
{
    const int NSTATE = system->StateSize();
    // const Real         dx     = dx_arr[dir];

    const auto &flux_bx = surroundingNodes(bx, dir);

    const int i_off = (dir == 0) ? 1 : 0;
    const int j_off = (dir == 1) ? 1 : 0;
    const int k_off = (dir == 2) ? 1 : 0;

    AMREX_ALWAYS_ASSERT(system->GetSystemType() == SystemType::fluid);
    ignore_unused(d_parm);

    const Real adiabatic_L = h_parm->adiabatic;
    const Real adiabatic_R = h_parm->adiabatic;
    const Real epsilon     = h_parm->epsilon;

    ParallelFor(
        flux_bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            Real speed_L;
            Real speed_R;
            Real speed_interm;
            estimate_wave_speed_HLLC_Toro<dir>(
                i, j, k, consv_values_L, consv_values_R, adiabatic_L,
                adiabatic_R, epsilon, speed_L, speed_R, speed_interm);

            if (0 <= speed_L)
            {
                const auto euler_flux_L
                    = euler_flux<dir>(consv_values_L, adiabatic_L, epsilon,
                                      i - i_off, j - j_off, k - k_off);
                AMREX_PRAGMA_SIMD
                for (int n = 0; n < NSTATE; ++n)
                {
                    flux(i, j, k, n) = euler_flux_L[n];
                }
            }
            else if (speed_L <= 0 && 0 <= speed_interm)
            {
                const auto euler_flux_L
                    = euler_flux<dir>(consv_values_L, adiabatic_L, epsilon,
                                      i - i_off, j - j_off, k - k_off);
                auto primv_L
                    = primv_from_consv(consv_values_L, adiabatic_L, epsilon,
                                       i - i_off, j - j_off, k - k_off);
                // the below replaces primv_L with the intermediate
                // state (denoted star L)
                HLLC_interm_state_replace<dir>(
                    primv_L,
                    consv_values_L(i - i_off, j - j_off, k - k_off,
                                   1 + AMREX_SPACEDIM),
                    speed_L, speed_interm, epsilon);
                AMREX_PRAGMA_SIMD
                for (int n = 0; n < NSTATE; ++n)
                {
                    flux(i, j, k, n)
                        = euler_flux_L[n]
                          + speed_L
                                * (primv_L[n]
                                   - consv_values_L(i - i_off, j - j_off,
                                                    k - k_off, n));
                }
            }
            else if (speed_interm <= 0 && 0 <= speed_R)
            {
                const auto euler_flux_R = euler_flux<dir>(
                    consv_values_R, adiabatic_R, epsilon, i, j, k);
                auto primv_R = primv_from_consv(consv_values_R, adiabatic_R,
                                                epsilon, i, j, k);
                // the below replaces primv_L with the intermediate
                // state (denoted star L)
                HLLC_interm_state_replace<dir>(
                    primv_R, consv_values_R(i, j, k, 1 + AMREX_SPACEDIM),
                    speed_R, speed_interm, epsilon);
                AMREX_PRAGMA_SIMD
                for (int n = 0; n < NSTATE; ++n)
                {
                    flux(i, j, k, n)
                        = euler_flux_R[n]
                          + speed_R
                                * (primv_R[n] - consv_values_R(i, j, k, n));
                }
            }
            else if (0 >= speed_R)
            {
                const auto euler_flux_R = euler_flux<dir>(
                    consv_values_R, adiabatic_R, epsilon, i, j, k);
                AMREX_PRAGMA_SIMD
                for (int n = 0; n < NSTATE; ++n)
                {
                    flux(i, j, k, n) = euler_flux_R[n];
                }
            }
            else
            {
                AMREX_ASSERT(false); // internal error
            }
        });
}

template <int dir>
AMREX_GPU_HOST void compute_HLLC_flux_LR_plasma(
    amrex::Real /*time*/, const amrex::Box &bx,
    const amrex::Array4<amrex::Real>       &flux,
    const amrex::Array4<const amrex::Real> &U_L,
    const amrex::Array4<const amrex::Real> &U_R,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const & /*dx_arr*/,
    amrex::Real /*dt*/, const System *system, const Parm * /*h_parm*/,
    const Parm *d_parm)
{
    const int NSTATE = system->StateSize();
    // const Real         dx     = dx_arr[dir];

    const auto &flux_bx = surroundingNodes(bx, dir);

    constexpr int i_off = (dir == 0) ? 1 : 0;
    constexpr int j_off = (dir == 1) ? 1 : 0;
    constexpr int k_off = (dir == 2) ? 1 : 0;

    ParallelFor(
        flux_bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            const auto Q_L
                = mhd_ctop_noB(i - i_off, j - j_off, k - k_off, U_L, *d_parm);
            const auto Q_R = mhd_ctop_noB(i, j, k, U_R, *d_parm);

            Real speed_L;
            Real speed_R;
            Real speed_interm;
            mhd_HLLC_speeds<dir>(i, j, k, U_L, U_R, Q_L, Q_R, *d_parm, speed_L,
                                 speed_R, speed_interm);

            // Do simple cases that don't require extra computation
            // first
            if (0 <= speed_L)
            {
                const auto mhd_flux_L
                    = mhd_flux<dir>(i - i_off, j - j_off, k - k_off, U_L, Q_L);
                AMREX_PRAGMA_SIMD
                for (int n = 0; n < NSTATE; ++n)
                {
                    flux(i, j, k, n) = mhd_flux_L[n];
                }
            }
            else if (speed_L <= 0 && 0 <= speed_interm)
            {
                const auto mhd_flux_L
                    = mhd_flux<dir>(i - i_off, j - j_off, k - k_off, U_L, Q_L);
                const auto mhd_flux_R = mhd_flux<dir>(i, j, k, U_R, Q_R);
                const auto B_HLL
                    = mhd_HLL_B<dir>(i, j, k, U_L, U_R, mhd_flux_L, mhd_flux_R,
                                     speed_L, speed_R);
                const auto U_HLLC = mhd_HLLC_U_star<dir>(
                    i - i_off, j - j_off, k - k_off, U_L, Q_L, B_HLL, speed_L,
                    speed_interm);
                AMREX_PRAGMA_SIMD
                for (int n = 0; n < NSTATE; ++n)
                {
                    flux(i, j, k, n)
                        = mhd_flux_L[n]
                          + speed_L
                                * (U_HLLC[n]
                                   - U_L(i - i_off, j - j_off, k - k_off, n));
                }
            }
            else if (speed_interm <= 0 && 0 <= speed_R)
            {
                const auto mhd_flux_L
                    = mhd_flux<dir>(i - i_off, j - j_off, k - k_off, U_L, Q_L);
                const auto mhd_flux_R = mhd_flux<dir>(i, j, k, U_R, Q_R);
                const auto B_HLL
                    = mhd_HLL_B<dir>(i, j, k, U_L, U_R, mhd_flux_L, mhd_flux_R,
                                     speed_L, speed_R);
                const auto U_HLLC = mhd_HLLC_U_star<dir>(
                    i, j, k, U_R, Q_R, B_HLL, speed_R, speed_interm);
                AMREX_PRAGMA_SIMD
                for (int n = 0; n < NSTATE; ++n)
                {
                    flux(i, j, k, n)
                        = mhd_flux_R[n]
                          + speed_R * (U_HLLC[n] - U_R(i, j, k, n));
                }
            }
            else if (0 >= speed_R)
            {
                const auto mhd_flux_R = mhd_flux<dir>(i, j, k, U_R, Q_R);
                AMREX_PRAGMA_SIMD
                for (int n = 0; n < NSTATE; ++n)
                {
                    flux(i, j, k, n) = mhd_flux_R[n];
                }
            }
            else
            {
                AMREX_ASSERT(false); // internal error
            }
        });
}

template AMREX_GPU_HOST void compute_HLLC_flux_LR<0>(
    amrex::Real, const amrex::Box &, const amrex::Array4<amrex::Real> &,
    const amrex::Array4<const amrex::Real> &,
    const amrex::Array4<const amrex::Real> &,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &, amrex::Real,
    const System *, const Parm *, const Parm *);

#if AMREX_SPACEDIM >= 2
template AMREX_GPU_HOST void compute_HLLC_flux_LR<1>(
    amrex::Real, const amrex::Box &, const amrex::Array4<amrex::Real> &,
    const amrex::Array4<const amrex::Real> &,
    const amrex::Array4<const amrex::Real> &,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &, amrex::Real,
    const System *, const Parm *, const Parm *);
#endif
#if AMREX_SPACEDIM >= 3
template AMREX_GPU_HOST void compute_HLLC_flux_LR<2>(
    amrex::Real, const amrex::Box &, const amrex::Array4<amrex::Real> &,
    const amrex::Array4<const amrex::Real> &,
    const amrex::Array4<const amrex::Real> &,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &, amrex::Real,
    const System *, const Parm *, const Parm *);
#endif