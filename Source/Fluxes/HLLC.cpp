#include "Fluxes/HLLC.H"
#include "AmrLevelAdv.H"
#include "Fluxes/Flux_Helpers.H"
#include "Fluxes/Fluxes.H"
#include "Fluxes/Limiters.H"

#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_Extension.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

using namespace amrex;

template <int dir>
AMREX_GPU_HOST void compute_HLLC_flux_LR(
    amrex::Real /*time*/, const amrex::Box &bx,
    const amrex::Array4<amrex::Real>       &flux,
    const amrex::Array4<const amrex::Real> &consv_values_L,
    const amrex::Array4<const amrex::Real> &consv_values_R,
    const amrex::Real adiabatic_L, const amrex::Real adiabatic_R,
    const amrex::Real epsilon,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const & /*dx_arr*/,
    amrex::Real /*dt*/)
{
    const int NSTATE = AmrLevelAdv::NUM_STATE;
    // const Real         dx     = dx_arr[dir];

    const auto &flux_bx = surroundingNodes(bx, dir);

    const int i_off = (dir == 0) ? 1 : 0;
    const int j_off = (dir == 1) ? 1 : 0;
    const int k_off = (dir == 2) ? 1 : 0;

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

            // Do simple cases that don't require extra computation first
            if (0 <= speed_L)
            {
                const auto euler_flux_L
                    = euler_flux<dir>(consv_values_L, adiabatic_L, epsilon,
                                      i - i_off, j - j_off, k - k_off);
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
                // the below replaces primv_L with the intermediate state
                // (denoted star L)
                HLLC_interm_state_replace<dir>(
                    primv_L,
                    consv_values_L(i - i_off, j - j_off, k - k_off,
                                   1 + AMREX_SPACEDIM),
                    speed_L, speed_interm, epsilon);
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
                // the below replaces primv_L with the intermediate state
                // (denoted star L)
                HLLC_interm_state_replace<dir>(
                    primv_R, consv_values_R(i, j, k, 1 + AMREX_SPACEDIM),
                    speed_R, speed_interm, epsilon);
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

template AMREX_GPU_HOST void compute_HLLC_flux_LR<0>(
    amrex::Real, const amrex::Box &, const amrex::Array4<amrex::Real> &,
    const amrex::Array4<const amrex::Real> &,
    const amrex::Array4<const amrex::Real> &, const amrex::Real,
    const amrex::Real, const amrex::Real,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &, amrex::Real);

#if AMREX_SPACEDIM >= 2
template AMREX_GPU_HOST void compute_HLLC_flux_LR<1>(
    amrex::Real, const amrex::Box &, const amrex::Array4<amrex::Real> &,
    const amrex::Array4<const amrex::Real> &,
    const amrex::Array4<const amrex::Real> &, const amrex::Real,
    const amrex::Real, const amrex::Real,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &, amrex::Real);
#endif
#if AMREX_SPACEDIM >= 3
template AMREX_GPU_HOST void compute_HLLC_flux_LR<2>(
    amrex::Real, const amrex::Box &, const amrex::Array4<amrex::Real> &,
    const amrex::Array4<const amrex::Real> &,
    const amrex::Array4<const amrex::Real> &, const amrex::Real,
    const amrex::Real, const amrex::Real,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &, amrex::Real);
#endif