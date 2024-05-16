#include "AmrLevelAdv.H"
#include "Fluxes.H"
#include "HLLC.H"

using namespace amrex;

template <int dir>
AMREX_GPU_HOST void compute_corrected_HLLC_flux(
    amrex::Real time, const amrex::Box &bx,
    const amrex::Array4<amrex::Real>                &flux,
    const amrex::Array4<const amrex::Real>          &consv,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr, amrex::Real dt)
{
    compute_HLLC_flux<dir>(time, bx, flux, consv, dx_arr, dt);

    // const Real         dx     = dx_arr[dir];
    const Real adia = AmrLevelAdv::h_prob_parm->adiabatic;
    const Real eps  = AmrLevelAdv::h_prob_parm->epsilon;

    const auto &flux_bx = surroundingNodes(bx, dir);

    const int i_off = (dir == 0) ? 1 : 0;
    const int j_off = (dir == 1) ? 1 : 0;
    const int k_off = (dir == 2) ? 1 : 0;

    ParallelFor(
        flux_bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            Real S_L;
            Real S_R;
            Real S_interm;
            estimate_wave_speed_HLLC_Toro<dir>(i, j, k, consv, consv, adia,
                                               adia, eps, S_L, S_R, S_interm);

            const Real u_LL
                = consv(i - i_off * 2, j - j_off * 2, k - k_off * 2, 1 + dir)
                  / consv(i - i_off * 2, j - j_off * 2, k - k_off * 2, 0);
            const Real u_L = consv(i - i_off, j - j_off, k - k_off, 1 + dir)
                             / consv(i - i_off, j - j_off, k - k_off, 0);
            const Real u_R  = consv(i, j, k, 1 + dir) / consv(i, j, k, 0);
            const Real u_RR = consv(i + i_off, j + j_off, k + k_off, 1 + dir)
                              / consv(i + i_off, j + j_off, k + k_off, 0);
            const Real c_LL
                = sound_speed(primv_from_consv(consv, adia, eps, i - i_off * 2,
                                               j - j_off * 2, k - k_off * 2),
                              adia, eps);
            const Real c_L
                = sound_speed(primv_from_consv(consv, adia, eps, i - i_off,
                                               j - j_off, k - k_off),
                              adia, eps);
            const Real c_R = sound_speed(
                primv_from_consv(consv, adia, eps, i, j, k), adia, eps);
            const Real c_RR
                = sound_speed(primv_from_consv(consv, adia, eps, i + i_off,
                                               j + j_off, k + k_off),
                              adia, eps);
            const Real k_L = ((u_LL - c_LL) * (u_L - c_L) < 0
                              || (u_L - c_L) * (u_R - c_L) < 0
                              || (u_LL + c_LL) * (u_L + c_L) < 0
                              || (u_L + c_L) * (u_R + c_L) < 0)
                                 ? 1
                                 : 0;
            const Real k_R = ((u_RR - c_RR) * (u_R - c_R) < 0
                              || (u_L - c_L) * (u_R - c_L) < 0
                              || (u_RR + c_RR) * (u_R + c_R) < 0
                              || (u_L + c_L) * (u_R + c_L) < 0)
                                 ? 1
                                 : 0;
            const Real M   = max(u_L / c_L, u_R / c_R);
            // g(M) - 1
            const Real gm1 = (k_L == 0 && k_R == 0) ? min(1., M) - 1 : 0;

            const Real rho_L = consv(i - i_off, j - j_off, k - k_off, 0);
            const Real rho_R = consv(i, j, k, 0);
            const Real prefactor
                = (rho_L * rho_R * (S_L - u_L) * (S_R - u_R))
                  / (rho_R * (S_R - u_R) - rho_L * (S_L - u_L)) * (u_R - u_L);

            flux(i, j, k, 1 + dir) += gm1 * prefactor;
            flux(i, j, k, 1 + AMREX_SPACEDIM) += gm1 * prefactor * S_interm;
        });
}

template AMREX_GPU_HOST void compute_corrected_HLLC_flux<0>(
    amrex::Real time, const amrex::Box &bx,
    const amrex::Array4<amrex::Real>                &flux,
    const amrex::Array4<const amrex::Real>          &consv,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr, amrex::Real dt);

#if AMREX_SPACEDIM >= 2
template AMREX_GPU_HOST void compute_corrected_HLLC_flux<1>(
    amrex::Real time, const amrex::Box &bx,
    const amrex::Array4<amrex::Real>                &flux,
    const amrex::Array4<const amrex::Real>          &consv,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr, amrex::Real dt);
#endif

#if AMREX_SPACEDIM >= 3
template AMREX_GPU_HOST void compute_corrected_HLLC_flux<0>(
    amrex::Real time, const amrex::Box &bx,
    const amrex::Array4<amrex::Real>                &flux,
    const amrex::Array4<const amrex::Real>          &consv,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr, amrex::Real dt);
#endif