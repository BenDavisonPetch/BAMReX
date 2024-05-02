#include "IMEX_Fluxes.H"
#include "AmrLevelAdv.H"
#include "IMEX_K/euler_K.H"

using namespace amrex;

AMREX_GPU_HOST
void advance_rusanov_adv(const amrex::Real /*time*/, const amrex::Box &bx,
                         amrex::GpuArray<amrex::Box, AMREX_SPACEDIM> nbx,
                         const amrex::FArrayBox                     &statein,
                         amrex::FArrayBox                           &stateout,
                         AMREX_D_DECL(amrex::FArrayBox &fx,
                                      amrex::FArrayBox &fy,
                                      amrex::FArrayBox &fz),
                         amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
                         const amrex::Real                            dt)
{
    const int     NSTATE  = AmrLevelAdv::NUM_STATE;
    constexpr int STENSIL = 1;
    const Box     gbx     = amrex::grow(bx, STENSIL);
    // Temporary fab
    FArrayBox tmpfab;
    const int ntmpcomps = 2 * NSTATE * AMREX_SPACEDIM;
    // Using the Async Arena stops temporary fab from having its destructor
    // called too early. It is equivalent on old versions of CUDA to using
    // Elixir
    tmpfab.resize(gbx, ntmpcomps, The_Async_Arena());

    GpuArray<Array4<Real>, AMREX_SPACEDIM> adv_fluxes{ AMREX_D_DECL(
        tmpfab.array(0, NSTATE), tmpfab.array(NSTATE, NSTATE),
        tmpfab.array(NSTATE * 2, NSTATE)) };

    // Temporary fluxes. Need to be declared here instead of using passed
    // fluxes as otherwise we will get a race condition
    GpuArray<Array4<Real>, AMREX_SPACEDIM> tfluxes{ AMREX_D_DECL(
        tmpfab.array(NSTATE * AMREX_SPACEDIM, NSTATE),
        tmpfab.array(NSTATE * (AMREX_SPACEDIM + 1), NSTATE),
        tmpfab.array(NSTATE * (AMREX_SPACEDIM + 2), NSTATE)) };

    // Const version of above temporary fluxes
    GpuArray<Array4<const Real>, AMREX_SPACEDIM> tfluxes_c{ AMREX_D_DECL(
        tfluxes[0], tfluxes[1], tfluxes[2]) };

    Array4<const Real> consv_in  = statein.array();
    Array4<Real>       consv_out = stateout.array();

    GpuArray<Array4<Real>, AMREX_SPACEDIM> fluxes{ AMREX_D_DECL(
        fx.array(), fy.array(), fz.array()) };

    // Compute physical fluxes
    ParallelFor(gbx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    AMREX_D_TERM(
                        adv_flux<0>(i, j, k, consv_in, adv_fluxes[0]);
                        , adv_flux<1>(i, j, k, consv_in, adv_fluxes[1]);
                        , adv_flux<2>(i, j, k, consv_in, adv_fluxes[2]);)
                });

    // Compute Rusanov fluxes
    for (int d = 0; d < AMREX_SPACEDIM; d++)
    {
        const int i_off = (d == 0) ? 1 : 0;
        const int j_off = (d == 1) ? 1 : 0;
        const int k_off = (d == 2) ? 1 : 0;
        ParallelFor(
            amrex::surroundingNodes(bx, d), NSTATE,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
            {
                const Real u_max = amrex::max(
                    fabs(consv_in(i, j, k, 1 + d) / consv_in(i, j, k, 0)),
                    fabs(consv_in(i - i_off, j - j_off, k - k_off, 1 + d)
                         / consv_in(i - i_off, j - j_off, k - k_off, 0)));
                tfluxes[d](i, j, k, n)
                    = 0.5
                      * ((adv_fluxes[d](i, j, k, n)
                          + adv_fluxes[d](i - i_off, j - j_off, k - k_off, n))
                         - u_max
                               * (consv_in(i, j, k, n)
                                  - consv_in(i - i_off, j - j_off, k - k_off,
                                             n)));
            });
    }

    // Conservative update
    AMREX_D_TERM(const Real dtdx = dt / dx[0];, const Real dtdy = dt / dx[1];
                 , const Real                              dtdz = dt / dx[2];)
    ParallelFor(bx, NSTATE,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                {
                    consv_out(i, j, k, n)
                        = consv_in(i, j, k, n)
                          - (AMREX_D_TERM(dtdx
                                              * (tfluxes_c[0](i + 1, j, k, n)
                                                 - tfluxes_c[0](i, j, k, n)),
                                          +dtdy
                                              * (tfluxes_c[1](i, j + 1, k, n)
                                                 - tfluxes_c[1](i, j, k, n)),
                                          +dtdz
                                              * (tfluxes_c[2](i, j, k + 1, n)
                                                 - tfluxes_c[2](i, j, k, n))));
                });

    // Copy temporary fluxes onto potentially smaller nodal tile
    for (int d = 0; d < AMREX_SPACEDIM; d++)
        ParallelFor(nbx[d], NSTATE,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    { fluxes[d](i, j, k, n) = tfluxes_c[d](i, j, k, n); });
}