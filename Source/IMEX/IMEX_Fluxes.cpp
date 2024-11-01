#include "IMEX_Fluxes.H"
#include "AmrLevelAdv.H"
#include "Fluxes/Limiters.H"
#include "IMEX_K/euler_K.H"

#include <AMReX_Arena.H>

using namespace amrex;

AMREX_GPU_HOST
void advance_rusanov_adv(const amrex::Real /*time*/, const amrex::Box &bx,
                         amrex::GpuArray<amrex::Box, AMREX_SPACEDIM> nbx,
                         const amrex::FArrayBox                     &statein,
                         const amrex::FArrayBox &state_toupdate,
                         amrex::FArrayBox       &stateout,
                         AMREX_D_DECL(amrex::FArrayBox &fx,
                                      amrex::FArrayBox &fy,
                                      amrex::FArrayBox &fz),
                         amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
                         const amrex::Real                            dt)
{
    BL_PROFILE("advance_rusanov_adv()");
    const int     NSTATE  = EULER_NCOMP;
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
    Array4<const Real> toupdate  = state_toupdate.const_array();

    GpuArray<Array4<Real>, AMREX_SPACEDIM> fluxes{ AMREX_D_DECL(
        fx.array(), fy.array(), fz.array()) };

    const Parm *parm = AmrLevelAdv::d_parm;

    // Compute physical fluxes
    ParallelFor(gbx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    AMREX_D_TERM(
                        adv_flux<0>(i, j, k, consv_in, adv_fluxes[0], *parm);
                        , adv_flux<1>(i, j, k, consv_in, adv_fluxes[1], *parm);
                        ,
                        adv_flux<2>(i, j, k, consv_in, adv_fluxes[2], *parm);)
                });

    // Compute Rusanov fluxes using statein
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

    // Apply fluxes to state_toupdate
    AMREX_D_TERM(const Real dtdx = dt / dx[0];, const Real dtdy = dt / dx[1];
                 , const Real                              dtdz = dt / dx[2];)
    ParallelFor(bx, NSTATE,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                {
                    consv_out(i, j, k, n)
                        = toupdate(i, j, k, n)
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

AMREX_GPU_HOST
void advance_MUSCL_rusanov_adv(const amrex::Real, const amrex::Box &bx,
                               amrex::GpuArray<amrex::Box, AMREX_SPACEDIM> nbx,
                               const amrex::FArrayBox &statein,
                               const amrex::FArrayBox &state_toupdate,
                               amrex::FArrayBox       &stateout,
                               AMREX_D_DECL(amrex::FArrayBox &fx,
                                            amrex::FArrayBox &fy,
                                            amrex::FArrayBox &fz),
                               amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
                               const amrex::Real dt, bool local_time_step)
{
    BL_PROFILE("advance_MUSCL_rusanov_adv()");
    const Box gbx = amrex::grow(bx, 1);

    constexpr int ntmpcomps = EULER_NCOMP * 5 * AMREX_SPACEDIM;
    FArrayBox     tmpfab(gbx, ntmpcomps, The_Async_Arena());
    // Using the Async Arena stops temporary fab from having its destructor
    // called too early. It is equivalent on old versions of CUDA to using
    // Elixir

    GpuArray<Array4<Real>, AMREX_SPACEDIM> state_L{ AMREX_D_DECL(
        tmpfab.array(0 * AMREX_SPACEDIM * EULER_NCOMP, EULER_NCOMP),
        tmpfab.array((0 * AMREX_SPACEDIM + 1) * EULER_NCOMP, EULER_NCOMP),
        tmpfab.array((0 * AMREX_SPACEDIM + 2) * EULER_NCOMP, EULER_NCOMP)) };

    GpuArray<Array4<Real>, AMREX_SPACEDIM> state_R{ AMREX_D_DECL(
        tmpfab.array(1 * AMREX_SPACEDIM * EULER_NCOMP, EULER_NCOMP),
        tmpfab.array((1 * AMREX_SPACEDIM + 1) * EULER_NCOMP, EULER_NCOMP),
        tmpfab.array((1 * AMREX_SPACEDIM + 2) * EULER_NCOMP, EULER_NCOMP)) };

    GpuArray<Array4<Real>, AMREX_SPACEDIM> adv_fluxes_L{ AMREX_D_DECL(
        tmpfab.array(2 * AMREX_SPACEDIM * EULER_NCOMP, EULER_NCOMP),
        tmpfab.array((2 * AMREX_SPACEDIM + 1) * EULER_NCOMP, EULER_NCOMP),
        tmpfab.array((2 * AMREX_SPACEDIM + 2) * EULER_NCOMP, EULER_NCOMP)) };

    GpuArray<Array4<Real>, AMREX_SPACEDIM> adv_fluxes_R{ AMREX_D_DECL(
        tmpfab.array(3 * AMREX_SPACEDIM * EULER_NCOMP, EULER_NCOMP),
        tmpfab.array((3 * AMREX_SPACEDIM + 1) * EULER_NCOMP, EULER_NCOMP),
        tmpfab.array((3 * AMREX_SPACEDIM + 2) * EULER_NCOMP, EULER_NCOMP)) };

    // Temporary fluxes. Need to be declared here instead of using passed
    // fluxes as otherwise we will get a race condition
    GpuArray<Array4<Real>, AMREX_SPACEDIM> tfluxes{ AMREX_D_DECL(
        tmpfab.array((4 * AMREX_SPACEDIM) * EULER_NCOMP, EULER_NCOMP),
        tmpfab.array((4 * AMREX_SPACEDIM + 1) * EULER_NCOMP, EULER_NCOMP),
        tmpfab.array((4 * AMREX_SPACEDIM + 2) * EULER_NCOMP, EULER_NCOMP)) };

    // Const version of above temporary fluxes
    GpuArray<Array4<const Real>, AMREX_SPACEDIM> tfluxes_c{ AMREX_D_DECL(
        tfluxes[0], tfluxes[1], tfluxes[2]) };

    const auto &in        = statein.const_array();
    const auto &consv_out = stateout.array();
    const auto &toupdate  = state_toupdate.const_array();

    GpuArray<Array4<Real>, AMREX_SPACEDIM> fluxes{ AMREX_D_DECL(
        fx.array(), fy.array(), fz.array()) };

    AMREX_D_TERM(const Real dtdx = dt / dx[0];, const Real dtdy = dt / dx[1];
                 , const Real                              dtdz = dt / dx[2];)
    const GpuArray<Real, AMREX_SPACEDIM> hdtdxarr{ AMREX_D_DECL(
        0.5 * dtdx, 0.5 * dtdy, 0.5 * dtdz) };

    const Parm *parm = AmrLevelAdv::d_parm;

    // Now we can do actual computation

    // Reconstruct
    BL_PROFILE_VAR("advance_MUSCL_rusanov_adv::reconstruction",
                   preconstruction);
    ParallelFor(gbx, EULER_NCOMP,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                {
                    const Real delta_L_x = in(i, j, k, n) - in(i - 1, j, k, n);
                    const Real delta_R_x = in(i + 1, j, k, n) - in(i, j, k, n);
                    const Real hdelta_x  = 0.5 * minmod(delta_L_x, delta_R_x);
                    state_L[0](i, j, k, n) = in(i, j, k, n) - hdelta_x;
                    state_R[0](i, j, k, n) = in(i, j, k, n) + hdelta_x;

#if AMREX_SPACEDIM >= 2
                    const Real delta_L_y = in(i, j, k, n) - in(i, j - 1, k, n);
                    const Real delta_R_y = in(i, j + 1, k, n) - in(i, j, k, n);
                    const Real hdelta_y  = 0.5 * minmod(delta_L_y, delta_R_y);
                    state_L[1](i, j, k, n) = in(i, j, k, n) - hdelta_y;
                    state_R[1](i, j, k, n) = in(i, j, k, n) + hdelta_y;
#endif
#if AMREX_SPACEDIM == 3
                    const Real delta_L_z = in(i, j, k, n) - in(i, j, k - 1, n);
                    const Real delta_R_z = in(i, j, k + 1, n) - in(i, j, k, n);
                    const Real hdelta_z  = 0.5 * minmod(delta_L_z, delta_R_z);
                    state_L[2](i, j, k, n) = in(i, j, k, n) - hdelta_z;
                    state_R[2](i, j, k, n) = in(i, j, k, n) + hdelta_z;
#endif
                });
    BL_PROFILE_VAR_STOP(preconstruction);

    if (local_time_step)
    {
        // Compute physical fluxes
        ParallelFor(
            gbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                AMREX_D_TERM(
                    adv_flux<0>(i, j, k, state_L[0], adv_fluxes_L[0], *parm);
                    , adv_flux<1>(i, j, k, state_L[1], adv_fluxes_L[1], *parm);
                    ,
                    adv_flux<2>(i, j, k, state_L[2], adv_fluxes_L[2], *parm);)
                AMREX_D_TERM(
                    adv_flux<0>(i, j, k, state_R[0], adv_fluxes_R[0], *parm);
                    , adv_flux<1>(i, j, k, state_R[1], adv_fluxes_R[1], *parm);
                    ,
                    adv_flux<2>(i, j, k, state_R[2], adv_fluxes_R[2], *parm);)
            });

        // Local update
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            ParallelFor(gbx, EULER_NCOMP,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                        {
                            const Real local_update
                                = hdtdxarr[d]
                                  * (adv_fluxes_R[d](i, j, k, n)
                                     - adv_fluxes_L[d](i, j, k, n));
                            state_L[d](i, j, k, n) -= local_update;
                            state_R[d](i, j, k, n) -= local_update;
                        });
        }
    }

    // Compute physical fluxes
    BL_PROFILE_VAR("advance_MUSCL_rusanov_adv::physical_fluxes", pfluxes);
    ParallelFor(
        gbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            AMREX_D_TERM(
                adv_flux<0>(i, j, k, state_L[0], adv_fluxes_L[0], *parm);
                , adv_flux<1>(i, j, k, state_L[1], adv_fluxes_L[1], *parm);
                , adv_flux<2>(i, j, k, state_L[2], adv_fluxes_L[2], *parm);)
            AMREX_D_TERM(
                adv_flux<0>(i, j, k, state_R[0], adv_fluxes_R[0], *parm);
                , adv_flux<1>(i, j, k, state_R[1], adv_fluxes_R[1], *parm);
                , adv_flux<2>(i, j, k, state_R[2], adv_fluxes_R[2], *parm);)
        });
    BL_PROFILE_VAR_STOP(pfluxes);

    // Compute Rusanov fluxes using reconstructed states
    BL_PROFILE_VAR("advance_MUSCL_rusanov_adv::flux_calculation", pfluxcalc);
    for (int d = 0; d < AMREX_SPACEDIM; d++)
    {
        const int i_off = (d == 0) ? 1 : 0;
        const int j_off = (d == 1) ? 1 : 0;
        const int k_off = (d == 2) ? 1 : 0;
        ParallelFor(
            amrex::surroundingNodes(bx, d), EULER_NCOMP,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
            {
                const Real u_max = amrex::max(
                    fabs(state_L[d](i, j, k, 1 + d) / state_L[d](i, j, k, 0)),
                    fabs(state_R[d](i - i_off, j - j_off, k - k_off, 1 + d)
                         / state_R[d](i - i_off, j - j_off, k - k_off, 0)));

                tfluxes[d](i, j, k, n)
                    = 0.5
                      * ((adv_fluxes_L[d](i, j, k, n)
                          + adv_fluxes_R[d](i - i_off, j - j_off, k - k_off,
                                            n))
                         - u_max
                               * (state_L[d](i, j, k, n)
                                  - state_R[d](i - i_off, j - j_off, k - k_off,
                                               n)));
            });
    }
    BL_PROFILE_VAR_STOP(pfluxcalc);

    // Apply fluxes to state_toupdate
    BL_PROFILE_VAR("advance_MUSCL_rusanov_adv::update", pupdate);
    ParallelFor(bx, EULER_NCOMP,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                {
                    consv_out(i, j, k, n)
                        = toupdate(i, j, k, n)
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
    BL_PROFILE_VAR_STOP(pupdate);

    // Copy temporary fluxes onto potentially smaller nodal tile
    BL_PROFILE_VAR("advance_MUSCL_rusanov_adv::flux_copy", pfluxcopy);
    for (int d = 0; d < AMREX_SPACEDIM; d++)
        ParallelFor(nbx[d], EULER_NCOMP,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    { fluxes[d](i, j, k, n) = tfluxes_c[d](i, j, k, n); });
    BL_PROFILE_VAR_STOP(pfluxcopy);
}