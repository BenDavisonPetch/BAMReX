#include "Fluxes/Fluxes.H"

#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>

using namespace amrex;

template <int dir>
AMREX_GPU_HOST void
compute_force_flux(amrex::Real time, const amrex::Box &bx,
                   const amrex::Array4<amrex::Real>       &flux,
                   const amrex::Array4<const amrex::Real> &consv_values,
                   amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr,
                   amrex::Real dt, const System *system, const Parm *h_parm,
                   const Parm *d_parm)
{
    compute_force_flux_LR<dir>(time, bx, flux, consv_values, consv_values,
                               dx_arr, dt, system, h_parm, d_parm);
}

template <int dir>
AMREX_GPU_HOST void
compute_force_flux_LR(amrex::Real time, const amrex::Box &bx,
                      const amrex::Array4<amrex::Real>       &flux,
                      const amrex::Array4<const amrex::Real> &consv_values_L,
                      const amrex::Array4<const amrex::Real> &consv_values_R,
                      amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr,
                      amrex::Real dt, const System *system, const Parm *h_parm,
                      const Parm *d_parm)
{
    const Real &dx = dx_arr[dir];
    // define boxes (i.e. index domains) for cell-centred values with ghost
    // cells, and for fluxes in this direction
    const Box &ghost_bx = grow(bx, dir, 1);
    const Box &flux_bx  = growHi(bx, dir, 1);
    // Create a temporary fab to store our intermediate values. This is done
    // with FArrayBox, which is the datatype that MultiFab contains and
    // represents an array over one box. Because GPU kernels are asynchronous,
    // we need to ensure that this doesn't go out of memory before the GPU
    // kernels can access its data. To do this we use Elixir, part of AMReX's
    // GPU memory management system.
    FArrayBox tmpfab;
    // We size the temporary fab to cover the same indices as convs_values,
    // i.e. including ghost cells We only need to store
    const int NSTATE = system->StateSize();
    AMREX_ASSERT(flux.nComp() == NSTATE);
    AMREX_ASSERT(consv_values_L.nComp() == NSTATE);
    AMREX_ASSERT(consv_values_R.nComp() == NSTATE);
    tmpfab.resize(ghost_bx, NSTATE * 3);
    Elixir tmpeli = tmpfab.elixir();

    // We're going to reuse this to store the flux function evaluated on the
    // half updated values after we've computed the LF contribution to the
    // force flux
    Array4<Real> flux_func_values_L  = tmpfab.array(0, NSTATE);
    Array4<Real> flux_func_values_R  = tmpfab.array(NSTATE, NSTATE);
    Array4<Real> half_updated_values = tmpfab.array(NSTATE * 2, NSTATE);

    system->FillFluxes(dir, time, ghost_bx, flux_func_values_L, consv_values_L,
                       h_parm, d_parm);
    system->FillFluxes(dir, time, ghost_bx, flux_func_values_R, consv_values_R,
                       h_parm, d_parm);

    int i_offset = (dir == 0) ? 1 : 0;
    int j_offset = (dir == 1) ? 1 : 0;
    int k_offset = (dir == 2) ? 1 : 0;

    // add half of the Lax Friedrichs flux to the force flux,
    // and compute half updated values
    ParallelFor(
        flux_bx, NSTATE,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
        {
            flux(i, j, k, n)
                = 0.5
                  * ((0.5 * dx / dt)
                         * (consv_values_L(i - i_offset, j - j_offset,
                                           k - k_offset, n)
                            - consv_values_R(i, j, k, n))
                     + 0.5
                           * (flux_func_values_L(i - i_offset, j - j_offset,
                                                 k - k_offset, n)
                              + flux_func_values_R(i, j, k, n)));

            half_updated_values(i, j, k, n)
                = 0.5
                      * (consv_values_L(i - i_offset, j - j_offset,
                                        k - k_offset, n)
                         + consv_values_R(i, j, k, n))
                  + (dt / (2 * dx))
                        * (flux_func_values_L(i - i_offset, j - j_offset,
                                              k - k_offset, n)
                           - flux_func_values_R(i, j, k, n));
        });

    // we just reuse the L flux func values
    system->FillFluxes(dir, time, flux_bx, flux_func_values_L,
                       half_updated_values, h_parm, d_parm);

    // Now flux_func_values contains the Richtmeyer flux, so we add half to the
    // total flux.
    ParallelFor(flux_bx, NSTATE,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                { flux(i, j, k, n) += 0.5 * flux_func_values_L(i, j, k, n); });

    // and we're done
}

template <int dir>
AMREX_GPU_HOST void
compute_LF_flux(amrex::Real time, const amrex::Box &bx,
                const amrex::Array4<amrex::Real>                &flux,
                const amrex::Array4<const amrex::Real>          &consv_values,
                amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr,
                amrex::Real dt, const System *system, const Parm *h_parm,
                const Parm *d_parm)
{
    const Real dx = dx_arr[dir];

    const Box &ghost_bx = grow(bx, 1);
    const Box &flux_bx  = surroundingNodes(bx, dir);

    const int NSTATE = system->StateSize();

    FArrayBox tmpfab;
    tmpfab.resize(ghost_bx, NSTATE);
    Elixir tmpeli = tmpfab.elixir();

    const Array4<Real> &flux_func_values = tmpfab.array(0, NSTATE);

    system->FillFluxes(dir, time, ghost_bx, flux_func_values, consv_values,
                       h_parm, d_parm);

    int i_off = (dir == 0) ? 1 : 0;
    int j_off = (dir == 1) ? 1 : 0;
    int k_off = (dir == 2) ? 1 : 0;

    ParallelFor(
        flux_bx, NSTATE,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
        {
            flux(i, j, k, n)
                = 0.5 * dx / dt
                      * (consv_values(i - i_off, j - j_off, k - k_off, n)
                         - consv_values(i, j, k, n))
                  + 0.5
                        * (flux_func_values(i - i_off, j - j_off, k - k_off, n)
                           + flux_func_values(i, j, k, n));
        });
}

//
// Template instantiations
//

template AMREX_GPU_HOST void
compute_force_flux<0>(amrex::Real, const amrex::Box &,
                      const amrex::Array4<amrex::Real> &,
                      const amrex::Array4<const amrex::Real> &,
                      amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &,
                      amrex::Real, const System *, const Parm *, const Parm *);

#if AMREX_SPACEDIM >= 2
template AMREX_GPU_HOST void
compute_force_flux<1>(amrex::Real, const amrex::Box &,
                      const amrex::Array4<amrex::Real> &,
                      const amrex::Array4<const amrex::Real> &,
                      amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &,
                      amrex::Real, const System *, const Parm *, const Parm *);
#endif
#if AMREX_SPACEDIM >= 3
template AMREX_GPU_HOST void
compute_force_flux<2>(amrex::Real, const amrex::Box &,
                      const amrex::Array4<amrex::Real> &,
                      const amrex::Array4<const amrex::Real> &,
                      amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &,
                      amrex::Real, const System *, const Parm *, const Parm *);
#endif

template AMREX_GPU_HOST void compute_force_flux_LR<0>(
    amrex::Real, const amrex::Box &, const amrex::Array4<amrex::Real> &,
    const amrex::Array4<const amrex::Real> &,
    const amrex::Array4<const amrex::Real> &,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &, amrex::Real,
    const System *, const Parm *, const Parm *);

#if AMREX_SPACEDIM >= 2
template AMREX_GPU_HOST void compute_force_flux_LR<1>(
    amrex::Real, const amrex::Box &, const amrex::Array4<amrex::Real> &,
    const amrex::Array4<const amrex::Real> &,
    const amrex::Array4<const amrex::Real> &,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &, amrex::Real,
    const System *, const Parm *, const Parm *);
#endif
#if AMREX_SPACEDIM >= 3
template AMREX_GPU_HOST void compute_force_flux_LR<2>(
    amrex::Real, const amrex::Box &, const amrex::Array4<amrex::Real> &,
    const amrex::Array4<const amrex::Real> &,
    const amrex::Array4<const amrex::Real> &,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &, amrex::Real,
    const System *, const Parm *, const Parm *);
#endif

template AMREX_GPU_HOST void
compute_LF_flux<0>(amrex::Real, const amrex::Box &,
                   const amrex::Array4<amrex::Real> &,
                   const amrex::Array4<const amrex::Real> &,
                   amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &,
                   amrex::Real, const System *, const Parm *, const Parm *);

#if AMREX_SPACEDIM >= 2
template AMREX_GPU_HOST void
compute_LF_flux<1>(amrex::Real, const amrex::Box &,
                   const amrex::Array4<amrex::Real> &,
                   const amrex::Array4<const amrex::Real> &,
                   amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &,
                   amrex::Real, const System *, const Parm *, const Parm *);
#endif
#if AMREX_SPACEDIM >= 3
template AMREX_GPU_HOST void
compute_LF_flux<2>(amrex::Real, const amrex::Box &,
                   const amrex::Array4<amrex::Real> &,
                   const amrex::Array4<const amrex::Real> &,
                   amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &,
                   amrex::Real, const System *, const Parm *, const Parm *);
#endif