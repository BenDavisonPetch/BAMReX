#include "AmrLevelAdv.H"
#include "Euler/Euler.H"
#include "Fluxes/Fluxes.H"

#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>

using namespace amrex;

AMREX_GPU_HOST
void compute_force_flux(
    const int dir, amrex::Real time, const amrex::Box &bx,
    const amrex::Array4<amrex::Real>                &flux,
    const amrex::Array4<const amrex::Real>          &consv_values,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr, amrex::Real dt)
{
    const Real &dx = dx_arr[dir];
    // define boxes (i.e. index domains) for cell-centred values with ghost
    // cells, and for fluxes in this direction
    const Box &ghost_bx = grow(bx, AmrLevelAdv::NUM_GROW);
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
    tmpfab.resize(ghost_bx, AmrLevelAdv::NUM_STATE * 4);
    Elixir tmpeli = tmpfab.elixir();

    // We're going to reuse this to store the flux function evaluated on the
    // half updated values after we've computed the LF contribution to the
    // force flux
    Array4<Real> flux_func_values = tmpfab.array(0, AmrLevelAdv::NUM_STATE);
    Array4<Real> primv_values
        = tmpfab.array(AmrLevelAdv::NUM_STATE, AmrLevelAdv::NUM_STATE);
    Array4<Real> half_updated_values
        = tmpfab.array(AmrLevelAdv::NUM_STATE * 2, AmrLevelAdv::NUM_STATE);
    Array4<Real> half_updated_primv_values
        = tmpfab.array(AmrLevelAdv::NUM_STATE * 3, AmrLevelAdv::NUM_STATE);
    // compute primitive values and flux function values
    compute_primitive_values(time, ghost_bx, primv_values, consv_values);
    compute_flux_function(dir, time, ghost_bx, flux_func_values, primv_values,
                          consv_values);

    int i_offset = (dir == 0) ? 1 : 0;
    int j_offset = (dir == 1) ? 1 : 0;
    int k_offset = (dir == 2) ? 1 : 0;

    // add half of the Lax Friedrichs flux to the force flux
    ParallelFor(
        flux_bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
        {
            flux(i, j, k, n)
                = 0.5
                  * ((0.5 * dx / dt)
                         * (consv_values(i - i_offset, j - j_offset,
                                         k - k_offset, n)
                            - consv_values(i, j, k, n))
                     + 0.5
                           * (flux_func_values(i - i_offset, j - j_offset,
                                               k - k_offset, n)
                              + flux_func_values(i, j, k, n)));
        });

    ParallelFor(
        ghost_bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
        {
            half_updated_values(i, j, k, n)
                = 0.5
                  * ((consv_values(i - i_offset, j - j_offset, k - k_offset, n)
                      + consv_values(i, j, k, n))
                     - dt / dx
                           * (flux_func_values(i, j, k, n)
                              - flux_func_values(i - i_offset, j - j_offset,
                                                 k - k_offset, n)));
        });

    compute_primitive_values(time, ghost_bx, half_updated_primv_values,
                             half_updated_values);

    compute_flux_function(dir, time, ghost_bx, flux_func_values,
                          half_updated_primv_values, half_updated_values);

    // Now flux_func_values contains the Richtmeyer flux, so we add half to the
    // total flux.
    ParallelFor(flux_bx, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                { flux(i, j, k, n) += 0.5 * flux_func_values(i, j, k, n); });

    // and we're done
}
