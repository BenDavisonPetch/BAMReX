#include "AmrLevelAdv.H"
#include "Fluxes/Fluxes.H"

#include <AMReX_Box.H>
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
    compute_force_flux_LR(dir, time, bx, flux, consv_values, consv_values,
                          dx_arr, dt);
}

AMREX_GPU_HOST
void compute_force_flux_LR(
    const int dir, amrex::Real time, const amrex::Box &bx,
    const amrex::Array4<amrex::Real>                &flux,
    const amrex::Array4<const amrex::Real>          &consv_values_L,
    const amrex::Array4<const amrex::Real>          &consv_values_R,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr, amrex::Real dt)
{
    const Real &dx = dx_arr[dir];
    // define boxes (i.e. index domains) for cell-centred values with ghost
    // cells, and for fluxes in this direction
    const Box &ghost_bx = grow(bx, 1);
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
    const int NSTATE = AmrLevelAdv::NUM_STATE;
    tmpfab.resize(ghost_bx, NSTATE * 4);
    Elixir tmpeli = tmpfab.elixir();

    // We're going to reuse this to store the flux function evaluated on the
    // half updated values after we've computed the LF contribution to the
    // force flux
    Array4<Real> flux_func_values_L  = tmpfab.array(0, NSTATE);
    Array4<Real> flux_func_values_R  = tmpfab.array(NSTATE, NSTATE);
    Array4<Real> primv_values        = tmpfab.array(NSTATE * 2, NSTATE);
    Array4<Real> half_updated_values = tmpfab.array(NSTATE * 3, NSTATE);

    // compute primitive values and flux function values
    compute_primitive_values(time, ghost_bx, primv_values, consv_values_L);
    compute_flux_function(dir, time, ghost_bx, flux_func_values_L,
                          primv_values, consv_values_L);
    compute_primitive_values(time, ghost_bx, primv_values, consv_values_R);
    compute_flux_function(dir, time, ghost_bx, flux_func_values_R,
                          primv_values, consv_values_R);

    int i_offset = (dir == 0) ? 1 : 0;
    int j_offset = (dir == 1) ? 1 : 0;
    int k_offset = (dir == 2) ? 1 : 0;

    // add half of the Lax Friedrichs flux to the force flux,
    // and compute half updated values
    ParallelFor(
        flux_bx, AmrLevelAdv::NUM_STATE,
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

    compute_primitive_values(time, flux_bx, primv_values, half_updated_values);

    // we just reuse the L flux func values
    compute_flux_function(dir, time, flux_bx, flux_func_values_L, primv_values,
                          half_updated_values);

    // Now flux_func_values contains the Richtmeyer flux, so we add half to the
    // total flux.
    ParallelFor(flux_bx, AmrLevelAdv::NUM_STATE,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                { flux(i, j, k, n) += 0.5 * flux_func_values_L(i, j, k, n); });

    // and we're done
}

AMREX_GPU_HOST
void compute_LF_flux(const int dir, amrex::Real time, const amrex::Box &bx,
                     const amrex::Array4<amrex::Real>       &flux,
                     const amrex::Array4<const amrex::Real> &consv_values,
                     amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr,
                     amrex::Real                                      dt)
{
    const Real dx = dx_arr[dir];

    const Box &ghost_bx = grow(bx, AmrLevelAdv::NUM_GROW);
    const Box &flux_bx  = surroundingNodes(bx, dir);

    FArrayBox tmpfab;
    tmpfab.resize(ghost_bx, AmrLevelAdv::NUM_STATE * 2);
    Elixir tmpeli = tmpfab.elixir();

    const Array4<Real> &primv_values = tmpfab.array(0, AmrLevelAdv::NUM_STATE);
    const Array4<Real> &flux_func_values
        = tmpfab.array(AmrLevelAdv::NUM_STATE, AmrLevelAdv::NUM_STATE);

    compute_primitive_values(time, ghost_bx, primv_values, consv_values);
    compute_flux_function(dir, time, ghost_bx, flux_func_values, primv_values,
                          consv_values);

    int i_off = (dir == 0) ? 1 : 0;
    int j_off = (dir == 1) ? 1 : 0;
    int k_off = (dir == 2) ? 1 : 0;

    ParallelFor(
        flux_bx, AmrLevelAdv::NUM_STATE,
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