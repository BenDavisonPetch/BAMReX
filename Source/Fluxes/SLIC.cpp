#include "AmrLevelAdv.H"
#include "Fluxes/Fluxes.H"
#include "Fluxes/Limiters.H"

#include <AMReX_Box.H>
#include <AMReX_Extension.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_GpuLaunchFunctsC.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

using namespace amrex;

AMREX_GPU_HOST
void compute_SLIC_flux(const int dir, amrex::Real time, const amrex::Box &bx,
                       const amrex::Array4<amrex::Real>       &flux,
                       const amrex::Array4<const amrex::Real> &consv_values,
                       amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr,
                       amrex::Real                                      dt)
{
    const Real &dx = dx_arr[dir];
    // define boxes (i.e. index domains) for cell-centred values with ghost
    // cells, and for fluxes in this direction
    AMREX_ASSERT(AmrLevelAdv::NUM_GROW >= 2);
    // box grown by 1
    const Box &bx_g1   = grow(bx, 1);
    const Box &flux_bx = growHi(bx, dir, 1);

    // Temporary fab
    FArrayBox tmpfab;
    // We size the temporary fab to cover the same indices as convs_values,
    // i.e. including ghost cells
    const int NSTATE = AmrLevelAdv::NUM_STATE;
    tmpfab.resize(bx_g1, NSTATE * 8);
    Elixir tmpeli = tmpfab.elixir();

    Array4<Real> reconstructed_L = tmpfab.array(0, NSTATE);
    Array4<Real> reconstructed_R = tmpfab.array(NSTATE, NSTATE);
    Array4<Real> primv_L         = tmpfab.array(2 * NSTATE, NSTATE);
    Array4<Real> primv_R         = tmpfab.array(3 * NSTATE, NSTATE);
    Array4<Real> flux_func_L     = tmpfab.array(4 * NSTATE, NSTATE);
    Array4<Real> flux_func_R     = tmpfab.array(5 * NSTATE, NSTATE);
    Array4<Real> half_stepped_L  = tmpfab.array(6 * NSTATE, NSTATE);
    Array4<Real> half_stepped_R  = tmpfab.array(7 * NSTATE, NSTATE);

    int i_off = (dir == 0) ? 1 : 0;
    int j_off = (dir == 1) ? 1 : 0;
    int k_off = (dir == 2) ? 1 : 0;

    //
    // Reconstruct
    //
    ParallelFor(bx_g1, NSTATE,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                {
                    // Calculate slope ratio
                    // This is technically recomputed for every component of
                    // the variables vector but the alternative isn't great for
                    // cache locality

                    // limiter variable index
                    int        iq = 1 + AMREX_SPACEDIM;
                    const Real numerator
                        = consv_values(i, j, k, iq)
                          - consv_values(i - i_off, j - j_off, k - k_off, iq);
                    const Real denom
                        = consv_values(i + i_off, j + j_off, k + k_off, iq)
                          - consv_values(i, j, k, iq);
                    Real limiter;
                    if (numerator == 0. && denom == 0.)
                        limiter = minbee(1);
                    else if (denom == 0)
                        limiter = 0;
                    else
                        limiter = minbee(numerator / denom);

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

                    reconstructed_L(i, j, k, n)
                        = consv_values(i, j, k, n) - 0.5 * limiter * delta;
                    reconstructed_R(i, j, k, n)
                        = consv_values(i, j, k, n) + 0.5 * limiter * delta;
                });


    //
    // Half time-step update
    //
    compute_primitive_values(time, bx_g1, primv_L, reconstructed_L);
    compute_primitive_values(time, bx_g1, primv_R, reconstructed_R);
    compute_flux_function(dir, time, bx_g1, flux_func_L, primv_L,
                          reconstructed_L);
    compute_flux_function(dir, time, bx_g1, flux_func_R, primv_R,
                          reconstructed_R);

    double hdtdx = 0.5 * dt / dx;

    ParallelFor(
        bx_g1, NSTATE,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
        {
            half_stepped_L(i, j, k, n)
                = reconstructed_L(i, j, k, n)
                  - hdtdx
                        * (flux_func_R(i, j, k, n) - flux_func_L(i, j, k, n));
            half_stepped_R(i, j, k, n)
                = reconstructed_R(i, j, k, n)
                  - hdtdx
                        * (flux_func_R(i, j, k, n) - flux_func_L(i, j, k, n));
        });


    //
    // Compute force flux from half time stepped values
    //
    compute_force_flux_LR(dir, time, bx, flux, half_stepped_R, half_stepped_L,
                          dx_arr, dt);
}