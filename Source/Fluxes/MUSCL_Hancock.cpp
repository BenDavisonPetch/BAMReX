#include "AmrLevelAdv.H"
#include "Fluxes/Flux_Helpers.H"
#include "Fluxes/Fluxes.H"
#include "Fluxes/Limiters.H"

#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_Extension.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_GpuLaunchFunctsC.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

using namespace amrex;

AMREX_GPU_HOST
void compute_MUSCL_hancock_flux(
    const int dir, amrex::Real time, const amrex::Box &bx,
    const amrex::Array4<amrex::Real>                &flux,
    const amrex::Array4<const amrex::Real>          &consv_values,
    amrex::GpuArray<amrex::Real, BL_SPACEDIM> const &dx_arr, amrex::Real dt)
{
    const Real  adiabatic = AmrLevelAdv::h_prob_parm->adiabatic;
    // const Real &dx        = dx_arr[dir];
    // define boxes (i.e. index domains) for cell-centred values with ghost
    // cells, and for fluxes in this direction
    AMREX_ASSERT(AmrLevelAdv::NUM_GROW >= 2);
    // box grown by 1
    const Box &bx_g1   = grow(bx, 1);
    // const Box &flux_bx = growHi(bx, dir, 1);

    // Temporary fab
    FArrayBox tmpfab;
    // We size the temporary fab to cover the same indices as convs_values,
    // i.e. including ghost cells
    const int NSTATE = AmrLevelAdv::NUM_STATE;
    tmpfab.resize(bx_g1, NSTATE * 2);
    Elixir tmpeli = tmpfab.elixir();

    Array4<Real> half_stepped_L = tmpfab.array(0, NSTATE);
    Array4<Real> half_stepped_R = tmpfab.array(NSTATE, NSTATE);

    reconstruct_and_half_time_step(dir, bx_g1, time, half_stepped_L,
                                   half_stepped_R, adiabatic, adiabatic,
                                   consv_values, dx_arr, dt);

    //
    // Compute force flux from half time stepped values
    //
    compute_HLLC_flux_LR(dir, time, bx, flux, half_stepped_R, half_stepped_L,
                         adiabatic, adiabatic, dx_arr, dt);
}