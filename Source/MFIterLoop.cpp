#include "MFIterLoop.H"
#include "AmrLevelAdv.H"
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>

using namespace amrex;

/**
 * \brief Performs a CPU/GPU portable MFIter loop and calls @c inner_func.
 *
 * Adapted from the boilerplate in AMReX's example code Amr/Advection_AmrLevel
 *
 * \n only write to the given FArrayBox's for flux using the provided nodal
 * tile boxes!
 */
void MFIter_loop(
    const amrex::Real time, const amrex::Geometry &geom,
    const amrex::MultiFab &MFin, amrex::MultiFab &MFout,
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes,
    const amrex::Real                              dt,
    std::function<
        void(const amrex::MFIter &, const amrex::Real, const amrex::Box &,
             amrex::GpuArray<amrex::Box, AMREX_SPACEDIM>,
             const amrex::FArrayBox &, amrex::FArrayBox &,
             AMREX_D_DECL(amrex::FArrayBox &, amrex::FArrayBox &,
                          amrex::FArrayBox &),
             amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, const amrex::Real)>
              inner_func,
    const int ngrow, const amrex::Vector<int> &flux_directions)
{
    GpuArray<Real, BL_SPACEDIM> dx = geom.CellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        FArrayBox  fluxfab[AMREX_SPACEDIM];
        FArrayBox *flux[AMREX_SPACEDIM];

        for (MFIter mfi(MFout, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            // Set up tileboxes and nodal tileboxes
            const Box                 &bx = mfi.growntilebox(ngrow);
            GpuArray<Box, BL_SPACEDIM> nbx;
            AMREX_D_TERM(nbx[0] = mfi.grownnodaltilebox(0, ngrow);
                         , nbx[1] = mfi.grownnodaltilebox(1, ngrow);
                         , nbx[2] = mfi.grownnodaltilebox(2, ngrow));

            // Grab fab pointers from state multifabs
            const FArrayBox &statein  = MFin[mfi];
            FArrayBox       &stateout = MFout[mfi];

            for (const int i : flux_directions)
            {
                // Point flux fab pointer to untiled fabs. It is important that
                // inner_func does not try to write to the provided FArrayBox's
                // for flux outside of the provided nodal tile boxes, otherwise
                // a race condition will occur
                flux[i] = &(fluxes[i][mfi]);
            }

            // Call the provided function
            inner_func(mfi, time, bx, nbx, statein, stateout,
                       AMREX_D_DECL(*flux[0], *flux[1], *flux[2]), dx, dt);
        }
    }
}