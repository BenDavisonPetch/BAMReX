/**
 * Contains routines that can be used to reduce the boilerplate in dispatching
 * MFIter loops in a performance parallel way. Adapted from the example code
 * provided by AMReX
 */

#ifndef MFIterLoop_H_
#define MFIterLoop_H_

#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_GpuQualifiers.H>

/**
 * \brief Performs a CPU/GPU portable MFIter loop and calls @c inner_func.
 *
 * Adapted from the boilerplate in AMReX's example code Amr/Advection_AmrLevel
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
    const int                 ngrow           = 0,
    const amrex::Vector<int> &flux_directions = { AMREX_D_DECL(0, 1, 2) });

#endif