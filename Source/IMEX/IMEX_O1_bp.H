/**
 * Boscheri and Pareshi formulation
 */
#ifndef IMEX_O1_BP_H_
#define IMEX_O1_BP_H_

#include <AMReX_BCRec.H>
#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MLABecLaplacian.H>

/**
 * Boscheri and Pareshi imex update
 */
AMREX_GPU_HOST
void advance_o1_pimex_bp(int level, amrex::IntVect &crse_ratio,
                         const amrex::Real time, const amrex::Geometry &geom,
                         const amrex::MultiFab &statein,
                         amrex::MultiFab       &stateout,
                         amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes,
                         const amrex::MultiFab             *crse_soln,
                         const amrex::Real                  dt,
                         const amrex::Vector<amrex::BCRec> &bcs, int verbose,
                         int bottom_verbose);

// Boscheri and Pareshi version
AMREX_GPU_HOST
void pressure_source_vector_bp(
    const amrex::MultiFab &MFex, const amrex::MultiFab &MFscaledenth_cc,
    const amrex::MultiFab &MFpgradcoef, amrex::MultiFab &dst, amrex::Real hedt,
    AMREX_D_DECL(amrex::Real hedtdx, amrex::Real hedtdy, amrex::Real hedtdz));

AMREX_GPU_HOST
void update_ex_from_pressure_bp(
    amrex::Real time, const amrex::Geometry &geom,
    const amrex::MultiFab &statein, const amrex::MultiFab &MFpressure,
    const amrex::MultiFab &MFex, const amrex::MultiFab &MFscaledenth_cc,
    amrex::MultiFab                               &stateout,
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes, amrex::Real dt);

#endif