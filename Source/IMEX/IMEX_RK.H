#ifndef IMEX_RK_H_
#define IMEX_RK_H_

#include "IMEX_O1.H"

/**
 * Uses IMEX-RK method of Boscarino et al. 2016, as presented in Boscheri and
 * Pareshi 2021
 */
void advance_imex_rk(int level, amrex::IntVect &crse_ratio,
                     const amrex::Real time, const amrex::Geometry &geom,
                     const amrex::MultiFab &statein, amrex::MultiFab &stateout,
                     amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes,
                     const amrex::Real                              dt,
                     const amrex::Vector<amrex::BCRec>             &domainbcs,
                     const IMEXSettings settings, int verbose,
                     int bottom_verbose);

#endif