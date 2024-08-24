#ifndef IMEX_RK_H_
#define IMEX_RK_H_

#include "IMEX/IMEXSettings.H"
#include "IMEX_O1.H"

/**
 * \brief Uses IMEX-RK method of Boscarino et al. 2016, as presented in
 * Boscheri and Pareshi 2021
 */
AMREX_GPU_HOST
void advance_imex_rk(const amrex::Real time, const amrex::Geometry &geom,
                     const amrex::MultiFab &statein, amrex::MultiFab &stateout,
                     amrex::MultiFab                               &pressure,
                     amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes,
                     const amrex::Real dt, const bamrexBCData &bc_data,
                     const IMEXSettings settings);

/**
 * \brief Same as above but implements high-Mach number stabilisation for high
 * order methods (only active if \c settings.stabilize=true, otherwise
 * identical to above)
 */
AMREX_GPU_HOST
void advance_imex_rk_stab(
    const amrex::Real time, const amrex::Geometry &geom,
    const amrex::MultiFab &statein, amrex::MultiFab &stateout,
    amrex::MultiFab                               &pressure,
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes,
    const amrex::Real dt, const bamrexBCData &bc_data,
    const IMEXSettings settings);

/**
 * \brief Sums the given fluxes with the given weighting
 */
AMREX_GPU_HOST
void sum_fluxes(
    const amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> >
                                                  &fluxes,
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &dst,
    const amrex::Vector<amrex::Real>              &weighting);

/**
 * \brief Computes the shock indicator on each cell and returns the maximum
 * value
 */
AMREX_GPU_HOST
amrex::Real compute_shock_indicator(const amrex::Geometry &geom,
                                    amrex::MultiFab       &dst,
                                    const amrex::MultiFab &state,
                                    const bamrexBCData    &bc_data,
                                    const amrex::Real      k1);

#endif