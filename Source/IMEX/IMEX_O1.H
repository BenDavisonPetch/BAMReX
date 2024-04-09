#ifndef IMEX_O1_H_
#define IMEX_O1_H_

#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_GpuQualifiers.H>

/**
 * \brief Option 1 for AMR implementation of PIMEX scheme. Solves for one
 * level, using the (interpolated in time) level down as dirichlet boundary
 * conditions
 *
 * \param level the AMR level
 * \param crse_ratio the refinement ratio
 * \param time
 * \param geom the AmrLevel's geometry object
 * \param statein input state. Should have additional ghost cells already
 * filled
 * \param stateout output state to write to
 * \param fluxes the (conservative) fluxes to write to
 * \param dt
 * \param verbose
 * \param bottom_verbose
 */
AMREX_GPU_HOST
void advance_o1_pimex(int level, amrex::IntVect &crse_ratio,
                      const amrex::Real time, const amrex::Geometry &geom,
                      const amrex::MultiFab                         &statein,
                      amrex::MultiFab                               &stateout,
                      amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes,
                      const amrex::MultiFab *crse_soln, const amrex::Real dt,
                      int verbose, int bottom_verbose);

/**
 * \brief Advances using the Rusanov flux for the advection system and returns
 * the conservative fluxes
 *
 * \param time
 * \param bx the box of real values
 * \param nbx nodal tile boxes for each flux FArrayBox (should be such as to
 * avoid race conditions when using tiling) \param statein input state \param
 * stateout state to update \param fx x-flux \param fy y-flux \param fz z-flux
 * \param dx
 * \param dt
 *
 * \note Computes the flux on a domain grown by 1 from bx, so \c stateout
 * should be defined on this domain
 */
AMREX_GPU_HOST
void advance_rusanov_adv(const amrex::Real time, const amrex::Box &bx,
                         amrex::GpuArray<amrex::Box, AMREX_SPACEDIM> nbx,
                         const amrex::FArrayBox                     &statein,
                         amrex::FArrayBox                           &stateout,
                         AMREX_D_DECL(amrex::FArrayBox &fx,
                                      amrex::FArrayBox &fy,
                                      amrex::FArrayBox &fz),
                         amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
                         const amrex::Real                            dt);

AMREX_GPU_HOST
void pressure_source_vector(
    const amrex::MultiFab &MFex, const amrex::MultiFab &MFscaledenth_cc,
    amrex::MultiFab &dst,
    AMREX_D_DECL(amrex::Real hedtdx, amrex::Real hedtdy, amrex::Real hedtdz));

/**
 * \brief Takes a solution for the pressure obtained by linear solution, and
 * the explicitly updated conservative variables, and computes the final state
 * and the fluxes.
 *
 * \param time
 * \param geom
 * \param MFpressure the pressure, obtained by solving the pressure equation
 * \param MFex the explicitly updated conservative variables. Will be
 * overwritten with temporary values.
 * \param MFscaledenth_cc a temporary MultiFab that will be overwritten. Should
 * be defined with at least one set of ghost cells.
 * \param stateout the MutliFab to write the final state to
 * \param fluxes should contain, at input, the conservative fluxes used to
 * compute the explicitly updated variables. Will be output as the final
 * conservative fluxes.
 * \param dt
 */
AMREX_GPU_HOST
void update_ex_from_pressure(
    amrex::Real time, const amrex::Geometry &geom,
    const amrex::MultiFab &MFpressure, amrex::MultiFab &MFex,
    amrex::MultiFab &MFscaledenth_cc, amrex::MultiFab &stateout,
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes, amrex::Real dt);

AMREX_GPU_HOST
void copy_pressure(const amrex::MultiFab &statesrc, amrex::MultiFab &dst);

#endif