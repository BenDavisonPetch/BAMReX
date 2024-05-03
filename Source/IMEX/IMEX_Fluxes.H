#ifndef IMEX_Fluxes_H_
#define IMEX_Fluxes_H_

#include <AMReX_FArrayBox.H>

#include "IMEXSettings.H"

/**
 * \brief Advances using the Rusanov flux for the advection system and returns
 * the conservative fluxes
 *
 * \param time
 * \param bx the box of real values
 * \param nbx nodal tile boxes for each flux FArrayBox (should be such as to
 * avoid race conditions when using tiling)
 * \param statein used to calculate fluxes
 * \param state_toupdate updated with the fluxes to obtain \c stateout
 * \param stateout
 * \param fx x-flux
 * \param fy y-flux
 * \param fz z-flux
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
                         const amrex::FArrayBox &state_toupdate,
                         amrex::FArrayBox       &stateout,
                         AMREX_D_DECL(amrex::FArrayBox &fx,
                                      amrex::FArrayBox &fy,
                                      amrex::FArrayBox &fz),
                         amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
                         const amrex::Real                            dt);

#endif