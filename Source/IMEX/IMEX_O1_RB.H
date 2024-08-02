/**
 * Linear solver setup for rigid body calculations
 */
#ifndef IMEX_O1_RB_H_
#define IMEX_O1_RB_H_

#include <AMReX.H>

#ifdef AMREX_USE_EB
#include <AMReX_MLMG.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MLEBABecLap.H>

#include "BCs.H"
#include "IMEXSettings.H"

namespace IMEXEB
{

/**
 * Sets up the pressure operator when using rigid bodies
 *
 * \param geom
 * \param enthalpy
 * \param rhs
 * \param pressure_scratch filled with the dirichlet boundary conditions and
 * used to set in the linear solver if present
 * \param pressure_op
 * \param U conservative variables used by some boundary conditions to
 * determine dirichlet BC for pressure
 * \param dt
 * \param bc_data
 * \param settings
 */
AMREX_GPU_HOST
void setup_pressure_op_rb(const amrex::Geometry &geom,
                          const amrex::MultiFab &enthalpy,
                          const amrex::MultiFab &rhs,
                          amrex::MultiFab       &pressure_scratch,
                          amrex::MLEBABecLap    &pressure_op,
                          const amrex::MultiFab &U, const amrex::Real dt,
                          const bamrexBCData &bc_data);

} // namespace IMEXEB

#endif
#endif