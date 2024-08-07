#ifndef IMEX_O1_H_
#define IMEX_O1_H_

#include <AMReX_BCRec.H>
#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MLMG.H>

#include "BCs.H"
#include "IMEXSettings.H"
#include "LinearSolver/MLALaplacianGrad.H"

/**
 * \brief First order routine that calculates the fluxes to go from
 * \c statein_imp to a state stepped by \c dt, using \c statein_exp as the
 * initial values for all explicitly treated variables
 */
AMREX_GPU_HOST
void compute_flux_imex_o1(
    const amrex::Real time, const amrex::Geometry &geom,
    const amrex::MultiFab &statein_imp, const amrex::MultiFab &statein_exp,
    amrex::MultiFab                               &pressure,
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes,
    const amrex::Real dt, const bamrexBCData &bc_data,
    const IMEXSettings &settings);

/**
 * \brief Writes the solution to the pressure equation to \c pressure
 *
 * \param geom
 * \param enthalpy cell-centred specific enthalpy
 * \param velocity only used in linearised IMEX with the split KE term. Should
 * be equal to the velocity at the previous timestep
 * \param rhs of the pressure equation
 * \param pressure used for output
 * \param dt
 * \param bc_data
 * \param settings
 */
AMREX_GPU_HOST
void solve_pressure_eqn(const amrex::Geometry &geom,
                        const amrex::MultiFab &enthalpy,
                        const amrex::MultiFab &velocity,
                        const amrex::MultiFab &rhs, amrex::MultiFab &pressure,
                        const amrex::Real dt, const bamrexBCData &bc_data,
                        const IMEXSettings &settings);

/**
 * \brief Computes the enthalpy on cell-centres. Will update the pressure
 * accordingly with an initial guess if the settings determine we are not
 * reusing it
 *
 * \c statenew is used during Picard iterations when settings.linearise = false
 */
AMREX_GPU_HOST
void compute_enthalpy(const amrex::Box &bx, const amrex::FArrayBox &statein,
                      const amrex::FArrayBox &stateexp,
                      const amrex::FArrayBox &statenew,
                      amrex::FArrayBox &pressure, amrex::FArrayBox &enthalpy,
                      const IMEXSettings &settings);

/**
 * Computes enthalpy from the pressure and the density in \c state
 */
AMREX_GPU_HOST
void compute_enthalpy(const amrex::Box &bx, const amrex::FArrayBox &state,
                      amrex::FArrayBox &pressure, amrex::FArrayBox &enthalpy);

/**
 * \brief Computes the (epsilon scaled) kinetic energy on cell-centres.
 *
 * \c statenew is used during Picard iterations when settings.linearise = false
 */
AMREX_GPU_HOST
void compute_ke(const amrex::Box &bx, const amrex::FArrayBox &statein,
                const amrex::FArrayBox &stateexp,
                const amrex::FArrayBox &statenew, amrex::FArrayBox &ke,
                const IMEXSettings &settings);

AMREX_GPU_HOST
void compute_velocity(const amrex::Box &bx, const amrex::FArrayBox &state,
                      amrex::FArrayBox &dst);

/**
 * Used by the RK method to retrieve the pressure for given initial conditions.
 *
 * @param statein_exp_new \[ (U_E)^l \]
 * @param statein_exp_old \[ (U_E)^{l-1} \]
 * @param a_pressure
 * @param geom
 * @param adt \[ a_{ll} \Delta t \]
 * @param settings
 */
AMREX_GPU_HOST
void compute_pressure(const amrex::MultiFab &statein_exp_new,
                      const amrex::MultiFab &statein_exp_old,
                      amrex::MultiFab &a_pressure, const amrex::Geometry &geom,
                      const amrex::Real adt, const IMEXSettings &settings);

AMREX_GPU_HOST
void pressure_source_vector(const amrex::Box       &bx,
                            const amrex::FArrayBox &stateexp,
                            const amrex::FArrayBox &enthalpy,
                            const amrex::FArrayBox &ke, amrex::FArrayBox &dst,
                            AMREX_D_DECL(amrex::Real hdtdx, amrex::Real hdtdy,
                                         amrex::Real hdtdz));

/**
 * Calculates the momentum from the explicitly updated values and pressure and
 * places them in \c dst
 */
AMREX_GPU_HOST
void update_momentum(const amrex::Box &bx, const amrex::FArrayBox &stateex,
                     const amrex::FArrayBox &pressure, amrex::FArrayBox &dst,
                     AMREX_D_DECL(amrex::Real hdtdxe, amrex::Real hdtdye,
                                  amrex::Real hdtdze));

/**
 * \brief Updates momentum flux, given fluxes from explicit update and solution
 * to pressure equation
 */
AMREX_GPU_HOST
void update_momentum_flux(
    const amrex::GpuArray<amrex::Box, AMREX_SPACEDIM> &nbx,
    const amrex::FArrayBox                            &pressure,
    AMREX_D_DECL(amrex::FArrayBox &fx, amrex::FArrayBox &fy,
                 amrex::FArrayBox &fz));

/**
 * Computes energy flux
 */
AMREX_GPU_HOST
void update_energy_flux(
    amrex::GpuArray<amrex::Box, AMREX_SPACEDIM> nbx,
    const amrex::FArrayBox &stateex, const amrex::FArrayBox &enthalpy,
    const amrex::FArrayBox &pressure,
    AMREX_D_DECL(amrex::FArrayBox &fx, amrex::FArrayBox &fy,
                 amrex::FArrayBox &fz),
    AMREX_D_DECL(amrex::Real hdtdxe, amrex::Real hdtdye, amrex::Real hdtdze));

namespace details
{
/**
 * Sets up the pressure operator
 *
 * \param geom
 * \param enthalpy
 * \param velocity
 * \param rhs
 * \param pressure_scratch filled with the dirichlet boundary conditions and
 * used to set in the linear solver if present
 * \param pressure_op
 * \param dt
 * \param bc_data
 * \param settings
 */
AMREX_GPU_HOST
void setup_pressure_op(const amrex::Geometry      &geom,
                       const amrex::MultiFab      &enthalpy,
                       const amrex::MultiFab      &velocity,
                       const amrex::MultiFab      &rhs,
                       amrex::MultiFab            &pressure_scratch,
                       amrex::MLABecLaplacianGrad &pressure_op,
                       const amrex::Real dt, const bamrexBCData &bc_data,
                       const IMEXSettings &settings);

/**
 * Configures and runs the solver for pressure given a configured pressure
 * operator and populated RHS
 */
AMREX_GPU_HOST
void solve_pressure(const amrex::Geometry &geom, const amrex::MultiFab &rhs,
                    amrex::MultiFab &pressure, amrex::MLMG &solver,
                    const bamrexBCData &bc_data, const IMEXSettings &settings);

/**
 * \param pressure_op
 * \param geom
 * \param pressure_scratch filled and used to specify the dirichlet boundary
 * conditions if present \param bc_data
 */
AMREX_GPU_HOST
void set_pressure_domain_BC(amrex::MLLinOp        &pressure_op,
                            const amrex::Geometry &geom,
                            amrex::MultiFab       &pressure_scratch,
                            const bamrexBCData    &bc_data);

AMREX_GPU_HOST
void copy_pressure(const amrex::MultiFab &statesrc, amrex::MultiFab &dst);

AMREX_GPU_HOST
void picard_iterate(
    const amrex::Geometry &geom, const amrex::MultiFab &statein,
    const amrex::MultiFab &stateex, const amrex::MultiFab &velocity,
    amrex::MultiFab &stateout, amrex::MultiFab &enthalpy, amrex::MultiFab &ke,
    amrex::MultiFab &rhs, amrex::MultiFab &pressure, amrex::Real dt,
    AMREX_D_DECL(amrex::Real hdtdx, amrex::Real hdtdy, amrex::Real hdtdz),
    AMREX_D_DECL(amrex::Real hdtdxe, amrex::Real hdtdye, amrex::Real hdtdze),
    const bamrexBCData &bc_data, const IMEXSettings &settings);
}

#endif