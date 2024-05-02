#ifndef IMEX_O1_H_
#define IMEX_O1_H_

#include <AMReX_BCRec.H>
#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MLABecLaplacian.H>

#include "IMEXSettings.H"

/**
 * \brief Computes the new state and writes it to \c stateout and writes
 * conservative fluxes to \c fluxes
 */
AMREX_GPU_HOST
void advance_imex_o1(const amrex::Real time, const amrex::Geometry &geom,
                     const amrex::MultiFab &statein, amrex::MultiFab &stateout,
                     amrex::MultiFab                               &pressure,
                     amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes,
                     const amrex::Real                              dt,
                     const amrex::Vector<amrex::BCRec>             &domainbcs,
                     const IMEXSettings                            &settings);

/**
 * \brief Writes the solution to the pressure equation to \c pressure
 *
 * \param geom
 * \param enthalpy cell-centred specific enthalpy
 * \param velocity only used in linearised IMEX with the split KE term. Should
 * be equal to the velocity at the previous timestep \param rhs of the pressure
 * equation \param pressure used for output \param dt \param domainbcs \param
 * settings
 */
AMREX_GPU_HOST
void solve_pressure_eqn(const amrex::Geometry &geom,
                        const amrex::MultiFab &enthalpy,
                        const amrex::MultiFab &velocity,
                        const amrex::MultiFab &rhs, amrex::MultiFab &pressure,
                        const amrex::Real                  dt,
                        const amrex::Vector<amrex::BCRec> &domainbcs,
                        const IMEXSettings                &settings);

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
 * Same as above but also computes flux
 */
AMREX_GPU_HOST
void update_momentum(const amrex::Box                           &bx,
                     amrex::GpuArray<amrex::Box, AMREX_SPACEDIM> nbx,
                     const amrex::FArrayBox                     &stateex,
                     const amrex::FArrayBox &pressure, amrex::FArrayBox &dst,
                     AMREX_D_DECL(amrex::FArrayBox &fx, amrex::FArrayBox &fy,
                                  amrex::FArrayBox &fz),
                     AMREX_D_DECL(amrex::Real hdtdxe, amrex::Real hdtdye,
                                  amrex::Real hdtdze));

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
AMREX_GPU_HOST
void set_pressure_domain_BC(amrex::MLLinOp                    &pressure_op,
                            const amrex::Geometry             &geom,
                            const amrex::Vector<amrex::BCRec> &bcs);

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
    const amrex::Vector<amrex::BCRec> &domainbcs,
    const IMEXSettings                &settings);
}

#endif