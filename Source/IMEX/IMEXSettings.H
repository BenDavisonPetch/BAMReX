#ifndef IMEXSettings_H_
#define IMEXSettings_H_

#include "ButcherTableau.H"

struct IMEXSettings
{
    /**
     * Whether or not the method should linearise the pressure equation or use
     * Picard iteration between momentum and pressure
     */
    bool linearise;
    int  picard_max_iter;
    bool compute_residual;

    /**
     * The relative tolerance on the residual for the Picard iteration to
     * terminate. If negative, Picard iteration will always go to max number
     * of iterations
     */
    amrex::Real res_tol = 1e-12;

    /**
     * The method for calculating enthalpy in the energy flux when using a
     * linearised scheme.
     * \c reuse_pressure uses the pressure from the previous
     * timestep
     * \c conservative uses the conservative variables from the
     * previous timestep
     * \c conservative_ex uses the explicitly updated
     * conservative variables
     */
    enum EnthalpyMethod
    {
        reuse_pressure,
        conservative,
        conservative_ex
    } enthalpy_method;

    /**
     * The method for calculating kinetic energy in the energy flux when using
     * a linearised scheme.
     * \c ex uses the explicitly updated conservative variables
     * \c split uses (eps / 2) u^n . q^{n+1} which results in an
     * additional term in the pressure equation of form u^n . grad p^{n+1}
     */
    enum KineticEnergyMethod
    {
        conservative_kinetic,
        ex,
        split
    } ke_method;

    /**
     * Advection flux method
     */
    enum AdvectionFlux {
        rusanov, // 1st order
        muscl_rusanov // 2nd order
    } advection_flux;

    /**
     * Which Butcher Tableau to use
     */
    IMEXButcherTableau::TableauType butcher_tableau;

    bool verbose               = false;
    bool solver_verbose        = false;
    bool bottom_solver_verbose = false;
};

/**
 * Generates settings object from input file and verbosity of AmrLevelAdv.
 */
IMEXSettings get_imex_settings(bool verbose);

#endif