#ifndef PROB_PARM_H_
#define PROB_PARM_H_

#include "Fluxes/Limiters.H"
#include "System/PlasmaIndex.H"
#include <AMReX_Array.H>
#include <AMReX_REAL.H>

#define MAX_NCOMP PUInd::size

using namespace amrex::literals;

//! Contains parameters common to all problems
struct Parm
{
    //! EoS details

    //! Adiabatic index
    amrex::Real adiabatic;
    //! mean molecular weight (g/mol)
    amrex::Real eos_mu;
    //! Prandtl number
    amrex::Real Pr;
    //! Sutherland law parameters: eta = eta' (T/T')^beta (T'+S)/(T+S)
    //! If using scaled equations these have to be scaled as well
    //! To get constant viscosity = eta_prime, set S_eta = 0 and beta = 1
    amrex::Real beta;
    amrex::Real eta_prime;
    amrex::Real T_prime;
    amrex::Real S_eta;

    //! Scaling parameters (if you're unsure of how to use the variable
    //! transformations, keep these as 1)
    amrex::Real epsilon;
    amrex::Real Re;

    //! Derived quantities
    amrex::Real R_spec;
    amrex::Real cv;
    amrex::Real cp;
    /**
     * Precomputed adiabatic-index related values
     * 0: gamma
     * 1: gamma + 1
     * 2: gamma - 1
     * 3: 1 / (gamma + 1)
     * 4: 1 / (gamma - 1)
     * 5: (gamma + 1) / (2 * gamma)
     * 6: (gamma - 1) / (2 * gamma)
     * 7: gamma / (gamma + 1)
     * 8: gamma / (gamma - 1)
     */
    amrex::GpuArray<amrex::Real, 9> adia;
    amrex::Real                     Reinv;

    //! MHM slope limiting options
    Limiter                                    limiter;
    amrex::GpuArray<unsigned int, MAX_NCOMP> limiter_indices;
    amrex::GpuArray<bool, MAX_NCOMP>         limit_on_primv;
    bool need_primitive_for_limiting;
};

#endif
