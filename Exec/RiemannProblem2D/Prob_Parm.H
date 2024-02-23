#ifndef PROB_PARM_H_
#define PROB_PARM_H_

#include "Euler/NComp.H"
#include "Fluxes/Limiters.H"
#include <AMReX_REAL.H>
#include <AMReX_SPACE.H>

// Define the struct to contain problem-specific variables here.
struct ProbParm
{
    amrex::Real  adiabatic;
    Limiter limiter = minbee_limiter;
    unsigned int limiter_indices[EULER_NCOMP]{
        1 + AMREX_SPACEDIM,
        AMREX_D_DECL(1 + AMREX_SPACEDIM, 1 + AMREX_SPACEDIM,
                     1 + AMREX_SPACEDIM),
        1 + AMREX_SPACEDIM
    };
};

#endif
