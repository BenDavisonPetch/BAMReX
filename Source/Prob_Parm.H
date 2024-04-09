#ifndef PROB_PARM_H_
#define PROB_PARM_H_

#include "Euler/NComp.H"
#include "Fluxes/Limiters.H"
#include <AMReX_Array.H>
#include <AMReX_REAL.H>
#include <AMReX_SPACE.H>

// Define the struct to contain problem-specific variables here.
// Initialised by Adv_prob.cpp
struct ProbParm
{
    amrex::Real  adiabatic;
    amrex::Real  epsilon;
    Limiter limiter = minbee_limiter;
    amrex::GpuArray<unsigned int, EULER_NCOMP> limiter_indices{
        1 + AMREX_SPACEDIM,
        AMREX_D_DECL(1 + AMREX_SPACEDIM, 1 + AMREX_SPACEDIM,
                     1 + AMREX_SPACEDIM),
        1 + AMREX_SPACEDIM
    };
};

#endif
