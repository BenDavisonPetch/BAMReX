#ifndef PROB_PARM_H_
#define PROB_PARM_H_

#include "Euler/NComp.H"
#include "Fluxes/Limiters.H"
#include <AMReX_Array.H>
#include <AMReX_REAL.H>

using namespace amrex::literals;

//! Contains parameters common to all problems
struct Parm
{
    amrex::Real                                adiabatic;
    amrex::Real                                epsilon;
    Limiter                                    limiter;
    amrex::GpuArray<unsigned int, EULER_NCOMP> limiter_indices;

    //! Set default values
    void set_defaults();
    //! Finish initialisation
    void init();
};

#endif
