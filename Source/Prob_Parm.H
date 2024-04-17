#ifndef PROB_PARM_H_
#define PROB_PARM_H_

#include "Euler/NComp.H"
#include "Fluxes/Limiters.H"
#include <AMReX_Array.H>
#include <AMReX_REAL.H>

using namespace amrex::literals;

// Define the struct to contain problem-specific variables here.
// Initialised by Adv_prob.cpp
struct ProbParm
{
    amrex::Real  adiabatic;
    amrex::Real  epsilon;
    Limiter limiter;
    amrex::GpuArray<unsigned int, EULER_NCOMP> limiter_indices;
};

extern AMREX_GPU_CONSTANT ProbParm d_prob_parm;

#endif
