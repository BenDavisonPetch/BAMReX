#ifndef PROB_H_
#define PROB_H_

#include "System/Euler/Euler.H"

#include <AMReX_Geometry.H>

void initdata(amrex::MultiFab &S_tmp, const amrex::Geometry &geom);

/**
 * @brief Creates a plotfile for a 1000x1x1 grid containing the values of the
 * exact solution to the Riemann problem at time \c time
 */
void write_exact_solution(
    const amrex::Real                                time,
    const amrex::GpuArray<amrex::Real, EULER_NCOMP> &primv_L,
    const amrex::GpuArray<amrex::Real, EULER_NCOMP> &primv_R,
    const amrex::Real adiabatic_L, const amrex::Real adiabatic_R,
    const amrex::Real epsilon);

#endif
