#ifndef PROB_H_
#define PROB_H_

#include "Euler/Euler.H"

#include <AMReX_Geometry.H>

void initdata(amrex::MultiFab &S_tmp, const amrex::Geometry &geom);

/**
 * @brief Creates a plotfile for a 1000x1x1 grid containing the values of the
 * exact solution to the Riemann problem at time \c time
 */
void write_exact_solution(amrex::Real time, amrex::Real p, amrex::Real adiabatic,
                          amrex::Real epsilon);

#endif
