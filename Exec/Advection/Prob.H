#ifndef PROB_H_
#define PROB_H_

#include "System/Euler/Euler.H"

#include <AMReX_Geometry.H>

void initdata(amrex::MultiFab &S_tmp, const amrex::Geometry &geom);

/**
 * @brief initialises the given multifab with the solution at time \c time.
 * Used for initial conditions as well
 */
void init_exact_solution(
    const amrex::Real time, amrex::MultiFab &S, const amrex::Geometry &geom,
    const amrex::Real density_bg, const amrex::Real density_packet,
    const amrex::Real u_bg, const amrex::Real v_bg, const amrex::Real w_bg,
    const amrex::Real pressure_bg, const amrex::Real x_c,
    const amrex::Real y_c, const amrex::Real side_length,
    const amrex::Real adiabatic, const amrex::Real epsilon);

void write_exact_solution(const amrex::Real time, const amrex::MultiFab &soln,
                          const amrex::Geometry &geom);

#endif
