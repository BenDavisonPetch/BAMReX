#ifndef PROB_H_
#define PROB_H_

// dummy Prob.H file for tests that don't need it

#include <AMReX.H>
#include <AMReX_Geometry.H>

void initdata(amrex::MultiFab &S_tmp, const amrex::Geometry &geom) {
    amrex::ignore_unused(S_tmp, geom);
}

#endif
