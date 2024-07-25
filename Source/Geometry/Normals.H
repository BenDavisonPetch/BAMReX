/**
 * Normal vector computation for the supported geometry types. Normal vectors
 * always point into the fluid. Since LS > 0 inside body, normal is -grad(LS)
 */

#ifndef Normals_H_
#define Normals_H_

#include <AMReX_RealVect.H>

void fill_ls_normals(const amrex::Geometry &geom, amrex::MultiFab &LS);

#endif