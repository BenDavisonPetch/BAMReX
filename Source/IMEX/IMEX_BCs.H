#ifndef IMEX_BCs_H_
#define IMEX_BCs_H_

#include <AMReX_MultiFab.H>

AMREX_GPU_HOST
void extrap_grad(const amrex::Geometry &geom, const amrex::MultiFab &src,
                 amrex::MultiFab &dst);

#endif