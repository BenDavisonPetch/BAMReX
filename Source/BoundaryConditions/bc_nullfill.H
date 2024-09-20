#ifndef bc_nullfill_H_
#define bc_nullfill_H_

#include <AMReX_FArrayBox.H>
#include <AMReX_BCRec.H>

void nullfill(amrex::Box const &bx, amrex::FArrayBox &data, int dcomp,
              int numcomp, amrex::Geometry const &geom, amrex::Real time,
              const amrex::Vector<amrex::BCRec> &bcr, int bcomp, int scomp);

#endif