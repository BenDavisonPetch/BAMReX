#ifndef AmrLevelAdv_derive_H_
#define AmrLevelAdv_derive_H_

#include <AMReX_FArrayBox.H>

AMREX_FORCE_INLINE amrex::Box the_same_box(const amrex::Box &b) { return b; }

void derive_rank(const amrex::Box &bx, amrex::FArrayBox &derfab, int dcomp,
                 int ncomp, const amrex::FArrayBox &datafab,
                 const amrex::Geometry &geomdata, amrex::Real time,
                 const int *bcrec, int level);

#endif