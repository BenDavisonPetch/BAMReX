#ifndef PlasmaDerive_H_
#define PlasmaDerive_H_

#include <AMReX_FArrayBox.H>

void mhd_derpres(const amrex::Box &bx, amrex::FArrayBox &derfab, int dcomp,
                 int ncomp, const amrex::FArrayBox &datafab,
                 const amrex::Geometry &geomdata, amrex::Real time,
                 const int *bcrec, int level);

void mhd_dertemp(const amrex::Box &bx, amrex::FArrayBox &derfab, int dcomp,
                 int ncomp, const amrex::FArrayBox &datafab,
                 const amrex::Geometry &geomdata, amrex::Real time,
                 const int *bcrec, int level);

#endif