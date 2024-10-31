#ifndef EulerDerive_H_
#define EulerDerive_H_

#include <AMReX_FArrayBox.H>

void euler_derpres(const amrex::Box &bx, amrex::FArrayBox &derfab, int dcomp,
                   int ncomp, const amrex::FArrayBox &datafab,
                   const amrex::Geometry &geomdata, amrex::Real time,
                   const int *bcrec, int level);

void euler_dervel(const amrex::Box &bx, amrex::FArrayBox &derfab, int dcomp,
                  int ncomp, const amrex::FArrayBox &datafab,
                  const amrex::Geometry &geomdata, amrex::Real time,
                  const int *bcrec, int level);

void euler_dertemp(const amrex::Box &bx, amrex::FArrayBox &derfab, int dcomp,
                   int ncomp, const amrex::FArrayBox &datafab,
                   const amrex::Geometry &geomdata, amrex::Real time,
                   const int *bcrec, int level);

#endif