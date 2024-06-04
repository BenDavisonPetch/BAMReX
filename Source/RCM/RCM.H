#ifndef RCM_H_
#define RCM_H_

#include <AMReX_FArrayBox.H>

AMREX_GPU_HOST
void advance_RCM(amrex::Real dt, amrex::Real dx, amrex::Real random,
                 amrex::Real adia, amrex::Real eps, const amrex::Box &bx,
                 const amrex::FArrayBox &statein, amrex::FArrayBox &stateout);

#endif