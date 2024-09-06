#ifndef Visc_H_
#define Visc_H_

#include <AMReX_FArrayBox.H>

AMREX_GPU_HOST
void compute_visc_fluxes_exp(
    const amrex::Box &bx,
    AMREX_D_DECL(amrex::FArrayBox &fx, amrex::FArrayBox &fy,
                 amrex::FArrayBox &fz),
    const amrex::FArrayBox                             &U,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dxinv);

AMREX_GPU_HOST
amrex::Real max_visc_eigv(const amrex::MultiFab &U);

#endif