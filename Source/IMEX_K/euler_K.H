#ifndef euler_K_H_
#define euler_K_H_

#include "AmrLevelAdv.H"
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real kinetic_energy(int i, int j, int k,
                           const amrex::Array4<const amrex::Real> &consv_in,
                           const Parm                             &parm)
{
    using namespace amrex;
    const Real epsilon = parm.epsilon;
    return epsilon
           * (AMREX_D_TERM(consv_in(i, j, k, 1) * consv_in(i, j, k, 1),
                           +consv_in(i, j, k, 2) * consv_in(i, j, k, 2),
                           +consv_in(i, j, k, 3) * consv_in(i, j, k, 3)))
           / (2 * consv_in(i, j, k, 0));
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real pressure(int i, int j, int k,
                     const amrex::Array4<const amrex::Real> &consv_in,
                     const Parm                             &parm)
{
    using namespace amrex;
    const Real adia = parm.adiabatic;
    return (adia - 1)
           * (consv_in(i, j, k, 1 + AMREX_SPACEDIM)
              - kinetic_energy(i, j, k, consv_in, parm));
}

/**
 * \brief Computes volumetric enthalpy from \c consv_in
 */
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real vol_enthalpy(int i, int j, int k,
                         const amrex::Array4<const amrex::Real> &consv_in,
                         const Parm                             &parm)
{
    using namespace amrex;
    const Real adia = parm.adiabatic;
    return adia
           * (consv_in(i, j, k, 1 + AMREX_SPACEDIM)
              - kinetic_energy(i, j, k, consv_in, parm));
}

template <int dir>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
adv_flux(int i, int j, int k, const amrex::Array4<const amrex::Real> &consv_in,
         const amrex::Array4<amrex::Real> &flux, const Parm &parm)
{
    const amrex::Real u_dir
        = consv_in(i, j, k, 1 + dir) / consv_in(i, j, k, 0);

    flux(i, j, k, 0) = consv_in(i, j, k, 1 + dir);
    for (int n = 1; n < AMREX_SPACEDIM + 1; ++n)
        flux(i, j, k, n) = consv_in(i, j, k, n) * u_dir;
    flux(i, j, k, 1 + AMREX_SPACEDIM)
        = u_dir * kinetic_energy(i, j, k, consv_in, parm);
}

#endif