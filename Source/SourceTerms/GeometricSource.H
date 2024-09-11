#ifndef GeometricSource_H_
#define GeometricSource_H_

#include "BCs.H"
#include "Index.H"
#include "Parm.H"

#include <AMReX_AmrLevel.H>
#include <AMReX_Array4.H>
#include <AMReX_FArrayBox.H>

/**
 * \brief Advances Euler equations geometric source terms for cylindrically and
 * spherically symmetric problems.
 *
 * \param geom
 * \param dt
 * \param alpha geometric coefficient. 1 for cylindrical symmetry and 2 for
 * sphercial
 * \param rot_axis which axis the domain is rotated around. In 2D,
 * rot_axis = 0 -> x lo, = 1 -> y lo
 * \param statein
 * \param stateout
 */
AMREX_GPU_HOST
void advance_geometric(const amrex::Geometry &geom, amrex::Real dt, int alpha,
                       int rot_axis, const amrex::MultiFab &statein,
                       amrex::MultiFab &stateout);

/**
 * \brief Advances Navier-Stokes equations geometric source terms for
 * cylindrically symmetric problems. Spherical symmetry is currently not
 * supported, so alpha must be 1.
 *
 * \param geom
 * \param dt
 * \param time
 * \param alpha geometric coefficient. 1 for cylindrical symmetry and 2 for
 * sphercial
 * \param rot_axis which axis the domain is rotated around. In 2D,
 * rot_axis = 0 -> x lo, = 1 -> y lo
 * \param statein
 * \param stateout
 * \param bc
 */
AMREX_GPU_HOST
void advance_geometric_visc(const amrex::Geometry &geom, amrex::Real dt,
                            amrex::Real time, int alpha, int rot_axis,
                            const amrex::MultiFab &statein,
                            amrex::MultiFab &stateout, const bamrexBCData &bc);

/*********************/

AMREX_GPU_HOST
void advance_geometric_1d(amrex::Real dt, amrex::Real dr, amrex::Real r_lo,
                          int alpha, const amrex::Box &bx,
                          const amrex::FArrayBox &statein,
                          amrex::FArrayBox       &stateout);

//! rot_axis = 0 -> x lo, = 1 -> y lo
template <int rot_axis>
AMREX_GPU_HOST void
advance_geometric_2d(amrex::Real dt, amrex::Real dr, amrex::Real r_lo,
                     const amrex::Box &bx, const amrex::FArrayBox &statein,
                     amrex::FArrayBox &stateout);

// =======
// Kernels
// =======

//! Geometric source term for the Euler equations in 1D
[[nodiscard]] AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE amrex::GpuArray<amrex::Real, 3>
    geometric_source(amrex::Real r, int alpha, amrex::Real adia,
                     amrex::Real                             eps,
                     const amrex::Array4<const amrex::Real> &U, int i)
{
    using namespace amrex;
    AMREX_ASSERT(U.nComp() == 3);
    GpuArray<amrex::Real, 3> src;
    const Real               prefactor = -alpha / r;
    const Real               u         = U(i, 0, 0, 1) / U(i, 0, 0, 0);
    src[0]                             = prefactor * U(i, 0, 0, 1); // rho u
    src[1] = prefactor * U(i, 0, 0, 1) * u;                         // rho u^2
    src[2] = prefactor * u
             * (adia * U(i, 0, 0, 2)
                - (adia - 1) * 0.5 * eps * U(i, 0, 0, 1) * u); // (E + p)u

    return src;
}

//! Geometric source term for the Navier-Stokes equations in 1D.
//! Only cylindrical symmetry (alpha = 1) is supported
//! Assumes u_z = u_phi = bulk modulus = 0
//! Q should be the primitive variables (including viscous coefficients)
[[nodiscard]] AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE amrex::GpuArray<amrex::Real, UInd::size>
                       geometric_source_visc(amrex::Real r, amrex::Real drinv,
                                             [[maybe_unused]] int alpha, const Parm &parm,
                                             const amrex::Array4<const amrex::Real> &Q, int i)
{
    using namespace amrex;
    AMREX_ASSERT(Q.nComp() == QInd::vsize);
    AMREX_ASSERT(alpha
                 == 1); // Currently only cylindrical symmetry is supported
    GpuArray<amrex::Real, UInd::size> src;
    constexpr Real                    four_thirds = (Real)4 / 3;
    constexpr Real                    two_thirds  = (Real)2 / 3;
    const Real                        durdr
        = (Q(i + 1, 0, 0, QInd::U) - Q(i - 1, 0, 0, QInd::U)) * drinv;
    const Real detadr
        = (Q(i + 1, 0, 0, QInd::ETA) - Q(i - 1, 0, 0, QInd::ETA)) * drinv;
    const Real dTdr
        = (Q(i + 1, 0, 0, QInd::T) - Q(i - 1, 0, 0, QInd::T)) * drinv;
    const Real eta   = Q(i, 0, 0, QInd::ETA);
    const Real kappa = Q(i, 0, 0, QInd::KAPPA);
    const Real u_r   = Q(i, 0, 0, QInd::U);
    const Real rinv  = (Real)1 / r;
    src[UInd::RHO]   = 0;
    src[UInd::MX]    = parm.Reinv
                    * (four_thirds * eta * rinv * rinv * (r * durdr - u_r)
                       - two_thirds * u_r * rinv * detadr);
    src[UInd::E]
        = parm.epsilon * parm.Reinv * (-two_thirds * u_r * u_r * rinv * detadr)
          + parm.Reinv * (kappa * rinv * dTdr);

    return src;
}

//! Geometric source term for the cylindrically-symmetric Euler equations in 2D
//! \tparam axis_loc = 0 for lo x, 1 for lo y
template <int axis_loc>
[[nodiscard]] AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE amrex::GpuArray<amrex::Real, 4>
    geometric_source(amrex::Real r, amrex::Real adia, amrex::Real eps,
                     const amrex::Array4<const amrex::Real> &U, int i, int j)
{
    using namespace amrex;
    AMREX_ASSERT(U.nComp() == 4);
    GpuArray<amrex::Real, 4> src;
    const Real               prefactor = -(Real)1 / r;
    constexpr int            u_r_idx   = (axis_loc == 0) ? 1 : 2;
    const Real               u_r       = U(i, j, 0, u_r_idx) / U(i, j, 0, 0);
    src[0] = prefactor * U(i, j, 0, u_r_idx); // rho u_r
    src[1] = prefactor * U(i, j, 0, 1) * u_r; // rho u_r u_r
    src[2] = prefactor * U(i, j, 0, 2) * u_r; // rho u_z u_r
    src[3] = prefactor * u_r
             * (adia * U(i, j, 0, 3)
                - (adia - 1) * 0.5 * eps
                      * (U(i, j, 0, 1) * U(i, j, 0, 1)
                         + U(i, j, 0, 2) * U(i, j, 0, 2))
                      / U(i, j, 0, 0)); // (E + p)u_r

    return src;
}

//! Geometric source term for the cylindrically symmetric Navier-Stokes in 2D
//! \tparam axis_loc = 0 for lo x, 1 for lo y
template <int axis_loc>
[[nodiscard]] AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE amrex::GpuArray<amrex::Real, UInd::size>
    geometric_source_visc(amrex::Real r, amrex::Real drinv, amrex::Real dzinv,
                          const Parm                             &parm,
                          const amrex::Array4<const amrex::Real> &Q, int i,
                          int j)
{
    using namespace amrex;
    GpuArray<amrex::Real, UInd::size> src;
    constexpr int                     rInd = (axis_loc == 0) ? 0 : 1;
    constexpr int                     zInd = (axis_loc == 0) ? 1 : 0;
    constexpr int                     UR   = (rInd == 0) ? QInd::U : QInd::V;
    constexpr int                     UZ   = (zInd == 0) ? QInd::U : QInd::V;

    // offsets for central differencing
    constexpr int i_r = (rInd == 0) ? 1 : 0;
    constexpr int j_r = (rInd == 1) ? 1 : 0;
    constexpr int i_z = (zInd == 0) ? 1 : 0;
    constexpr int j_z = (zInd == 1) ? 1 : 0;

    constexpr Real four_thirds = (Real)4 / 3;
    constexpr Real two_thirds  = (Real)2 / 3;
    constexpr Real one_third   = (Real)1 / 3;

    const Real rinv  = (Real)1 / r;
    const Real u_r   = Q(i, j, 0, UR);
    const Real u_z   = Q(i, j, 0, UZ);
    const Real eta   = Q(i, j, 0, QInd::ETA);
    const Real kappa = Q(i, j, 0, QInd::KAPPA);
    const Real durdr
        = (Q(i + i_r, j + j_r, 0, UR) - Q(i - i_r, j - j_r, 0, UR)) * drinv;
    const Real duzdr
        = (Q(i + i_r, j + j_r, 0, UZ) - Q(i - i_r, j - j_r, 0, UZ)) * drinv;
    const Real detadr = (Q(i + i_r, j + j_r, 0, QInd::ETA)
                         - Q(i - i_r, j - j_r, 0, QInd::ETA))
                        * drinv;
    const Real dTdr
        = (Q(i + i_r, j + j_r, 0, QInd::T) - Q(i - i_r, j - j_r, 0, QInd::T))
          * drinv;
    const Real durdz
        = (Q(i + i_z, j + j_z, 0, UR) - Q(i - i_z, j - j_z, 0, UR)) * dzinv;
    const Real duzdz
        = (Q(i + i_z, j + j_z, 0, UZ) - Q(i - i_z, j - j_z, 0, UZ)) * dzinv;
    const Real detadz = (Q(i + i_z, j + j_z, 0, QInd::ETA)
                         - Q(i - i_z, j - j_z, 0, QInd::ETA))
                        * dzinv;

    src[UInd::RHO] = 0;
    src[UR]        = parm.Reinv
              * (four_thirds * eta * rinv * rinv * (r * durdr - u_r)
                 - two_thirds * u_r * rinv * detadr);
    src[UZ] = parm.Reinv
              * (eta * rinv * (duzdr + one_third * durdz)
                 - two_thirds * u_r * rinv * detadz);
    const Real E_viscterm
        = parm.epsilon * parm.Reinv
          * (eta * u_z * rinv * duzdr - two_thirds * u_r * u_r * rinv * detadr
             - four_thirds * eta * u_r * rinv * duzdz
             + one_third * eta * u_z * rinv * durdz
             - two_thirds * u_r * u_z * rinv * detadz);
    const Real E_thermterm = parm.Reinv * kappa * rinv * dTdr;
    src[UInd::E]           = E_viscterm + E_thermterm;

    return src;
}

/**
 * \brief Convenience function combining 1D and 2D geometric_source_visc
 * kernels
 */
[[nodiscard]] AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE amrex::GpuArray<amrex::Real, UInd::size>
    geometric_source_visc(amrex::Real r, amrex::Real drinv, amrex::Real dzinv,
                          int alpha, int rot_axis, const Parm &parm,
                          const amrex::Array4<const amrex::Real> &Q, int i,
                          int j)
{
#if AMREX_SPACEDIM == 1
    amrex::ignore_unused(dzinv, rot_axis, j);
    return geometric_source_visc(r, drinv, alpha, parm, Q, i);
#elif AMREX_SPACEDIM == 2
    amrex::ignore_unused(alpha);
    if (rot_axis == 0)
        return geometric_source_visc<0>(r, drinv, dzinv, parm, Q, i, j);
    else if (rot_axis == 1)
        return geometric_source_visc<1>(r, drinv, dzinv, parm, Q, i, j);
    else
        return amrex::GpuArray<amrex::Real, UInd::size>();
#else
    amrex::ignore_unused(r, drinv, dzinv, alpha, rot_axis, parm, Q, i, j);
    return amrex::GpuArray<amrex::Real, UInd::size>();
#endif
}

#endif