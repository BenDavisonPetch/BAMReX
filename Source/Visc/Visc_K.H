#ifndef Visc_K_H_
#define Visc_K_H_

#include <AMReX_Array4.H>

#include "System/Index.H"
#include "System/Parm.H"

AMREX_FORCE_INLINE AMREX_GPU_DEVICE void
visc_ctop(int i, int j, int k, const amrex::Array4<amrex::Real> &Q,
          const amrex::Array4<const amrex::Real> &U, const Parm &parm) noexcept
{
    using namespace amrex;
    Q(i, j, k, QInd::RHO) = U(i, j, k, UInd::RHO);
    Q(i, j, k, QInd::U)   = U(i, j, k, UInd::MX) / U(i, j, k, UInd::RHO);
    Real u_sq             = Q(i, j, k, QInd::U) * Q(i, j, k, QInd::U);
#if AMREX_SPACEDIM >= 2
    Q(i, j, k, QInd::V) = U(i, j, k, UInd::MY) / U(i, j, k, UInd::RHO);
    u_sq += Q(i, j, k, QInd::V) * Q(i, j, k, QInd::V);
#endif
#if AMREX_SPACEDIM == 3
    Q(i, j, k, QInd::W) = U(i, j, k, UInd::MZ) / U(i, j, k, UInd::RHO);
    u_sq += Q(i, j, k, QInd::W) * Q(i, j, k, QInd::W);
#endif
    Q(i, j, k, QInd::P)
        = parm.adia[2]
          * (U(i, j, k, UInd::E)
             - 0.5 * parm.epsilon * U(i, j, k, UInd::RHO) * u_sq);

    Q(i, j, k, QInd::T)
        = Q(i, j, k, QInd::P) / (Q(i, j, k, QInd::RHO) * parm.R_spec);

    //! Compute viscous coefficients
    //! Sutherland law
    Q(i, j, k, QInd::ETA)
        = parm.eta_prime
          * std::pow(Q(i, j, k, QInd::T) / parm.T_prime, parm.beta)
          * (parm.T_prime + parm.S_eta) / (Q(i, j, k, QInd::T) + parm.S_eta);
    //! Assume bulk viscosity zero
    Q(i, j, k, QInd::ZETA) = 0;
    //! Use Prandtl number
    Q(i, j, k, QInd::KAPPA) = parm.cp * Q(i, j, k, QInd::ETA) / parm.Pr;
}

template <int idx, int dir, int fluxdir>
AMREX_FORCE_INLINE AMREX_GPU_DEVICE amrex::Real
deriv(int i, int j, int k, const amrex::Array4<const amrex::Real> &Q,
      const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dxinv) noexcept
{
    constexpr int i_off = (dir == 0) ? 1 : 0;
    constexpr int j_off = (dir == 1) ? 1 : 0;
    constexpr int k_off = (dir == 2) ? 1 : 0;

    if (fluxdir == dir)
    {
        return dxinv[dir]
               * (Q(i, j, k, idx) - Q(i - i_off, j - j_off, k - k_off, idx));
    }
    else
    {
        constexpr int i_offf = (fluxdir == 0) ? 1 : 0;
        constexpr int j_offf = (fluxdir == 1) ? 1 : 0;
        constexpr int k_offf = (fluxdir == 2) ? 1 : 0;
        return 0.25 * dxinv[dir]
               * (Q(i + i_off, j + j_off, k + k_off, idx)
                  - Q(i - i_off, j - j_off, k - k_off, idx)
                  + Q(i + i_off - i_offf, j + j_off - j_offf,
                      k + k_off - k_offf, idx)
                  - Q(i - i_off - i_offf, j - j_off - j_offf,
                      k - k_off - k_offf, idx));
    }
}

template <int idx, int fluxdir>
AMREX_FORCE_INLINE AMREX_GPU_DEVICE amrex::Real
                                    face_avg(int i, int j, int k,
                                             const amrex::Array4<const amrex::Real> &Q) noexcept
{
    constexpr int i_off = (fluxdir == 0) ? 1 : 0;
    constexpr int j_off = (fluxdir == 1) ? 1 : 0;
    constexpr int k_off = (fluxdir == 2) ? 1 : 0;
    return 0.5 * (Q(i - i_off, j - j_off, k - k_off, idx) + Q(i, j, k, idx));
}

#endif