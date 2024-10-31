#ifndef Plasma_K_H_
#define Plasma_K_H_

#include <AMReX_Array4.H>

#include "System/Euler/Euler.H"
#include "System/Parm.H"
#include "System/PlasmaIndex.H"

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
mhd_ctop(int i, int j, int k, const amrex::Array4<amrex::Real> &Q,
         const amrex::Array4<const amrex::Real> &U, const Parm &parm)
{
    using namespace amrex;
    Q(i, j, k, PQInd::RHO) = U(i, j, k, PUInd::RHO);
    Q(i, j, k, PQInd::U)   = U(i, j, k, PUInd::MX) / Q(i, j, k, PQInd::RHO);
    Q(i, j, k, PQInd::V)   = U(i, j, k, PUInd::MY) / Q(i, j, k, PQInd::RHO);
    Q(i, j, k, PQInd::W)   = U(i, j, k, PUInd::MZ) / Q(i, j, k, PQInd::RHO);
    const Real u_sq        = Q(i, j, k, PQInd::U) * Q(i, j, k, PQInd::U)
                      + Q(i, j, k, PQInd::V) * Q(i, j, k, PQInd::V)
                      + Q(i, j, k, PQInd::W) * Q(i, j, k, PQInd::W);
    const Real B_sq = U(i, j, k, PUInd::BX) * U(i, j, k, PUInd::BX)
                      + U(i, j, k, PUInd::BY) * U(i, j, k, PUInd::BY)
                      + U(i, j, k, PUInd::BZ) * U(i, j, k, PUInd::BZ);
    Q(i, j, k, PQInd::P)
        = parm.adia[2]
          * (U(i, j, k, PUInd::E) - 0.5 * U(i, j, k, PQInd::RHO) * u_sq
             - 0.5 * B_sq);
    Q(i, j, k, PQInd::BX) = U(i, j, k, PQInd::BX);
    Q(i, j, k, PQInd::BY) = U(i, j, k, PQInd::BY);
    Q(i, j, k, PQInd::BZ) = U(i, j, k, PQInd::BZ);
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE amrex::GpuArray<amrex::Real, PQInd::size_noB>
mhd_ctop_noB(int i, int j, int k, const amrex::Array4<const amrex::Real> &U,
             const Parm &parm)
{
    using namespace amrex;
    GpuArray<Real, PQInd::size_noB> Q;
    Q[PQInd::RHO]   = U(i, j, k, PUInd::RHO);
    Q[PQInd::U]     = U(i, j, k, PUInd::MX) / Q[PQInd::RHO];
    Q[PQInd::V]     = U(i, j, k, PUInd::MY) / Q[PQInd::RHO];
    Q[PQInd::W]     = U(i, j, k, PUInd::MZ) / Q[PQInd::RHO];
    const Real u_sq = Q[PQInd::U] * Q[PQInd::U] + Q[PQInd::V] * Q[PQInd::V]
                      + Q[PQInd::W] * Q[PQInd::W];
    const Real B_sq = U(i, j, k, PUInd::BX) * U(i, j, k, PUInd::BX)
                      + U(i, j, k, PUInd::BY) * U(i, j, k, PUInd::BY)
                      + U(i, j, k, PUInd::BZ) * U(i, j, k, PUInd::BZ);
    Q[PQInd::P]
        = parm.adia[2]
          * (U(i, j, k, PUInd::E) - 0.5 * Q[PQInd::RHO] * u_sq - 0.5 * B_sq);
    return Q;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE amrex::GpuArray<amrex::Real, PUInd::size>
mhd_ptoc(const amrex::GpuArray<amrex::Real, PQInd::size> &Q, const Parm &parm)
{
    using namespace amrex;
    GpuArray<Real, PUInd::size> U;
    U[PUInd::RHO]   = Q[PQInd::RHO];
    U[PUInd::MX]    = Q[PQInd::RHO] * Q[PQInd::U];
    U[PUInd::MY]    = Q[PQInd::RHO] * Q[PQInd::V];
    U[PUInd::MZ]    = Q[PQInd::RHO] * Q[PQInd::W];
    const Real u_sq = Q[PQInd::U] * Q[PQInd::U] + Q[PQInd::V] * Q[PQInd::V]
                      + Q[PQInd::W] * Q[PQInd::W];
    const Real B_sq = Q[PQInd::BX] * Q[PQInd::BX] + Q[PQInd::BY] * Q[PQInd::BY]
                      + Q[PQInd::BZ] * Q[PQInd::BZ];
    U[PUInd::E]
        = Q[PQInd::P] * parm.adia[4] + 0.5 * Q[PQInd::RHO] * u_sq + 0.5 * B_sq;
    U[PUInd::BX] = Q[PQInd::BX];
    U[PUInd::BY] = Q[PQInd::BY];
    U[PUInd::BZ] = Q[PQInd::BZ];
    return U;
}

template <int dir>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
mhd_flux(int i, int j, int k, const amrex::Array4<amrex::Real> &flux,
         const amrex::Array4<const amrex::Real> &U, const Parm &parm)
{
    const auto Q = mhd_ctop_noB(i, j, k, U, parm);
    const auto p_tot
        = Q[PQInd::P]
          + 0.5
                * (U(i, j, k, PUInd::BX) * U(i, j, k, PUInd::BX)
                   + U(i, j, k, PUInd::BY) * U(i, j, k, PUInd::BY)
                   + U(i, j, k, PUInd::BZ) * U(i, j, k, PUInd::BZ));
    // density flux
    flux(i, j, k, PUInd::RHO) = U(i, j, k, PUInd::Mdir(dir));
    // momentum flux
    //     rho u x u - B x B term
    flux(i, j, k, PUInd::MX)
        = U(i, j, k, PUInd::MX) * Q[PQInd::Udir(dir)]
          - U(i, j, k, PUInd::BX) * U(i, j, k, PUInd::Bdir(dir));
    flux(i, j, k, PUInd::MY)
        = U(i, j, k, PUInd::MY) * Q[PQInd::Udir(dir)]
          - U(i, j, k, PUInd::BY) * U(i, j, k, PUInd::Bdir(dir));
    flux(i, j, k, PUInd::MZ)
        = U(i, j, k, PUInd::MZ) * Q[PQInd::Udir(dir)]
          - U(i, j, k, PUInd::BZ) * U(i, j, k, PUInd::Bdir(dir));
    //     pressure term
    flux(i, j, k, PUInd::Mdir(dir)) += p_tot;
    // energy flux
    const auto udotB = Q[PQInd::U] * U(i, j, k, PUInd::BX)
                       + Q[PQInd::V] * U(i, j, k, PUInd::BY)
                       + Q[PQInd::W] * U(i, j, k, PUInd::BZ);
    flux(i, j, k, PUInd::E)
        = (U(i, j, k, PUInd::E) + p_tot) * Q[PQInd::Udir(dir)]
          - udotB * U(i, j, k, PUInd::Bdir(dir));
    // induction fluxes
    flux(i, j, k, PUInd::BX) = U(i, j, k, PUInd::BX) * Q[PQInd::Udir(dir)]
                               - Q[PQInd::U] * U(i, j, k, PUInd::Bdir(dir));
    flux(i, j, k, PUInd::BY) = U(i, j, k, PUInd::BY) * Q[PQInd::Udir(dir)]
                               - Q[PQInd::V] * U(i, j, k, PUInd::Bdir(dir));
    flux(i, j, k, PUInd::BZ) = U(i, j, k, PUInd::BZ) * Q[PQInd::Udir(dir)]
                               - Q[PQInd::W] * U(i, j, k, PUInd::Bdir(dir));
}

// template <>
// AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
// mhd_flux<0>(int i, int j, int k, const amrex::Array4<amrex::Real> &flux,
//             const amrex::Array4<const amrex::Real> &U, const Parm &parm)
// {
//     const auto Q = mhd_ctop_noB(i, j, k, U, parm);
//     const auto p_tot
//         = Q[PQInd::P]
//           + 0.5
//                 * (U(i, j, k, PUInd::BX) * U(i, j, k, PUInd::BX)
//                    + U(i, j, k, PUInd::BY) * U(i, j, k, PUInd::BY)
//                    + U(i, j, k, PUInd::BZ) * U(i, j, k, PUInd::BZ));
//     // density flux
//     flux(i, j, k, PUInd::RHO) = U(i, j, k, PUInd::MX);
//     // momentum flux
//     //     rho u x u - B x B term
//     flux(i, j, k, PUInd::MX) = U(i, j, k, PUInd::MX) * Q[PQInd::U] + p_tot
//                                - U(i, j, k, PUInd::BX) * U(i, j, k, PUInd::BX);
//     flux(i, j, k, PUInd::MY) = U(i, j, k, PUInd::MY) * Q[PQInd::U]
//                                - U(i, j, k, PUInd::BY) * U(i, j, k, PUInd::BX);
//     flux(i, j, k, PUInd::MZ) = U(i, j, k, PUInd::MZ) * Q[PQInd::U]
//                                - U(i, j, k, PUInd::BZ) * U(i, j, k, PUInd::BX);
//     // energy flux
//     const auto udotB = Q[PQInd::U] * U(i, j, k, PUInd::BX)
//                        + Q[PQInd::V] * U(i, j, k, PUInd::BY)
//                        + Q[PQInd::W] * U(i, j, k, PUInd::BZ);
//     flux(i, j, k, PUInd::E) = (U(i, j, k, PUInd::E) + p_tot) * Q[PQInd::U]
//                               - udotB * U(i, j, k, PUInd::BX);
//     // induction fluxes
//     flux(i, j, k, PUInd::BX) = 0;
//     flux(i, j, k, PUInd::BY) = U(i, j, k, PUInd::BY) * Q[PQInd::U]
//                                - Q[PQInd::V] * U(i, j, k, PUInd::BX);
//     flux(i, j, k, PUInd::BZ) = U(i, j, k, PUInd::BZ) * Q[PQInd::U]
//                                - Q[PQInd::W] * U(i, j, k, PUInd::BX);
// }

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
                                    mhd_alfven_speed_sq(int i, int j, int k,
                                                        const amrex::Array4<const amrex::Real> &U)
{
    using namespace amrex;
    const Real B_sq = U(i, j, k, PUInd::BX) * U(i, j, k, PUInd::BX)
                      + U(i, j, k, PUInd::BY) * U(i, j, k, PUInd::BY)
                      + U(i, j, k, PUInd::BZ) * U(i, j, k, PUInd::BZ);
    return B_sq / U(i, j, k, PUInd::RHO);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
mhd_sound_speed_sq(const amrex::GpuArray<amrex::Real, PQInd::size_noB> &Q,
                   const Parm                                          &parm)
{
    return parm.adiabatic * Q[PQInd::P] / (Q[PQInd::RHO] * parm.epsilon);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
mhd_fast_speed(int i, int j, int k, int dir, amrex::Real c_s_sq,
               amrex::Real v_a_sq, const amrex::Array4<const amrex::Real> &U)
{
    return std::sqrt(0.5
                     * (c_s_sq + v_a_sq
                        + std::sqrt((c_s_sq + v_a_sq) * (c_s_sq + v_a_sq)
                                    - 4 * c_s_sq * U(i, j, k, PUInd::Bdir(dir))
                                          * U(i, j, k, PUInd::Bdir(dir))
                                          / U(i, j, k, PUInd::RHO))));
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real mhd_exp_timestep_dim_split(
    int i, int j, int k, const amrex::Array4<const amrex::Real> &U,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx, const Parm &parm)
{
    using namespace amrex;
    const Real v_a_sq = mhd_alfven_speed_sq(i, j, k, U);
    const auto Q      = mhd_ctop_noB(i, j, k, U, parm);
    const Real c_s_sq = mhd_sound_speed_sq(Q, parm);
    // Real       dt_est = 1e20;
    // for (int d = 0; d < AMREX_SPACEDIM; ++d)
    // {
    //     dt_est = std::min(
    //         dt_est, dx[d]
    //                     / (std::fabs(Q[PQInd::Udir(d)])
    //                        + mhd_fast_speed(i, j, k, d, c_s_sq, v_a_sq,
    //                        U)));
    // }
    const Real u_mag
        = std::sqrt(Q[PQInd::U] * Q[PQInd::U] + Q[PQInd::V] * Q[PQInd::V]
                    + Q[PQInd::W] * Q[PQInd::W]);
    const Real max_fast_speed
        = std::max(mhd_fast_speed(i, j, k, 0, c_s_sq, v_a_sq, U),
                   std::max(mhd_fast_speed(i, j, k, 1, c_s_sq, v_a_sq, U),
                            mhd_fast_speed(i, j, k, 2, c_s_sq, v_a_sq, U)));
    Real dt_est = 1e20;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        dt_est = std::min(dt_est, dx[d] / (u_mag + max_fast_speed));
    }
    return dt_est;
}

#endif