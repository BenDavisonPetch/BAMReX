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
          * (U(i, j, k, PUInd::E) - 0.5 * U(i, j, k, PUInd::RHO) * u_sq
             - 0.5 * B_sq);
    Q(i, j, k, PQInd::PTOT) = Q(i, j, k, PQInd::P) + 0.5 * B_sq;
    Q(i, j, k, PQInd::BX)   = U(i, j, k, PUInd::BX);
    Q(i, j, k, PQInd::BY)   = U(i, j, k, PUInd::BY);
    Q(i, j, k, PQInd::BZ)   = U(i, j, k, PUInd::BZ);
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
    Q[PQInd::PTOT] = Q[PQInd::P] + 0.5 * B_sq;
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

//! Fills total pressure of Q assuming that gas pressure and magnetic fields
//! have already been specified
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
mhd_fill_p_tot(amrex::GpuArray<amrex::Real, PQInd::size> &Q)
{
    Q[PQInd::PTOT]
        = Q[PQInd::P]
          + 0.5
                * (Q[PQInd::BX] * Q[PQInd::BX] + Q[PQInd::BY] * Q[PQInd::BY]
                   + Q[PQInd::BZ] * Q[PQInd::BZ]);
}

template <int dir>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
mhd_flux(int i, int j, int k, const amrex::Array4<amrex::Real> &flux,
         const amrex::Array4<const amrex::Real> &U, const Parm &parm)
{
    const auto Q     = mhd_ctop_noB(i, j, k, U, parm);
    const auto p_tot = Q[PQInd::PTOT];
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

//! Variant of above but returning a local array with precomputed primitive
//! variables
template <int dir>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::GpuArray<amrex::Real, PUInd::size>
mhd_flux(int i, int j, int k, const amrex::Array4<const amrex::Real> &U,
         const amrex::GpuArray<amrex::Real, PQInd::size_noB> &Q)
{
    using namespace amrex;
    GpuArray<Real, PUInd::size> flux;
    const auto                  p_tot = Q[PQInd::PTOT];
    // density flux
    flux[PUInd::RHO] = U(i, j, k, PUInd::Mdir(dir));
    // momentum flux
    //     rho u x u - B x B term
    flux[PUInd::MX] = U(i, j, k, PUInd::MX) * Q[PQInd::Udir(dir)]
                      - U(i, j, k, PUInd::BX) * U(i, j, k, PUInd::Bdir(dir));
    flux[PUInd::MY] = U(i, j, k, PUInd::MY) * Q[PQInd::Udir(dir)]
                      - U(i, j, k, PUInd::BY) * U(i, j, k, PUInd::Bdir(dir));
    flux[PUInd::MZ] = U(i, j, k, PUInd::MZ) * Q[PQInd::Udir(dir)]
                      - U(i, j, k, PUInd::BZ) * U(i, j, k, PUInd::Bdir(dir));
    //     pressure term
    flux[PUInd::Mdir(dir)] += p_tot;
    // energy flux
    const auto udotB = Q[PQInd::U] * U(i, j, k, PUInd::BX)
                       + Q[PQInd::V] * U(i, j, k, PUInd::BY)
                       + Q[PQInd::W] * U(i, j, k, PUInd::BZ);
    flux[PUInd::E] = (U(i, j, k, PUInd::E) + p_tot) * Q[PQInd::Udir(dir)]
                     - udotB * U(i, j, k, PUInd::Bdir(dir));
    // induction fluxes
    flux[PUInd::BX] = U(i, j, k, PUInd::BX) * Q[PQInd::Udir(dir)]
                      - Q[PQInd::U] * U(i, j, k, PUInd::Bdir(dir));
    flux[PUInd::BY] = U(i, j, k, PUInd::BY) * Q[PQInd::Udir(dir)]
                      - Q[PQInd::V] * U(i, j, k, PUInd::Bdir(dir));
    flux[PUInd::BZ] = U(i, j, k, PUInd::BZ) * Q[PQInd::Udir(dir)]
                      - Q[PQInd::W] * U(i, j, k, PUInd::Bdir(dir));
    return flux;
}

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
    Real       dt_est = 1e20;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        dt_est = std::min(
            dt_est, dx[d]
                        / (std::fabs(Q[PQInd::Udir(d)])
                           + mhd_fast_speed(i, j, k, d, c_s_sq, v_a_sq, U)));
    }
    return dt_est;
}

//! Q_L must be the primitive state vector of U_L at i-i_off, j-j_off, k-k_off
//! where i_off = 1 if dir == 0 and 0 otherwise, j_off = 1 if dir == 1 and 0
//! otherwise, etc
template <int dir>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void mhd_HLLC_speeds(
    int i, int j, int k, const amrex::Array4<const amrex::Real> &U_L,
    const amrex::Array4<const amrex::Real>              &U_R,
    const amrex::GpuArray<amrex::Real, PQInd::size_noB> &Q_L,
    const amrex::GpuArray<amrex::Real, PQInd::size_noB> &Q_R, const Parm &parm,
    amrex::Real &S_L, amrex::Real &S_R, amrex::Real &S_star)
{
    using namespace amrex;
    constexpr int i_off = (dir == 0) ? 1 : 0;
    constexpr int j_off = (dir == 1) ? 1 : 0;
    constexpr int k_off = (dir == 2) ? 1 : 0;

    const Real v_a_sq_L
        = mhd_alfven_speed_sq(i - i_off, j - j_off, k - k_off, U_L);
    const Real v_a_sq_R = mhd_alfven_speed_sq(i, j, k, U_R);
    const Real c_s_sq_L = mhd_sound_speed_sq(Q_L, parm);
    const Real c_s_sq_R = mhd_sound_speed_sq(Q_R, parm);
    const Real c_f_L    = mhd_fast_speed(i - i_off, j - j_off, k - k_off, dir,
                                         c_s_sq_L, v_a_sq_L, U_L);
    const Real c_f_R = mhd_fast_speed(i, j, k, dir, c_s_sq_R, v_a_sq_R, U_R);

    const Real rho_L   = Q_L[PQInd::RHO];
    const Real v_L     = Q_L[PQInd::Udir(dir)];
    const Real p_tot_L = Q_L[PQInd::PTOT];
    const Real B_d_L_sq
        = U_L(i - i_off, j - j_off, k - k_off, PUInd::Bdir(dir))
          * U_L(i - i_off, j - j_off, k - k_off, PUInd::Bdir(dir));

    const Real rho_R   = Q_R[PQInd::RHO];
    const Real v_R     = Q_R[PQInd::Udir(dir)];
    const Real p_tot_R = Q_R[PQInd::PTOT];
    const Real B_d_R_sq
        = U_R(i, j, k, PUInd::Bdir(dir)) * U_R(i, j, k, PUInd::Bdir(dir));

    S_L    = amrex::min(v_L, v_R) - amrex::max(c_f_L, c_f_R);
    S_R    = amrex::max(v_L, v_R) + amrex::max(c_f_L, c_f_R);
    S_star = (rho_R * v_R * (S_R - v_R) - rho_L * v_L * (S_L - v_L) + p_tot_L
              - p_tot_R - B_d_L_sq + B_d_R_sq)
             / (rho_R * (S_R - v_R) - rho_L * (S_L - v_L));
}

template <int dir>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::GpuArray<amrex::Real, 3>
mhd_HLL_B(int i, int j, int k, const amrex::Array4<const amrex::Real> &U_L,
          const amrex::Array4<const amrex::Real>          &U_R,
          const amrex::GpuArray<amrex::Real, PUInd::size> &flux_L,
          const amrex::GpuArray<amrex::Real, PUInd::size> &flux_R,
          amrex::Real S_L, amrex::Real S_R)
{
    constexpr int i_off = (dir == 0) ? 1 : 0;
    constexpr int j_off = (dir == 1) ? 1 : 0;
    constexpr int k_off = (dir == 2) ? 1 : 0;

    amrex::GpuArray<amrex::Real, 3> B_HLL;
    for (int n = 0; n < 3; ++n)
    {
        B_HLL[n]
            = (S_R * U_R(i, j, k, PUInd::Bdir(n))
               - S_L * U_L(i - i_off, j - j_off, k - k_off, PUInd::Bdir(n))
               + flux_L[PUInd::Bdir(n)] - flux_R[PUInd::Bdir(n)])
              / (S_R - S_L);
    }
    return B_HLL;
}

template <int dir>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::GpuArray<amrex::Real, PUInd::size>
                                    mhd_HLLC_U_star(int i, int j, int k,
                                                    const amrex::Array4<const amrex::Real>              &U_K,
                                                    const amrex::GpuArray<amrex::Real, PQInd::size_noB> &Q_K,
                                                    const amrex::GpuArray<amrex::Real, 3> &B_HLL, amrex::Real S_K,
                                                    amrex::Real S_star)
{
    using namespace amrex;
    GpuArray<Real, PUInd::size> U_HLLC;
    U_HLLC[PUInd::BX]         = B_HLL[0];
    U_HLLC[PUInd::BY]         = B_HLL[1];
    U_HLLC[PUInd::BZ]         = B_HLL[2];
    const Real rho_K          = Q_K[PQInd::RHO];
    const Real u_d_K          = Q_K[PQInd::Udir(dir)];
    const Real B_d_K          = U_K(i, j, k, PUInd::Bdir(dir));
    const Real B_HLLC_d       = B_HLL[dir];
    const Real S_K_S_star_inv = (Real)1 / (S_K - S_star);
    const Real p_T_HLLC       = rho_K * (S_K - u_d_K) * (S_star - u_d_K)
                          + Q_K[PQInd::PTOT] - B_d_K * B_d_K
                          + B_HLLC_d * B_HLLC_d;
    // rho_HLLC
    U_HLLC[PUInd::RHO]
        = rho_K * (S_K - Q_K[PQInd::Udir(dir)]) * S_K_S_star_inv;
    // mom_HLLC
    U_HLLC[PUInd::Mdir(dir)] = U_HLLC[PUInd::RHO] * S_star;
    for (int d = 0; d < 3; ++d)
    {
        if (d == dir)
            continue;
        U_HLLC[PUInd::Mdir(d)]
            = rho_K * Q_K[PQInd::Udir(d)] * (S_K - u_d_K) * S_K_S_star_inv
              - (B_HLLC_d * U_HLLC[PUInd::Bdir(d)]
                 - B_d_K * U_K(i, j, k, PUInd::Bdir(d)))
                    * S_K_S_star_inv;
    }
    // U_HLLC
    const Real udotB_HLLC
        = (U_HLLC[PUInd::MX] * B_HLL[0] + U_HLLC[PUInd::MY] * B_HLL[1]
           + U_HLLC[PUInd::MZ] * B_HLL[2])
          / U_HLLC[PUInd::RHO];
    const Real udotB_K = Q_K[PQInd::U] * U_K(i, j, k, PUInd::BX)
                         + Q_K[PQInd::V] * U_K(i, j, k, PUInd::BY)
                         + Q_K[PQInd::W] * U_K(i, j, k, PUInd::BZ);
    U_HLLC[PUInd::E] = U_K(i, j, k, PUInd::E) * (S_K - u_d_K) * S_K_S_star_inv
                       + (p_T_HLLC * S_star - Q_K[PQInd::PTOT] * u_d_K
                          - (B_HLLC_d * udotB_HLLC - B_d_K * udotB_K))
                             * S_K_S_star_inv;
    return U_HLLC;
}

#endif