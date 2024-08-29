#include "Visc.H"
#include "AmrLevelAdv.H"
#include "Visc_K.H"
#include <AMReX_Arena.H>

using namespace amrex;

AMREX_GPU_HOST
void visc_primv_from_consv(const amrex::Box &bx, amrex::FArrayBox &Q,
                           const amrex::FArrayBox &U)
{
    const auto &Qarr = Q.array();
    const auto &Uarr = U.const_array();
    const Parm *parm = AmrLevelAdv::d_parm;
    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                { visc_ctop(i, j, k, Qarr, Uarr, *parm); });
}

AMREX_GPU_HOST
void compute_visc_fluxes_exp(
    const amrex::Box                                  &bx,
    const amrex::GpuArray<amrex::Box, AMREX_SPACEDIM> &nbx,
    AMREX_D_DECL(amrex::FArrayBox &fx, amrex::FArrayBox &fy,
                 amrex::FArrayBox &fz),
    const amrex::FArrayBox                       &U,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dxinv)
{
    BL_PROFILE("compute_visc_fluxes_exp()");
    const Box &gbx = amrex::grow(bx, 1);
    FArrayBox  Qfab(gbx, QInd::vsize, The_Async_Arena());
    visc_primv_from_consv(gbx, Qfab, U);
    const auto &Q = Qfab.const_array();

    const Parm *parm = AmrLevelAdv::d_parm;

    [[maybe_unused]] constexpr int x = 0;
    [[maybe_unused]] constexpr int y = 1;
    [[maybe_unused]] constexpr int z = 2;

    constexpr Real twothirds = (Real)2 / 3;

#if AMREX_SPACEDIM == 1
    // x-fluxes
    const auto &fxarr = fx.array();
    ParallelFor(nbx[0],
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    const Real dudx  = deriv<QInd::U, x, x>(i, j, k, Q, dxinv);
                    const Real dTdx  = deriv<QInd::T, x, x>(i, j, k, Q, dxinv);
                    const Real eta   = face_avg<QInd::ETA, x>(i, j, k, Q);
                    const Real zeta  = face_avg<QInd::ZETA, x>(i, j, k, Q);
                    const Real kappa = face_avg<QInd::KAPPA, x>(i, j, k, Q);
                    const Real u     = face_avg<QInd::U, x>(i, j, k, Q);
                    const Real divu  = dudx;

                    const Real tauxx
                        = eta * (2 * dudx - twothirds * divu) + zeta * divu;

                    const Real Reinv = (*parm).Reinv;
                    const Real eps   = (*parm).epsilon;

                    fxarr(i, j, k, UInd::RHO) = 0;
                    fxarr(i, j, k, UInd::MX)  = -tauxx * Reinv;
                    fxarr(i, j, k, UInd::E)
                        = -(tauxx * u) * Reinv * eps - kappa * dTdx * Reinv;
                });
#elif AMREX_SPACEDIM == 2
    // x-fluxes
    const auto &fxarr = fx.array();
    ParallelFor(nbx[0],
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    const Real dudx  = deriv<QInd::U, x, x>(i, j, k, Q, dxinv);
                    const Real dudy  = deriv<QInd::U, y, x>(i, j, k, Q, dxinv);
                    const Real dvdx  = deriv<QInd::V, x, x>(i, j, k, Q, dxinv);
                    const Real dvdy  = deriv<QInd::V, y, x>(i, j, k, Q, dxinv);
                    const Real dTdx  = deriv<QInd::T, x, x>(i, j, k, Q, dxinv);
                    const Real eta   = face_avg<QInd::ETA, x>(i, j, k, Q);
                    const Real zeta  = face_avg<QInd::ZETA, x>(i, j, k, Q);
                    const Real kappa = face_avg<QInd::KAPPA, x>(i, j, k, Q);
                    const Real u     = face_avg<QInd::U, x>(i, j, k, Q);
                    const Real v     = face_avg<QInd::V, x>(i, j, k, Q);
                    const Real divu  = dudx + dvdy;

                    const Real tauxx
                        = eta * (2 * dudx - twothirds * divu) + zeta * divu;
                    const Real tauyx = eta * (dudy + dvdx);

                    const Real Reinv = (*parm).Reinv;
                    const Real eps   = (*parm).epsilon;

                    fxarr(i, j, k, UInd::RHO) = 0;
                    fxarr(i, j, k, UInd::MX)  = -tauxx * Reinv;
                    fxarr(i, j, k, UInd::MY)  = -tauyx * Reinv;
                    fxarr(i, j, k, UInd::E)
                        = -(tauxx * u + tauyx * v) * eps * Reinv
                          - kappa * dTdx * Reinv;
                });

    // y-fluxes
    const auto &fyarr = fy.array();
    ParallelFor(nbx[1],
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    const Real dudx  = deriv<QInd::U, x, y>(i, j, k, Q, dxinv);
                    const Real dudy  = deriv<QInd::U, y, y>(i, j, k, Q, dxinv);
                    const Real dvdx  = deriv<QInd::V, x, y>(i, j, k, Q, dxinv);
                    const Real dvdy  = deriv<QInd::V, y, y>(i, j, k, Q, dxinv);
                    const Real dTdy  = deriv<QInd::T, y, y>(i, j, k, Q, dxinv);
                    const Real eta   = face_avg<QInd::ETA, y>(i, j, k, Q);
                    const Real zeta  = face_avg<QInd::ZETA, y>(i, j, k, Q);
                    const Real kappa = face_avg<QInd::KAPPA, y>(i, j, k, Q);
                    const Real u     = face_avg<QInd::U, y>(i, j, k, Q);
                    const Real v     = face_avg<QInd::V, y>(i, j, k, Q);
                    const Real divu  = dudx + dvdy;

                    const Real tauxy = eta * (dudy + dvdx);
                    const Real tauyy
                        = eta * (2 * dvdy - twothirds * divu) + zeta * divu;

                    const Real Reinv = (*parm).Reinv;
                    const Real eps   = (*parm).epsilon;

                    fyarr(i, j, k, UInd::RHO) = 0;
                    fyarr(i, j, k, UInd::MX)  = -tauxy * Reinv;
                    fyarr(i, j, k, UInd::MY)  = -tauyy * Reinv;
                    fyarr(i, j, k, UInd::E)
                        = -(tauxy * u + tauyy * v) * Reinv * eps
                          - kappa * dTdy * Reinv;
                });

#elif AMREX_SPACEDIM == 3
    // x-fluxes
    const auto &fxarr = fx.array();
    ParallelFor(nbx[0],
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    const Real dudx  = deriv<QInd::U, x, x>(i, j, k, Q, dxinv);
                    const Real dudy  = deriv<QInd::U, y, x>(i, j, k, Q, dxinv);
                    const Real dudz  = deriv<QInd::U, z, x>(i, j, k, Q, dxinv);
                    const Real dvdx  = deriv<QInd::V, x, x>(i, j, k, Q, dxinv);
                    const Real dvdy  = deriv<QInd::V, y, x>(i, j, k, Q, dxinv);
                    const Real dwdx  = deriv<QInd::W, x, x>(i, j, k, Q, dxinv);
                    const Real dwdz  = deriv<QInd::W, z, x>(i, j, k, Q, dxinv);
                    const Real dTdx  = deriv<QInd::T, x, x>(i, j, k, Q, dxinv);
                    const Real eta   = face_avg<QInd::ETA, x>(i, j, k, Q);
                    const Real zeta  = face_avg<QInd::ZETA, x>(i, j, k, Q);
                    const Real kappa = face_avg<QInd::KAPPA, x>(i, j, k, Q);
                    const Real u     = face_avg<QInd::U, x>(i, j, k, Q);
                    const Real v     = face_avg<QInd::V, x>(i, j, k, Q);
                    const Real w     = face_avg<QInd::W, x>(i, j, k, Q);
                    const Real divu  = dudx + dvdy + dwdz;

                    const Real tauxx
                        = eta * (2 * dudx - twothirds * divu) + zeta * divu;
                    const Real tauyx = eta * (dudy + dvdx);
                    const Real tauzx = eta * (dudz + dwdx);

                    const Real Reinv = (*parm).Reinv;
                    const Real eps   = (*parm).epsilon;

                    fxarr(i, j, k, UInd::RHO) = 0;
                    fxarr(i, j, k, UInd::MX)  = -tauxx * Reinv;
                    fxarr(i, j, k, UInd::MY)  = -tauyx * Reinv;
                    fxarr(i, j, k, UInd::MZ)  = -tauzx * Reinv;
                    fxarr(i, j, k, UInd::E)
                        = -(tauxx * u + tauyx * v + tauzx * w) * Reinv * eps
                          - kappa * dTdx * Reinv;
                });

    // y-fluxes
    const auto &fyarr = fy.array();
    ParallelFor(nbx[1],
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    const Real dudx  = deriv<QInd::U, x, y>(i, j, k, Q, dxinv);
                    const Real dudy  = deriv<QInd::U, y, y>(i, j, k, Q, dxinv);
                    const Real dvdx  = deriv<QInd::V, x, y>(i, j, k, Q, dxinv);
                    const Real dvdy  = deriv<QInd::V, y, y>(i, j, k, Q, dxinv);
                    const Real dvdz  = deriv<QInd::V, z, y>(i, j, k, Q, dxinv);
                    const Real dwdy  = deriv<QInd::W, y, y>(i, j, k, Q, dxinv);
                    const Real dwdz  = deriv<QInd::W, z, y>(i, j, k, Q, dxinv);
                    const Real dTdy  = deriv<QInd::T, y, y>(i, j, k, Q, dxinv);
                    const Real eta   = face_avg<QInd::ETA, y>(i, j, k, Q);
                    const Real zeta  = face_avg<QInd::ZETA, y>(i, j, k, Q);
                    const Real kappa = face_avg<QInd::KAPPA, y>(i, j, k, Q);
                    const Real u     = face_avg<QInd::U, y>(i, j, k, Q);
                    const Real v     = face_avg<QInd::V, y>(i, j, k, Q);
                    const Real w     = face_avg<QInd::W, y>(i, j, k, Q);
                    const Real divu  = dudx + dvdy + dwdz;

                    const Real tauxy = eta * (dudy + dvdx);
                    const Real tauyy
                        = eta * (2 * dvdy - twothirds * divu) + zeta * divu;
                    const Real tauzy = eta * (dvdz + dwdy);

                    const Real Reinv = (*parm).Reinv;
                    const Real eps   = (*parm).epsilon;

                    fyarr(i, j, k, UInd::RHO) = 0;
                    fyarr(i, j, k, UInd::MX)  = -tauxy * Reinv;
                    fyarr(i, j, k, UInd::MY)  = -tauyy * Reinv;
                    fyarr(i, j, k, UInd::MZ)  = -tauzy * Reinv;
                    fyarr(i, j, k, UInd::E)
                        = -(tauxy * u + tauyy * v + tauzy * w) * Reinv * eps
                          - kappa * dTdy * Reinv;
                });

    // z-fluxes
    const auto &fzarr = fz.array();
    ParallelFor(nbx[2],
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    const Real dudx  = deriv<QInd::U, x, z>(i, j, k, Q, dxinv);
                    const Real dudz  = deriv<QInd::U, z, z>(i, j, k, Q, dxinv);
                    const Real dvdy  = deriv<QInd::V, y, z>(i, j, k, Q, dxinv);
                    const Real dvdz  = deriv<QInd::V, z, z>(i, j, k, Q, dxinv);
                    const Real dwdx  = deriv<QInd::W, x, z>(i, j, k, Q, dxinv);
                    const Real dwdy  = deriv<QInd::W, y, z>(i, j, k, Q, dxinv);
                    const Real dwdz  = deriv<QInd::W, z, z>(i, j, k, Q, dxinv);
                    const Real dTdz  = deriv<QInd::T, z, z>(i, j, k, Q, dxinv);
                    const Real eta   = face_avg<QInd::ETA, z>(i, j, k, Q);
                    const Real zeta  = face_avg<QInd::ZETA, z>(i, j, k, Q);
                    const Real kappa = face_avg<QInd::KAPPA, z>(i, j, k, Q);
                    const Real u     = face_avg<QInd::U, z>(i, j, k, Q);
                    const Real v     = face_avg<QInd::V, z>(i, j, k, Q);
                    const Real w     = face_avg<QInd::W, z>(i, j, k, Q);
                    const Real divu  = dudx + dvdy + dwdz;

                    const Real tauxz = eta * (dudz + dwdx);
                    const Real tauyz = eta * (dvdz + dwdy);
                    const Real tauzz
                        = eta * (2 * dwdz - twothirds * divu) + zeta * divu;

                    const Real Reinv = (*parm).Reinv;
                    const Real eps   = (*parm).epsilon;

                    fyarr(i, j, k, UInd::RHO) = 0;
                    fyarr(i, j, k, UInd::MX)  = -tauxz * Reinv;
                    fyarr(i, j, k, UInd::MY)  = -tauyz * Reinv;
                    fyarr(i, j, k, UInd::MZ)  = -tauzz * Reinv;
                    fyarr(i, j, k, UInd::E)
                        = -(tauxz * u + tauyz * v + tauzz * w) * Reinv * eps
                          - kappa * dTdz * Reinv;
                });
#endif
}

AMREX_GPU_HOST
amrex::Real max_visc_eigv(const amrex::MultiFab &U)
{
    BL_PROFILE("max_visc_eigv()");
    const Parm    *parm = AmrLevelAdv::d_parm;
    MultiFab       Q(U.boxArray(), U.DistributionMap(), QInd::vsize, 0);
    constexpr Real fourthirds = (Real)4 / 3;

    const auto &maU      = U.const_arrays();
    const auto &maQ      = Q.arrays();
    auto        max_eigv = ParReduce(
               TypeList<ReduceOpMax>{}, TypeList<Real>{}, U, IntVect(0),
               [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k)
                   noexcept -> GpuTuple<Real>
               {
            const Array4<const Real> &Uarr = maU[box_no];
            const Array4<Real>       &Qarr = maQ[box_no];
            visc_ctop(i, j, k, Qarr, Uarr, *parm);
            const Real visc_eigv = fourthirds * Qarr(i, j, k, QInd::ETA)
                                   / Qarr(i, j, k, QInd::RHO) * (*parm).Reinv;
            const Real therm_eigv = Qarr(i, j, k, QInd::KAPPA)
                                    / (Qarr(i, j, k, QInd::RHO) * (*parm).cv)
                                    * (*parm).Reinv;
            return max(visc_eigv, therm_eigv);
        });

    BL_PROFILE_VAR("max_visc_egiv::ReduceRealMax", preduce);
    ParallelDescriptor::ReduceRealMax(max_eigv);
    BL_PROFILE_VAR_STOP(preduce);
    return max_eigv;
}