/**
 * Boundary conditions for a co-axial jet
 */
#include "Euler/Euler.H"
#include "Euler/NComp.H"
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_PhysBCFunct.H>

using namespace amrex;

struct CoaxJetFill
{
    CoaxJetFill(const RealArray &center, Real radius_p, Real radius_s,
                int wall, Real dens, Real vel_p, Real vel_s, Real p, Real adia,
                Real eps)
        : m_c(makeXDim3(center))
        , m_r_p(radius_p)
        , m_r_s(radius_s)
        , m_wall(wall)
    {
        const Real                  v_sign = (m_wall % 2 == 0) ? 1 : -1;
        GpuArray<Real, EULER_NCOMP> p_primv{
            dens,
            AMREX_D_DECL((m_wall == 0 || m_wall == 1) ? v_sign * vel_p : 0,
                         (m_wall == 2 || m_wall == 3) ? v_sign * vel_p : 0,
                         (m_wall == 4 || m_wall == 5) ? v_sign * vel_p : 0),
            p
        };

        GpuArray<Real, EULER_NCOMP> s_primv{
            dens,
            AMREX_D_DECL((m_wall == 0 || m_wall == 1) ? v_sign * vel_s : 0,
                         (m_wall == 2 || m_wall == 3) ? v_sign * vel_s : 0,
                         (m_wall == 4 || m_wall == 5) ? v_sign * vel_s : 0),
            p
        };

        m_pstate = consv_from_primv(p_primv, adia, eps);
        m_sstate = consv_from_primv(s_primv, adia, eps);
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
    operator()(const IntVect &iv, Array4<Real> const &dest, int dcomp,
               int numcomp, GeometryData const &geom, Real /*time*/,
               const BCRec *bcr, int bcomp, int /*orig_comp*/) const
    {
        const auto &dx      = geom.CellSize();
        const auto &prob_lo = geom.ProbLo();

#if AMREX_SPACEDIM == 1
        const int  i = iv[0];
        const int  j = 0;
        const int  k = 0;
        const Real x = (i + 0.5) * dx[0] + prob_lo[0] - m_c.x;
        const Real y = 0;
        const Real z = 0;
#endif
#if AMREX_SPACEDIM == 2
        const int  i = iv[0];
        const int  j = iv[1];
        const int  k = 0;
        const Real x = (i + 0.5) * dx[0] + prob_lo[0] - m_c.x;
        const Real y = (j + 0.5) * dx[1] + prob_lo[1] - m_c.y;
        const Real z = 0;
#endif
#if AMREX_SPACEDIM == 3
        const int  i = iv[0];
        const int  j = iv[1];
        const int  k = iv[2];
        const Real x = (i + 0.5) * dx[0] + prob_lo[0] - m_c.x;
        const Real y = (j + 0.5) * dx[1] + prob_lo[1] - m_c.y;
        const Real z = (k + 0.5) * dx[2] + prob_lo[2] - m_c.z;
#endif

        const auto &domain_lo = geom.Domain().loVect3d();
        const auto &domain_hi = geom.Domain().hiVect3d();
        const int   ilo       = domain_lo[0];
        const int   jlo       = domain_lo[1];
        const int   klo       = domain_lo[2];
        const int   ihi       = domain_hi[0];
        const int   jhi       = domain_hi[1];
        const int   khi       = domain_hi[2];

        bool jet_wall
            = ((i < ilo && m_wall == 0) || (i > ihi && m_wall == 1)
               || (j < jlo && m_wall == 2) || (j > jhi && m_wall == 3)
               || (k < klo && m_wall == 4) || (k > khi && m_wall == 5));

        Real r = 0;
        switch (m_wall)
        {
        case 0:
        case 1: r = sqrt(y * y + z * z); break;
        case 2:
        case 3: r = sqrt(x * x + z * z); break;
        case 4:
        case 5: r = sqrt(x * x + y * y); break;
        }

        if (r > m_r_s || !jet_wall)
        {
            // normal AMReX routine
            for (int n = dcomp; n < numcomp + dcomp; ++n)
            {
                const BCRec &bc = bcr[bcomp + n - dcomp];

                if (i < ilo)
                {
                    switch (bc.lo(0))
                    {
                    case (BCType::foextrap):
                    {
                        dest(i, j, k, n) = dest(ilo, j, k, n);
                        break;
                    }
                    case (BCType::hoextrap):
                    {
                        if (i < ilo - 1)
                        {
                            dest(i, j, k, n) = dest(ilo, j, k, n);
                        }
                        // i == ilo-1
                        else if (ilo + 2 <= amrex::min(dest.end.x - 1, ihi))
                        {
                            dest(i, j, k, n)
                                = Real(0.125)
                                  * (Real(15.) * dest(i + 1, j, k, n)
                                     - Real(10.) * dest(i + 2, j, k, n)
                                     + Real(3.) * dest(i + 3, j, k, n));
                        }
                        else
                        {
                            dest(i, j, k, n)
                                = Real(0.5)
                                  * (Real(3.) * dest(i + 1, j, k, n)
                                     - dest(i + 2, j, k, n));
                        }
                        break;
                    }
                    case (BCType::hoextrapcc):
                    {
                        dest(i, j, k, n) = Real(ilo - i)
                                               * (dest(ilo, j, k, n)
                                                  - dest(ilo + 1, j, k, n))
                                           + dest(ilo, j, k, n);
                        break;
                    }
                    case (BCType::reflect_even):
                    {
                        dest(i, j, k, n) = dest(2 * ilo - i - 1, j, k, n);
                        break;
                    }
                    case (BCType::reflect_odd):
                    {
                        dest(i, j, k, n) = -dest(2 * ilo - i - 1, j, k, n);
                        break;
                    }
                    default:
                    {
                        break;
                    }
                    }
                }
                else if (i > ihi)
                {
                    switch (bc.hi(0))
                    {
                    case (BCType::foextrap):
                    {
                        dest(i, j, k, n) = dest(ihi, j, k, n);
                        break;
                    }
                    case (BCType::hoextrap):
                    {
                        if (i > ihi + 1)
                        {
                            dest(i, j, k, n) = dest(ihi, j, k, n);
                        }
                        // i == ihi+1
                        else if (ihi - 2 >= amrex::max(dest.begin.x, ilo))
                        {
                            dest(i, j, k, n)
                                = Real(0.125)
                                  * (Real(15.) * dest(i - 1, j, k, n)
                                     - Real(10.) * dest(i - 2, j, k, n)
                                     + Real(3.) * dest(i - 3, j, k, n));
                        }
                        else
                        {
                            dest(i, j, k, n)
                                = Real(0.5)
                                  * (Real(3.) * dest(i - 1, j, k, n)
                                     - dest(i - 2, j, k, n));
                        }
                        break;
                    }
                    case (BCType::hoextrapcc):
                    {
                        dest(i, j, k, n) = Real(i - ihi)
                                               * (dest(ihi, j, k, n)
                                                  - dest(ihi - 1, j, k, n))
                                           + dest(ihi, j, k, n);
                        break;
                    }
                    case (BCType::reflect_even):
                    {
                        dest(i, j, k, n) = dest(2 * ihi - i + 1, j, k, n);
                        break;
                    }
                    case (BCType::reflect_odd):
                    {
                        dest(i, j, k, n) = -dest(2 * ihi - i + 1, j, k, n);
                        break;
                    }
                    default:
                    {
                        break;
                    }
                    }
                }

                if (j < jlo)
                {
                    switch (bc.lo(1))
                    {
                    case (BCType::foextrap):
                    {
                        dest(i, j, k, n) = dest(i, jlo, k, n);
                        break;
                    }
                    case (BCType::hoextrap):
                    {
                        if (j < jlo - 1)
                        {
                            dest(i, j, k, n) = dest(i, jlo, k, n);
                        }
                        // j == jlo-1
                        else if (jlo + 2 <= amrex::min(dest.end.y - 1, jhi))
                        {
                            dest(i, j, k, n)
                                = Real(0.125)
                                  * (Real(15.) * dest(i, j + 1, k, n)
                                     - Real(10.) * dest(i, j + 2, k, n)
                                     + Real(3.) * dest(i, j + 3, k, n));
                        }
                        else
                        {
                            dest(i, j, k, n)
                                = Real(0.5)
                                  * (Real(3.) * dest(i, j + 1, k, n)
                                     - dest(i, j + 2, k, n));
                        }
                        break;
                    }
                    case (BCType::hoextrapcc):
                    {
                        dest(i, j, k, n) = Real(jlo - j)
                                               * (dest(i, jlo, k, n)
                                                  - dest(i, jlo + 1, k, n))
                                           + dest(i, jlo, k, n);
                        break;
                    }
                    case (BCType::reflect_even):
                    {
                        dest(i, j, k, n) = dest(i, 2 * jlo - j - 1, k, n);
                        break;
                    }
                    case (BCType::reflect_odd):
                    {
                        dest(i, j, k, n) = -dest(i, 2 * jlo - j - 1, k, n);
                        break;
                    }
                    default:
                    {
                        break;
                    }
                    }
                }
                else if (j > jhi)
                {
                    switch (bc.hi(1))
                    {
                    case (BCType::foextrap):
                    {
                        dest(i, j, k, n) = dest(i, jhi, k, n);
                        break;
                    }
                    case (BCType::hoextrap):
                    {
                        if (j > jhi + 1)
                        {
                            dest(i, j, k, n) = dest(i, jhi, k, n);
                        }
                        // j == jhi+1
                        else if (jhi - 2 >= amrex::max(dest.begin.y, jlo))
                        {
                            dest(i, j, k, n)
                                = Real(0.125)
                                  * (Real(15.) * dest(i, j - 1, k, n)
                                     - Real(10.) * dest(i, j - 2, k, n)
                                     + Real(3.) * dest(i, j - 3, k, n));
                        }
                        else
                        {
                            dest(i, j, k, n)
                                = Real(0.5)
                                  * (Real(3.) * dest(i, j - 1, k, n)
                                     - dest(i, j - 2, k, n));
                        }
                        break;
                    }
                    case (BCType::hoextrapcc):
                    {
                        dest(i, j, k, n) = Real(j - jhi)
                                               * (dest(i, jhi, k, n)
                                                  - dest(i, jhi - 1, k, n))
                                           + dest(i, jhi, k, n);
                        break;
                    }
                    case (BCType::reflect_even):
                    {
                        dest(i, j, k, n) = dest(i, 2 * jhi - j + 1, k, n);
                        break;
                    }
                    case (BCType::reflect_odd):
                    {
                        dest(i, j, k, n) = -dest(i, 2 * jhi - j + 1, k, n);
                        break;
                    }
                    default:
                    {
                        break;
                    }
                    }
                }

                if (k < klo)
                {
                    switch (bc.lo(2))
                    {
                    case (BCType::foextrap):
                    {
                        dest(i, j, k, n) = dest(i, j, klo, n);
                        break;
                    }
                    case (BCType::hoextrap):
                    {
                        if (k < klo - 1)
                        {
                            dest(i, j, k, n) = dest(i, j, klo, n);
                        }
                        // k == klo-1
                        else if (klo + 2 <= amrex::min(dest.end.z - 1, khi))
                        {
                            dest(i, j, k, n)
                                = Real(0.125)
                                  * (Real(15.) * dest(i, j, k + 1, n)
                                     - Real(10.) * dest(i, j, k + 2, n)
                                     + Real(3.) * dest(i, j, k + 3, n));
                        }
                        else
                        {
                            dest(i, j, k, n)
                                = Real(0.5)
                                  * (Real(3.) * dest(i, j, k + 1, n)
                                     - dest(i, j, k + 2, n));
                        }
                        break;
                    }
                    case (BCType::hoextrapcc):
                    {
                        dest(i, j, k, n) = Real(klo - k)
                                               * (dest(i, j, klo, n)
                                                  - dest(i, j, klo + 1, n))
                                           + dest(i, j, klo, n);
                        break;
                    }
                    case (BCType::reflect_even):
                    {
                        dest(i, j, k, n) = dest(i, j, 2 * klo - k - 1, n);
                        break;
                    }
                    case (BCType::reflect_odd):
                    {
                        dest(i, j, k, n) = -dest(i, j, 2 * klo - k - 1, n);
                        break;
                    }
                    default:
                    {
                        break;
                    }
                    }
                }
                else if (k > khi)
                {
                    switch (bc.hi(2))
                    {
                    case (BCType::foextrap):
                    {
                        dest(i, j, k, n) = dest(i, j, khi, n);
                        break;
                    }
                    case (BCType::hoextrap):
                    {
                        if (k > khi + 1)
                        {
                            dest(i, j, k, n) = dest(i, j, khi, n);
                        }
                        // k == khi+1
                        else if (khi - 2 >= amrex::max(dest.begin.z, klo))
                        {
                            dest(i, j, k, n)
                                = Real(0.125)
                                  * (Real(15.) * dest(i, j, k - 1, n)
                                     - Real(10.) * dest(i, j, k - 2, n)
                                     + Real(3.) * dest(i, j, k - 3, n));
                        }
                        else
                        {
                            dest(i, j, k, n)
                                = Real(0.5)
                                  * (Real(3.) * dest(i, j, k - 1, n)
                                     - dest(i, j, k - 2, n));
                        }
                        break;
                    }
                    case (BCType::hoextrapcc):
                    {
                        dest(i, j, k, n) = Real(k - khi)
                                               * (dest(i, j, khi, n)
                                                  - dest(i, j, khi - 1, n))
                                           + dest(i, j, khi, n);
                        break;
                    }
                    case (BCType::reflect_even):
                    {
                        dest(i, j, k, n) = dest(i, j, 2 * khi - k + 1, n);
                        break;
                    }
                    case (BCType::reflect_odd):
                    {
                        dest(i, j, k, n) = -dest(i, j, 2 * khi - k + 1, n);
                        break;
                    }
                    default:
                    {
                        break;
                    }
                    }
                }
            }
        }
        else if (r > m_r_p) // secondary jet
        {
            for (int n = dcomp; n < dcomp + numcomp; ++n)
            {
                dest(i, j, k, n) = m_sstate[n];
            }
        }
        else // primary jet
        {
            for (int n = dcomp; n < dcomp + numcomp; ++n)
            {
                dest(i, j, k, n) = m_pstate[n];
            }
        }
    }

  protected:
    const XDim3 m_c;          // center
    const Real  m_r_p, m_r_s; // primary and secondary jet radius
    const int m_wall; // 0 : x_lo, 1: x_hi, 2: y_lo, 3: y_hi, 4: z_lo, 5: z_hi
    GpuArray<Real, EULER_NCOMP> m_pstate, m_sstate;
};

BndryFuncFabDefault make_coax_jetfill_bcfunc(const RealArray &center,
                                             Real radius_p, Real radius_s,
                                             int wall, Real dens, Real vel_p,
                                             Real vel_s, Real p, Real adia,
                                             Real eps)
{
    GpuBndryFuncFab<CoaxJetFill> gpu_bndry_func(CoaxJetFill{
        center, radius_p, radius_s, wall, dens, vel_p, vel_s, p, adia, eps });
    return gpu_bndry_func;
}
