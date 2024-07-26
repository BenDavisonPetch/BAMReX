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

    AMREX_GPU_DEVICE
    void operator()(const IntVect &iv, Array4<Real> const &dest, int dcomp,
                    int numcomp, GeometryData const &geom, Real /*time*/,
                    const BCRec * /*bcr*/, int /*bcomp*/,
                    int /*orig_comp*/) const
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
            for (int n = dcomp; n < dcomp + numcomp; ++n)
            {
                if (i < ilo)
                {
                    dest(i, j, k, n) = dest(ilo, j, k, n);
                }
                else if (i > ihi)
                {
                    dest(i, j, k, n) = dest(ihi, j, k, n);
                }

                if (j < jlo)
                {
                    dest(i, j, k, n) = dest(i, jlo, k, n);
                }
                else if (j > jhi)
                {
                    dest(i, j, k, n) = dest(i, jhi, k, n);
                }

                if (k < klo)
                {
                    dest(i, j, k, n) = dest(i, j, klo, n);
                }
                else if (k > khi)
                {
                    dest(i, j, k, n) = dest(i, j, khi, n);
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
