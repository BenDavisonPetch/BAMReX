/**
 * Boundary conditions for a co-axial jet
 */
#include "bc_jet.H"
#include "Euler/Euler.H"
#include "Euler/NComp.H"
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_PhysBCFunct.H>

using namespace amrex;
using namespace bamrex;

/**
 * Fills boundaries for the coaxial jet problem.
 *
 * The primary and secondary jet are specified as velocity inflow boundary
 * conditions with known temperature. They are implemented such that density is
 * extrapolated from the domain to the boundary, and then pressure is
 * calculated from density using the temperature and specific gas constant.
 *
 * The specific gas constant should be in J/kg/K, such that p = rho * R_spec *
 * T. It is equal to k_B/m_particle.
 *
 * All boundaries other than the inflow are pressure outlet boundaries. Density
 * and velocity and extrapolated to the boundary and pressure is set to the
 * ambient pressure.
 *
 * wall specifies which wall the jet is on: 0 : x_lo, 1: x_hi, 2: y_lo, 3:
 * y_hi, 4: z_lo, 5: z_hi
 */
struct CoaxJetFill
{
    CoaxJetFill(const RealArray &center, Real radius_p, Real radius_s,
                int wall, Real v_p, Real v_s, Real T_p, Real T_s, Real R_spec,
                Real p_amb, Real adia, Real eps)
        : m_c(makeXDim3(center))
        , m_r_p(radius_p)
        , m_r_s(radius_s)
        , m_v_sign((wall % 2 == 0) ? 1 : -1)
        , m_v_p(
              { AMREX_D_DECL((wall == 0 || wall == 1) ? m_v_sign * v_p : 0,
                             (wall == 2 || wall == 3) ? m_v_sign * v_p : 0,
                             (wall == 4 || wall == 5) ? m_v_sign * v_p : 0) })
        , m_v_s(
              { AMREX_D_DECL((wall == 0 || wall == 1) ? m_v_sign * v_s : 0,
                             (wall == 2 || wall == 3) ? m_v_sign * v_s : 0,
                             (wall == 4 || wall == 5) ? m_v_sign * v_s : 0) })
        , m_T_p(T_p)
        , m_T_s(T_s)
        , m_R_spec(R_spec)
        , m_p_amb(p_amb)
        , m_adia(adia)
        , m_eps(eps)
        , m_wall(wall)
    {
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
    operator()(const IntVect &iv, Array4<Real> const &dest, int dcomp,
               int numcomp, GeometryData const &geom, Real /*time*/,
               const BCRec *bcr, int bcomp, int /*orig_comp*/) const
    {
        const auto &dx      = geom.CellSize();
        const auto &prob_lo = geom.ProbLo();

        AMREX_ASSERT(numcomp == EULER_NCOMP);
        AMREX_ASSERT(dcomp == 0);

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

        // bcrec of density component. If it is not user_1, we shouldn't do
        // anything here
        const BCRec &bc = bcr[bcomp - dcomp];
        if (i < ilo && bc.lo(0) != BCType::user_1)
            return;
        if (j < jlo && bc.lo(1) != BCType::user_1)
            return;
        if (k < klo && bc.lo(2) != BCType::user_1)
            return;
        if (i > ihi && bc.hi(0) != BCType::user_1)
            return;
        if (j > jhi && bc.hi(1) != BCType::user_1)
            return;
        if (k > khi && bc.hi(2) != BCType::user_1)
            return;

        [[maybe_unused]] const int ireal
            = (i < ilo) ? ilo : ((i > ihi) ? ihi : i);
        [[maybe_unused]] const int jreal
            = (j < jlo) ? jlo : ((j > jhi) ? jhi : j);
        [[maybe_unused]] const int kreal
            = (k < klo) ? klo : ((k > khi) ? khi : k);
        const IntVect ivreal(AMREX_D_DECL(ireal, jreal, kreal));

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

        // extrapolate from real cell at boundary
        auto gh_primv = primv_from_consv(dest, m_adia, m_eps, ivreal);

        if (r > m_r_s || !jet_wall)
        {
            // pressure outflow
            // density and velocity are extrapolated, but we set pressure
            gh_primv[1 + AMREX_SPACEDIM] = m_p_amb;
        }
        else if (r > m_r_p) // secondary jet
        {
            // velocity inflow with specified temperature
            // copy velocity
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                gh_primv[1 + d] = m_v_s[d];
            // calc pressure
            gh_primv[1 + AMREX_SPACEDIM] = gh_primv[0] * m_R_spec * m_T_s;
        }
        else // primary jet
        {
            // velocity inflow with specified temperature
            // copy velocity
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                gh_primv[1 + d] = m_v_p[d];
            // calc pressure
            gh_primv[1 + AMREX_SPACEDIM] = gh_primv[0] * m_R_spec * m_T_p;
        }

        auto gh_consv = consv_from_primv(gh_primv, m_adia, m_eps);

        for (int n = dcomp; n < dcomp + numcomp; ++n)
            dest(i, j, k, n) = gh_consv[n];
    }

  protected:
    const XDim3 m_c;          // center
    const Real  m_r_p, m_r_s; // primary and secondary jet radius
    const Real  m_v_sign;
    const GpuArray<Real, AMREX_SPACEDIM> m_v_p, m_v_s;
    const Real m_T_p, m_T_s, m_R_spec, m_p_amb, m_adia, m_eps;
    const int  m_wall; // 0 : x_lo, 1: x_hi, 2: y_lo, 3: y_hi, 4: z_lo, 5: z_hi
};

BndryFuncFabDefault make_coax_jetfill_bcfunc(const RealArray &center,
                                             Real radius_p, Real radius_s,
                                             int wall, Real v_p, Real v_s,
                                             Real T_p, Real T_s, Real R_spec,
                                             Real p_amb, Real adia, Real eps)
{
    GpuBndryFuncFab<CoaxJetFill> gpu_bndry_func(
        CoaxJetFill{ center, radius_p, radius_s, wall, v_p, v_s, T_p, T_s,
                     R_spec, p_amb, adia, eps });
    return gpu_bndry_func;
}

/**
 * Fills pressure boundaries for the coaxial jet problem.
 *
 * The primary and secondary jet are specified as velocity inflow boundary
 * conditions with known temperature. They are implemented such that density is
 * extrapolated from the domain to the boundary, and then pressure is
 * calculated from density using the temperature and specific gas constant.
 *
 * The specific gas constant should be in J/kg/K, such that p = rho * R_spec *
 * T. It is equal to k_B/m_particle.
 *
 * All boundaries other than the inflow are pressure outlet boundaries. Density
 * is dirichlet with value given by ambient pressure
 *
 * wall specifies which wall the jet is on: 0 : x_lo, 1: x_hi, 2: y_lo, 3:
 * y_hi, 4: z_lo, 5: z_hi
 */
struct CoaxJetPFill
{
    CoaxJetPFill(const RealArray &center, Real radius_p, Real radius_s,
                 int wall, Real T_p, Real T_s, Real R_spec, Real p_amb,
                 Real adia, Real eps)
        : m_c(makeXDim3(center))
        , m_r_p(radius_p)
        , m_r_s(radius_s)
        , m_T_p(T_p)
        , m_T_s(T_s)
        , m_R_spec(R_spec)
        , m_p_amb(p_amb)
        , m_adia(adia)
        , m_eps(eps)
        , m_wall(wall)
    {
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
    operator()(const IntVect &iv, Array4<Real> const &p,
               Array4<const Real> const &U, int dcomp, int numcomp,
               GeometryData const &geom, Real /*time*/, const BCRec *bcr,
               int bcomp, int /*orig_comp*/) const
    {
        const auto &dx      = geom.CellSize();
        const auto &prob_lo = geom.ProbLo();

        AMREX_ASSERT(numcomp == 1);
        AMREX_ASSERT(p.nComp() == 1);
        AMREX_ASSERT(U.nComp() == EULER_NCOMP);
        AMREX_ASSERT(dcomp == 0);

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

        // bcrec of pressure. If it is not user_1, we shouldn't do
        // anything here
        const BCRec &bc = bcr[bcomp - dcomp];
        if (i < ilo && bc.lo(0) != BCType::user_1)
            return;
        if (j < jlo && bc.lo(1) != BCType::user_1)
            return;
        if (k < klo && bc.lo(2) != BCType::user_1)
            return;
        if (i > ihi && bc.hi(0) != BCType::user_1)
            return;
        if (j > jhi && bc.hi(1) != BCType::user_1)
            return;
        if (k > khi && bc.hi(2) != BCType::user_1)
            return;

        [[maybe_unused]] const int ireal
            = (i < ilo) ? ilo : ((i > ihi) ? ihi : i);
        [[maybe_unused]] const int jreal
            = (j < jlo) ? jlo : ((j > jhi) ? jhi : j);
        [[maybe_unused]] const int kreal
            = (k < klo) ? klo : ((k > khi) ? khi : k);
        const IntVect ivreal(AMREX_D_DECL(ireal, jreal, kreal));

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

        // extrapolate from real cell at boundary
        const Real extrap_dens = U(ivreal, 0);

        if (r > m_r_s || !jet_wall)
        {
            p(i, j, k) = m_p_amb;
            return;
        }
        else if (r > m_r_p) // secondary jet
        {
            // velocity inflow with specified temperature
            // calc pressure
            p(i, j, k) = extrap_dens * m_R_spec * m_T_s;
            return;
        }
        else // primary jet
        {
            // velocity inflow with specified temperature
            // calc pressure
            p(i, j, k) = extrap_dens * m_R_spec * m_T_p;
            return;
        }
    }

  protected:
    const XDim3 m_c;          // center
    const Real  m_r_p, m_r_s; // primary and secondary jet radius
    const Real  m_T_p, m_T_s, m_R_spec, m_p_amb, m_adia, m_eps;
    const int m_wall; // 0 : x_lo, 1: x_hi, 2: y_lo, 3: y_hi, 4: z_lo, 5: z_hi
};

//! For the pressure
bamrex::MultiCCBndryFuncFabDefault make_coax_jetfill_bcfunc_p(
    const amrex::RealArray &center, amrex::Real radius_p, amrex::Real radius_s,
    int wall, amrex::Real T_p, amrex::Real T_s, amrex::Real R_spec,
    amrex::Real p_amb, amrex::Real adia, amrex::Real eps)
{
    MultiCCBndryFuncFab<CoaxJetPFill> multi_bndry_func(
        CoaxJetPFill{ center, radius_p, radius_s, wall, T_p, T_s, R_spec,
                      p_amb, adia, eps });
    return multi_bndry_func;
}