/**
 * SDF calculation for a sphere
 */
#ifndef SphereSDF_H_
#define SphereSDF_H_

#include <AMReX_Array.H>
#include <AMReX_Dim3.H>
#include <limits>

namespace SDF
{

using namespace amrex;

//! adapted from amrex::EB2 IFs but gives valid SDF
class SphereSDF
{
  public:
    SphereSDF(const RealArray &center, const Real radius,
              bool sphere_has_fluid_inside)
        : m_c(makeXDim3(center))
        , m_r(radius)
        , m_sign(sphere_has_fluid_inside ? 1 : -1)
    {
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    operator()(AMREX_D_DECL(Real x, Real y, Real z)) const noexcept
    {
        // calc -ve inside body then flip sign if needed
        const Real r = sqrt(AMREX_D_TERM((x - m_c.x) * (x - m_c.x),
                                         +(y - m_c.y) * (y - m_c.y),
                                         +(z - m_c.z) * (z - m_c.z)))
                       - m_r;
        return r * m_sign;
    }

  protected:
    XDim3 m_c;
    Real  m_r;
    int   m_sign;
};

} // namespace SDF

#endif