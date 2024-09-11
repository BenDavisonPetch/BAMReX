/**
 * SDF calculation for a sphere
 */
#ifndef PlaneSDF_H_
#define PlaneSDF_H_

#include <AMReX_Array.H>
#include <AMReX_Dim3.H>
#include <limits>

namespace SDF
{

using namespace amrex;

class PlaneSDF
{
  public:
    //! Normal points outwards from rigid body
    PlaneSDF(const XDim3 &plane_point, const XDim3 &normal)
        : m_c(plane_point)
        , m_n(normal)
    {
        // normalize
        const Real mag = sqrt(m_n.x * m_n.x + m_n.y * m_n.y + m_n.z * m_n.z);
        m_n.x /= mag;
        m_n.y /= mag;
        m_n.z /= mag;
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    operator()(AMREX_D_DECL(Real x, Real y, Real z)) const noexcept
    {
        // normal points away from rigid body
        const XDim3 p = makeXDim3({ AMREX_D_DECL(x, y, z) });
        const XDim3 relpos({ p.x - m_c.x, p.y - m_c.y, p.z - m_c.z });
        const Real  dot
            = relpos.x * m_n.x + relpos.y * m_n.y + relpos.z * m_n.z;
        return -1 * dot;
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    operator()(const amrex::RealArray &p) const noexcept
    {
        return this->operator()(AMREX_D_DECL(p[0], p[1], p[2]));
    }

  protected:
    XDim3 m_c, m_n;
};

} // namespace SDF

#endif