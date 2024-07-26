/**
 * SDF calculation for a sphere
 */
#ifndef CylinderSDF_H_
#define CylinderSDF_H_

#include <AMReX_Array.H>
#include <AMReX_Dim3.H>
#include <limits>

namespace SDF
{

using namespace amrex;

class CylinderSDF
{
  public:
    CylinderSDF(const XDim3 &center, const XDim3 &axis, const Real radius,
                bool has_fluid_inside)
        : m_c(center)
        , m_ax(axis)
        , m_r(radius)
        , m_sign(has_fluid_inside ? 1 : -1)
    {
        // normalize
        const Real mag
            = sqrt(m_ax.x * m_ax.x + m_ax.y * m_ax.y + m_ax.z * m_ax.z);
        m_ax.x /= mag;
        m_ax.y /= mag;
        m_ax.z /= mag;
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    operator()(AMREX_D_DECL(Real x, Real y, Real z)) const noexcept
    {
        // calc -ve inside body then flip sign if needed
        const XDim3 p = makeXDim3({ AMREX_D_DECL(x, y, z) });
        const XDim3 relpos({ p.x - m_c.x, p.y - m_c.y, p.z - m_c.z });
        const Real  dot
            = relpos.x * m_ax.x + relpos.y * m_ax.y + relpos.z * m_ax.z;
        // <3D z and mayb y components of dim3 are 0
        const XDim3 tang({ relpos.x - dot * m_ax.x, relpos.y - dot * m_ax.y,
                           relpos.z - dot * m_ax.z });
        const Real  r
            = sqrt(tang.x * tang.x + tang.y * tang.y + tang.z * tang.z) - m_r;
        return r * m_sign;
    }

  protected:
    XDim3 m_c, m_ax;
    Real  m_r;
    int   m_sign;
};

} // namespace SDF

#endif