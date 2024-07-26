/**
 * SDF calculation for a sphere
 */
#ifndef ConeSDF_H_
#define ConeSDF_H_

#include <AMReX_Array.H>
#include <AMReX_Dim3.H>
#include <limits>

namespace SDF
{

using namespace amrex;

class ConeSDF
{
  public:
    /**
     * Axis points in the direction the tip points, half angle is in degrees
     */
    ConeSDF(const XDim3 &tip, const XDim3 &axis, const Real half_angle,
            bool has_fluid_inside)
        : m_tip(tip)
        , m_ax(axis)
        , m_cos(cos(half_angle * M_PI / 180))
        , m_sin(sin(half_angle * M_PI / 180))
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
        const XDim3 relpos({ p.x - m_tip.x, p.y - m_tip.y, p.z - m_tip.z });
        const Real  dot
            = relpos.x * m_ax.x + relpos.y * m_ax.y + relpos.z * m_ax.z;
        const XDim3 cross({ relpos.y * m_ax.z - relpos.z * m_ax.y,
                            relpos.z * m_ax.x - relpos.x * m_ax.z,
                            relpos.x * m_ax.y - relpos.y * m_ax.x });
        const Real  crossmag
            = sqrt(cross.x * cross.x + cross.y * cross.y + cross.z * cross.z);
        return m_sign * ((crossmag * m_sin - dot * m_cos >= 0)
                   ? crossmag * m_cos + dot * m_sin
                   : sqrt(relpos.x * relpos.x + relpos.y * relpos.y
                          + relpos.z * relpos.z));
        // return crossmag * m_cos + dot * m_sin;
    }

  protected:
    XDim3 m_tip, m_ax;
    Real  m_cos, m_sin;
    int   m_sign;
};

} // namespace SDF

#endif