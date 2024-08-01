/**
 * SDF calculation for a box
 */
#ifndef BoxSDF_H_
#define BoxSDF_H_

#include <AMReX_Array.H>
#include <AMReX_Dim3.H>
#include <limits>

namespace SDF
{

using namespace amrex;

//! adapted from amrex::EB2::BoxIF but gives an actually valid SDF
class BoxSDF
{
  public:
    BoxSDF(const RealArray &box_lo, const RealArray &box_hi,
           bool box_has_fluid_inside)
        : m_lo(makeXDim3(box_lo))
        , m_hi(makeXDim3(box_hi))
        , m_sign(box_has_fluid_inside ? 1 : -1)
    {
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    operator()(AMREX_D_DECL(Real x, Real y, Real z)) const noexcept
    {
        Real r = std::numeric_limits<Real>::lowest();
        // calc +ve outside box
        AMREX_D_TERM(
            const Real x_dist = amrex::max(x - m_hi.x, -(x - m_lo.x));
            , const Real y_dist = amrex::max(y - m_hi.y, -(y - m_lo.y));
            , const Real z_dist = amrex::max(z - m_hi.z, -(z - m_lo.z));)

        AMREX_D_TERM(r = amrex::max(r, x_dist);, r = amrex::max(r, y_dist);
                     , r                           = amrex::max(r, z_dist););

        // account for corners (if more than one of x_dist, y_dist, z_dist is
        // +ve we need a corner)
        r = (r > 0) ? sqrt(AMREX_D_TERM(
                amrex::max(x_dist, 0.0) * amrex::max(x_dist, 0.0),
                +amrex::max(y_dist, 0.0) * amrex::max(y_dist, 0.0),
                +amrex::max(z_dist, 0.0) * amrex::max(z_dist, 0.0)))
                    : r;
        return r * m_sign;
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    operator()(const amrex::RealArray &p) const noexcept
    {
        return this->operator()(AMREX_D_DECL(p[0], p[1], p[2]));
    }

  protected:
    XDim3 m_lo, m_hi;
    int   m_sign;
};

} // namespace SDF

#endif