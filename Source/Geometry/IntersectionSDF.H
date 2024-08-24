/**
 * SDF calculation for an intersection of two other SDFs
 */
#ifndef IntersectionSDF_H_
#define IntersectionSDF_H_

#include <AMReX_Array.H>
#include <AMReX_Dim3.H>
#include <limits>

namespace SDF
{

using namespace amrex;

/**
 * Intersection of two rigid bodies. Incorrect (meaning not exactly an SDF)
 * outside the body for non-right angled intersections
 */
template <class F1, class F2> class IntersectionSDF
{
  public:
    IntersectionSDF(F1 func1, F2 func2)
        : m_f1(std::move(func1))
        , m_f2(std::move(func2))
    {
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    operator()(AMREX_D_DECL(Real x, Real y, Real z)) const noexcept
    {
        const Real f1 = m_f1(AMREX_D_DECL(x, y, z));
        const Real f2 = m_f2(AMREX_D_DECL(x, y, z));

        const Real inside_dist = max(min(f1, f2), 0.);
        const Real outside_dist
            = -sqrt((min(f1, 0.) * min(f1, 0.) + min(f2, 0.) * min(f2, 0.)));
        return inside_dist + outside_dist;
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    operator()(const amrex::RealArray &p) const noexcept
    {
        return this->operator()(AMREX_D_DECL(p[0], p[1], p[2]));
    }

  protected:
    F1 m_f1;
    F2 m_f2;
};

// We use this complicated perfect forwarding because we can avoid extra copies
// when passing temporaries (rvalues) into make_intersection
template <class SDF1, class SDF2>
IntersectionSDF<std::decay_t<SDF1>, std::decay_t<SDF2> >
make_intersection(SDF1 &&f1, SDF2 &&f2)
{
    return IntersectionSDF<std::decay_t<SDF1>, std::decay_t<SDF2> >(
        std::forward<SDF1>(f1), std::forward<SDF2>(f2));
}

} // namespace SDF

#endif