/**
 * SDF calculation for an intersection of two other SDFs
 */
#ifndef UnionSDF_H_
#define UnionSDF_H_

#include <AMReX_Array.H>
#include <AMReX_Dim3.H>
#include <limits>

namespace SDF
{

using namespace amrex;

/**
 * Intersection of two rigid bodies. Technically incorrect on the inside near
 * the corner when two rigid bodies intersect at non-right angles
 */
template <class F1, class F2> class UnionSDF
{
  public:
    UnionSDF(F1 func1, F2 func2)
        : m_f1(std::move(func1))
        , m_f2(std::move(func2))
    {
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    operator()(AMREX_D_DECL(Real x, Real y, Real z)) const noexcept
    {
        const Real f1           = m_f1(AMREX_D_DECL(x, y, z));
        const Real f2           = m_f2(AMREX_D_DECL(x, y, z));
        const Real outside_dist = min(max(f1, f2), 0.);
        const Real inside_dist
            = sqrt((max(f1, 0.) * max(f1, 0.) + max(f2, 0.) * max(f2, 0.)));
        return inside_dist + outside_dist;
    }

  protected:
    F1 m_f1;
    F2 m_f2;
};

// We use this complicated perfect forwarding because we can avoid extra copies
// when passing temporaries (rvalues) into make_intersection
template <class SDF1, class SDF2>
UnionSDF<std::decay_t<SDF1>, std::decay_t<SDF2> > make_union(SDF1 &&f1,
                                                             SDF2 &&f2)
{
    return UnionSDF<std::decay_t<SDF1>, std::decay_t<SDF2> >(
        std::forward<SDF1>(f1), std::forward<SDF2>(f2));
}

} // namespace SDF

#endif