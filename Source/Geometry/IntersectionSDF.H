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
 * Intersection of two rigid bodies
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
        // intersection => if both -ve, use smallest abs value, if different
        // signs, use -ve, if both +ve, take smallest
        const Real f1 = m_f1(AMREX_D_DECL(x, y, z));
        const Real f2 = m_f2(AMREX_D_DECL(x, y, z));
        return (f1 < 0 && f2 < 0) ? amrex::max(f1, f2) : amrex::min(f1, f2);
    }

  protected:
    F1 m_f1;
    F2 m_f2;
};

} // namespace SDF

#endif