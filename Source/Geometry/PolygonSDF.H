/**
 * SDF calculation for a polygon defined by a set of points
 */
#ifndef Polygon_H_
#define Polygon_H_

#include <AMReX_Array.H>
#include <AMReX_Dim3.H>
#include <AMReX_GpuAsyncArray.H>
#include <AMReX_ParmParse.H>
#include <AMReX_RealVect.H>
#include <limits>

namespace SDF
{

using namespace amrex;

//! Polygon geometry
class PolygonSDF
{
  public:
    //! vertices must be specified in clockwise order
    PolygonSDF(size_t n_vertices, const Vector<GpuArray<Real, 2> > &vertices,
               bool has_fluid_inside)
        : m_n_v(n_vertices)
        , m_v_ptr(std::make_shared<AsyncArray<GpuArray<Real, 2> > >(
              vertices.data(), vertices.size()))
        , m_vertices(m_v_ptr->data())
        , m_sign((has_fluid_inside) ? -1 : 1)
    {
        if (m_n_v < 3)
            throw amrex::RuntimeError("Too few points for a closed polygon");
        AMREX_ASSERT((Long)n_vertices == (Long)vertices.size());
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
    operator()(const RealArray &) const noexcept;

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
    operator()(AMREX_D_DECL(Real x, Real y, Real z)) const noexcept
    {
        return this->operator()(RealArray{ AMREX_D_DECL(x, y, z) });
    }

  private:
    //! Polygon's vertices
    size_t m_n_v;
    //! keeps vertices from being destroyed (cursed but works)
    std::shared_ptr<AsyncArray<GpuArray<Real, 2> > > m_v_ptr;
    const GpuArray<Real, 2>                         *m_vertices;
    int                                              m_sign;

  public:
    //! Helper function
    static size_t AMREX_GPU_HOST
    readParameters(Vector<GpuArray<Real, 2> > &vertices, const ParmParse &pp);
};

/** \defgroup ModulePolygon_Base_Geometry Polygon
    \ingroup ModuleGeometry
    <H2>%Polygon</H2>
    A polygon is defined by the following form:
    \code
    vertices = 8
    vertex_1 = x1 y1
    vertex_2 = x2 y2
    ...
    vertex_8 = x8 y8
    \endcode
    which defines a closed polygon with vertices \f$(x_1,y_1),\;(x_2,
   y_2),\;\ldots,\;(x_8, y_8)\f$.

    The ordering of the points defaults to clockwise, but can be changed by:
    \code
    order = anti-clockwise
    \endcode
*/

//! Signed distance to Polygon.
[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
PolygonSDF::operator()(const RealArray &p) const noexcept
{
    // Determine number of lines we have to cross, moving in x-direction
    // in order to reach -infinity
    // If this is odd, we're inside the polygon. Otherwise outside.

    const amrex::Real x = p[0];
    const amrex::Real y = p[1];

    int         numCrossedLines = 0;
    amrex::Real minDist         = std::numeric_limits<amrex::Real>::max();

    for (size_t i = 0; i < m_n_v; i++)
    {
        size_t j = (i + 1) % m_n_v;

        const amrex::Real x1 = m_vertices[i][0];
        const amrex::Real y1 = m_vertices[i][1];
        const amrex::Real x2 = m_vertices[j][0];
        const amrex::Real y2 = m_vertices[j][1];

        // If line crosses the horizontal line thro' (0,y),
        if (y1 != y2)
        {
            if ((y1 < y && y <= y2) || (y2 < y && y <= y1))
            {
                const amrex::Real l          = (y - y1) / (y2 - y1);
                const amrex::Real xIntercept = x1 + l * (x2 - x1);
                if (xIntercept < x)
                {
                    numCrossedLines++;
                }
            }
        }

        // Calculate distance to current line

        amrex::Real n[2] = { x2 - x1, y2 - y1 };

        amrex::Real nSquared   = n[0] * n[0] + n[1] * n[1];
        amrex::Real offsetDotN = n[0] * (x - x1) + n[1] * (y - y1);
        amrex::Real w;
        if (nSquared == 0.0)
        {
            // Zero normal implies the line is actually a point - i.e.
            // x1==x2,y1==y2 Just set w outside [0,1] so we compute distance to
            // this point
            w = -1.0;
        }
        else
        {
            w = offsetDotN / nSquared;
        }

        // If the perpendicular dropped to the line lands inside the line
        // then use perp. dist as shortest dist.
        if (0 <= w && w <= 1)
        {
            // Offset from (x,y) to nearest point on line
            amrex::Real q[2];
            q[0] = x - x1 - w * n[0];
            q[1] = y - y1 - w * n[1];

            minDist = std::min(minDist, q[0] * q[0] + q[1] * q[1]);
        }
        else if (w < 0)
        {
            // If we're off the start of the line, find dist to start of line
            minDist
                = std::min(minDist, (x - x1) * (x - x1) + (y - y1) * (y - y1));
        }
        else
        {
            // If we're off the end of the line, find dist to end of line
            minDist
                = std::min(minDist, (x - x2) * (x - x2) + (y - y2) * (y - y2));
        }
    }

    if (numCrossedLines % 2 == 1)
    {
        // We're inside, so +ve distance when m_sign +ve
        return m_sign * sqrt(minDist);
    }
    else
    {
        // We're outside so -ve distance when m_sign +ve
        return -m_sign * sqrt(minDist);
    }
}

} // namespace SDF

#endif
