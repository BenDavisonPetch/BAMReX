/**
 * Parameter reading for polygon vertices
 */
#include "PolygonSDF.H"

namespace SDF
{

using namespace amrex;

size_t AMREX_GPU_HOST PolygonSDF::readParameters(
    Vector<GpuArray<Real, 2> > &vertices, const ParmParse &pp)
{
    long n_vertices;
    pp.get("vertices", n_vertices);

    bool anticlockwise = false;
    pp.query("anticlockwise", anticlockwise);

    if (n_vertices < 3)
    {
        throw amrex::RuntimeError("Too few points for a closed polygon");
    }

    vertices.clear();
    vertices.resize(n_vertices);
    for (size_t v = 0; v < (size_t)n_vertices; v++)
    {
        std::ostringstream vname;
        // Number vertices from 1
        vname << "vertex_" << v + 1;
        std::array<amrex::Real, 2> vert;
        pp.get(vname.str().c_str(), vert);
        amrex::GpuArray<amrex::Real, 2> vertG{ vert[0], vert[1] };

        if (!anticlockwise)
        {
            vertices[v] = vertG;
        }
        else
        {
            vertices[n_vertices - 1 - v] = vertG;
        }
    }
    return n_vertices;
}

} // namespace SDF