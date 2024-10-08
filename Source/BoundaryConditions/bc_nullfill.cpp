#include "BoundaryConditions/bc_nullfill.H"

#include <AMReX_PhysBCFunct.H>

using namespace amrex;

struct NullFill
{
    AMREX_GPU_DEVICE
    void operator()(const IntVect & /*iv*/, Array4<Real> const & /*dest*/,
                    int /*dcomp*/, int /*numcomp*/,
                    GeometryData const & /*geom*/, Real /*time*/,
                    const BCRec * /*bcr*/, int /*bcomp*/,
                    int /*orig_comp*/) const
    {
        // no physical boundaries to fill because it is all periodic
    }
};

void nullfill(Box const &bx, FArrayBox &data, int dcomp, int numcomp,
              Geometry const &geom, Real time, const Vector<BCRec> &bcr,
              int bcomp, int scomp)
{
    GpuBndryFuncFab<NullFill> gpu_bndry_func(NullFill{});
    gpu_bndry_func(bx, data, dcomp, numcomp, geom, time, bcr, bcomp, scomp);
}
