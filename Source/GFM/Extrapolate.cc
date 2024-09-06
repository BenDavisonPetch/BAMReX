#include "Extrapolate.H"
#include <AMReX_GpuAsyncArray.H>
#include <AMReX_MFIter.H>

using namespace amrex;

AMREX_GPU_HOST
void extrapolate_from_boundary(const amrex::Geometry  &geom,
                               amrex::MultiFab        &data,
                               const amrex::MultiFab  &levelset,
                               const amrex::iMultiFab &gfm_flags,
                               const amrex::Vector<amrex::Real> &grads, int startcomp,
                               int ncomp)
{
    AsyncArray<Real> dgrads_aa(grads.data(), grads.size());
    Real* dgrads = (grads.size() > 0) ? dgrads_aa.data() : nullptr;
    const auto& dx = geom.CellSizeArray();
    for (int idim = 0; idim < AMREX_SPACEDIM+1; ++idim) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        {
            for (MFIter mfi(data, false); mfi.isValid(); ++mfi) {
                const auto& bx = mfi.validbox();
#ifdef AMREX_USE_GPU
#error "This section does not launch GPU kernels!"
#endif
                extrapolate_from_boundary(bx, data.array(mfi), levelset.const_array(mfi), gfm_flags.const_array(mfi), dx, dgrads, startcomp, ncomp);
            }
        }

        data.FillBoundary(geom.periodicity());
    }
}