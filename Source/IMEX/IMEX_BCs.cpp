#include "IMEX_BCs.H"

using namespace amrex;

AMREX_GPU_HOST
void extrap_grad(const amrex::Geometry &geom, const amrex::MultiFab &src,
                 amrex::MultiFab &dst)
{
    static_assert(AMREX_SPACEDIM == 1);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        const auto& dxinv = geom.InvCellSizeArray();
        for (MFIter mfi(src, MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            const auto& srcarr = src.const_array(mfi);
            const auto& dstarr = dst.array(mfi);

            const Box blox(IntVect(AMREX_D_DECL(bx.loVect()[0]-1, bx.loVect()[1], bx.loVect()[2])),
                       IntVect(AMREX_D_DECL(bx.loVect()[0]-1, bx.hiVect()[1], bx.hiVect()[2])));
            const Box bhix(IntVect(AMREX_D_DECL(bx.hiVect()[0]+1, bx.loVect()[1], bx.loVect()[2])),
                       IntVect(AMREX_D_DECL(bx.hiVect()[0]+1, bx.hiVect()[1], bx.hiVect()[2])));
            
            ParallelFor(blox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                dstarr(i,j,k) = (srcarr(i+2,j,k) - srcarr(i+1,j,k)) * dxinv[0];
            });
            ParallelFor(bhix, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                dstarr(i,j,k) = (srcarr(i-1,j,k) - srcarr(i-2,j,k)) * dxinv[0];
            });
        }
    }
}