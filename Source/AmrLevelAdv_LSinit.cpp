#include "AmrLevelAdv.H"

#include "BCs.H"
#include "Geometry/MakeSDF.H"
#include "Geometry/Normals.H"

#include <AMReX_MFIter.H>

using namespace amrex;

// For all implicit functions, >0 -> body, =0 -> boundary, <0 -> fluid

template <class F> void fill_from_IF(const Geometry geom, MultiFab &dst, F &&f)
{
    const auto &prob_lo = geom.ProbLoArray();
    const auto &dx      = geom.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(dst, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const auto &bx  = mfi.growntilebox();
            const auto &arr = dst.array(mfi);
            ParallelFor(bx,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                            arr(i, j, k) = f(RealArray{ AMREX_D_DECL(
                                (i + 0.5) * dx[0] + prob_lo[0],
                                (j + 0.5) * dx[1] + prob_lo[1],
                                (k + 0.5) * dx[2] + prob_lo[2]) });
                        });
        }
    }
}

void AmrLevelAdv::init_levelset(amrex::MultiFab       &LS,
                                const amrex::Geometry &geom,
                                bamrexBCData          &bc_data)
{
    ParmParse   ppls("ls");
    std::string geom_type;
    ppls.query("geom_type", geom_type);
    if (geom_type.empty() || geom_type == "all_regular")
    {
        LS.setVal(-1);
        bc_data.set_rb_enabled(false);
        return;
    }
    else
    {
        bc_data.set_rb_enabled(true);
        auto sdf = SDF::make_sdf();
        fill_from_IF(geom, LS, sdf);
        // normal vectors (point into fluid)
        fill_ls_normals(geom, LS);
    }

    AMREX_ASSERT(!LS.contains_nan(0, 1, LS.nGrow()));
    AMREX_ASSERT(!LS.contains_nan(1, AMREX_SPACEDIM, 0));
}