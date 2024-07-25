#include "AmrLevelAdv.H"

#include "Geometry/BoxSDF.H"
#include "Geometry/Normals.H"
#include "Geometry/SphereSDF.H"
#include "Geometry/IntersectionSDF.H"

// #include <AMReX_EB2_IF_Box.H>
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
                            arr(i, j, k) = f(
                                AMREX_D_DECL((i + 0.5) * dx[0] + prob_lo[0],
                                             (j + 0.5) * dx[1] + prob_lo[1],
                                             (k + 0.5) * dx[2] + prob_lo[2]));
                        });
        }
    }
}

void AmrLevelAdv::init_levelset(amrex::MultiFab       &LS,
                                const amrex::Geometry &geom)
{
    ParmParse   ppls("ls");
    std::string geom_type;
    ppls.query("geom_type", geom_type);
    if (geom_type.empty() || geom_type == "all_regular")
    {
        LS.setVal(1);
        return;
    }
    else if (geom_type == "box")
    {
        RealArray lo;
        ppls.get("box_lo", lo);

        RealArray hi;
        ppls.get("box_hi", hi);

        // Note that BoxIF does not give the SDF when outside & with one of the
        // corners being the closest point, so normal vector is wrong
        // using has_fluid_inside might give incorrect results at corners
        bool has_fluid_inside;
        ppls.get("box_has_fluid_inside", has_fluid_inside);

        // EB2::BoxIF bf(lo, hi, has_fluid_inside);
        SDF::BoxSDF bf(lo, hi, has_fluid_inside);

        fill_from_IF(geom, LS, bf);

        // normal vectors (point into fluid)
        fill_ls_normals(geom, LS);
    }
    else if (geom_type == "sphere") {
        RealArray center;
        ppls.get("sphere_center", center);

        Real radius;
        ppls.get("sphere_radius", radius);

        bool has_fluid_inside;
        ppls.get("sphere_has_fluid_inside", has_fluid_inside);

        SDF::SphereSDF sf(center, radius, has_fluid_inside);

        fill_from_IF(geom, LS, sf);

        fill_ls_normals(geom, LS);
    }
    else if (geom_type == "two_spheres") {
        RealArray center1, center2;
        ppls.get("sphere1_center", center1);
        ppls.get("sphere2_center", center2);

        Real radius1, radius2;
        ppls.get("sphere1_radius", radius1);
        ppls.get("sphere2_radius", radius2);

        bool has_fluid_inside1, has_fluid_inside2;
        ppls.get("sphere1_has_fluid_inside", has_fluid_inside1);
        ppls.get("sphere2_has_fluid_inside", has_fluid_inside2);

        SDF::SphereSDF sf1(center1, radius1, has_fluid_inside1);
        SDF::SphereSDF sf2(center2, radius2, has_fluid_inside2);
        SDF::IntersectionSDF<SDF::SphereSDF, SDF::SphereSDF> intf(sf1, sf2);

        fill_from_IF(geom, LS, intf);

        fill_ls_normals(geom, LS);
    }
    else
    {
        Abort("Unsupported geom_type " + geom_type);
    }

    AMREX_ASSERT(!LS.contains_nan(0, 1, LS.nGrow()));
    AMREX_ASSERT(!LS.contains_nan(1, AMREX_SPACEDIM, 0));
}