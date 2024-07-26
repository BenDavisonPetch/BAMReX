#include "AmrLevelAdv.H"

#include "Geometry/BoxSDF.H"
#include "Geometry/ConeSDF.H"
#include "Geometry/CylinderSDF.H"
#include "Geometry/IntersectionSDF.H"
#include "Geometry/Normals.H"
#include "Geometry/PlaneSDF.H"
#include "Geometry/SphereSDF.H"
#include "Geometry/UnionSDF.H"

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
        LS.setVal(-1);
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
    else if (geom_type == "sphere")
    {
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
    else if (geom_type == "two_spheres")
    {
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
    else if (geom_type == "cylinder")
    {
        Real radius;
        ppls.get("cylinder_radius", radius);

        Array<Real, 3> vcenter, vaxis;
        ppls.get("cylinder_center", vcenter);
        ppls.get("cylinder_axis", vaxis);
        XDim3 center({ vcenter[0], vcenter[1], vcenter[2] });
        XDim3 axis({ vaxis[0], vaxis[1], vaxis[2] });

        bool has_fluid_inside;
        ppls.get("cylinder_has_fluid_inside", has_fluid_inside);

        SDF::CylinderSDF cf(center, axis, radius, has_fluid_inside);

        fill_from_IF(geom, LS, cf);
        fill_ls_normals(geom, LS);
    }
    else if (geom_type == "cone")
    {
        Real half_angle;
        ppls.get("cone_half_angle", half_angle);

        Array<Real, 3> vtip, vaxis;
        ppls.get("cone_tip", vtip);
        ppls.get("cone_axis", vaxis);
        XDim3 tip({ vtip[0], vtip[1], vtip[2] });
        XDim3 axis({ vaxis[0], vaxis[1], vaxis[2] });

        bool has_fluid_inside;
        ppls.get("cone_has_fluid_inside", has_fluid_inside);

        SDF::ConeSDF conef(tip, axis, half_angle, has_fluid_inside);

        fill_from_IF(geom, LS, conef);
        fill_ls_normals(geom, LS);
    }
    else if (geom_type == "plane")
    {
        Array<Real, 3> vpoint, vnormal;
        ppls.get("plane_point", vpoint);
        ppls.get("plane_normal", vnormal);
        XDim3 point({ vpoint[0], vpoint[1], vpoint[2] });
        XDim3 normal({ vnormal[0], vnormal[1], vnormal[2] });

        SDF::PlaneSDF pf(point, normal);

        fill_from_IF(geom, LS, pf);
        fill_ls_normals(geom, LS);
    }
    else if (geom_type == "guitton_nozzle")
    {
        Real scale_factor, wall_thickness, inner_radius, outer_radius, length,
            tip_angle;
        ppls.get("scale_factor", scale_factor);
        ppls.get("wall_thickness", wall_thickness);
        ppls.get("inner_radius", inner_radius);
        ppls.get("outer_radius", outer_radius); // distance from centerline to
                                                // CENTER of outermost wall
        ppls.get("length", length);
        ppls.get("tip_angle", tip_angle);
        // nozzle center isn't scaled
        Array<Real, 3> vcenter;
        ppls.get("nozzle_center", vcenter);
        XDim3 center({ vcenter[0], vcenter[1], vcenter[2] });
        XDim3 axis({ 1, 0, 0 });

        wall_thickness *= scale_factor;
        inner_radius *= scale_factor;
        outer_radius *= scale_factor;
        length *= scale_factor;

        SDF::CylinderSDF outercyl_out(
            center, axis, outer_radius + wall_thickness / 2, false);
        SDF::CylinderSDF outercyl_in(center, axis,
                                     outer_radius - wall_thickness / 2, true);
        auto outer_cyl_full = make_intersection(std::move(outercyl_out),
                                                std::move(outercyl_in));

        XDim3        outercylcone_tip({ center.x + length
                                            + (outer_radius - wall_thickness / 2)
                                                  / tan(tip_angle * M_PI / 180),
                                        center.y, center.z });
        SDF::ConeSDF outercyl_tip_cone(outercylcone_tip, axis, tip_angle,
                                       false);
        auto         outer_cyl = make_intersection(std::move(outer_cyl_full),
                                                   std::move(outercyl_tip_cone));

        SDF::CylinderSDF innercyl_out(
            center, axis, inner_radius + wall_thickness / 2, false);
        SDF::CylinderSDF innercyl_in(center, axis,
                                     inner_radius - wall_thickness / 2, true);
        auto inner_cyl_full = make_intersection(std::move(innercyl_out),
                                                std::move(innercyl_in));

        XDim3 innercyl_otip_cone_tip(
            { center.x + length
                  + inner_radius / tan(tip_angle * 0.5 * M_PI / 180),
              center.y, center.z });
        XDim3 innercyl_itip_cone_tip(
            { center.x + length
                  - inner_radius / tan(tip_angle * 0.5 * M_PI / 180),
              center.y, center.z });
        XDim3        invaxis({ -1, 0, 0 });
        SDF::ConeSDF inner_otip_cone(innercyl_otip_cone_tip, axis,
                                     tip_angle / 2, false);
        SDF::ConeSDF inner_itip_cone(innercyl_itip_cone_tip, invaxis,
                                     tip_angle / 2, true);
        auto         inner_tips = make_intersection(std::move(inner_otip_cone),
                                                    std::move(inner_itip_cone));

        auto inner_cyl = make_intersection(std::move(inner_tips),
                                           std::move(inner_cyl_full));

        auto nozzle = make_union(std::move(outer_cyl), std::move(inner_cyl));
        // auto nozzle = make_union(std::move(outer_cyl_full),
        // std::move(inner_cyl_full));

        // to account for the SDF values in the fluid being wrong beyond the
        // non-right angled corners, let's take another intersection with a
        // plane at the outflow
        SDF::PlaneSDF plane({ center.x + length, center.y, center.z },
                            { 1, 0, 0 });
        auto          nozzle_chopped
            = make_intersection(std::move(nozzle), std::move(plane));

        fill_from_IF(geom, LS, nozzle_chopped);
        fill_ls_normals(geom, LS);
    }
    else
    {
        Abort("Unsupported geom_type " + geom_type);
    }

    AMREX_ASSERT(!LS.contains_nan(0, 1, LS.nGrow()));
    AMREX_ASSERT(!LS.contains_nan(1, AMREX_SPACEDIM, 0));
}