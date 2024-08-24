#include "BCs.H"
#include "AmrLevelAdv.H"
#include "Euler/Euler.H"
#include "Euler/NComp.H"
#include "MultiBCFunct.H"
#include "bc_extfill.H"
#include "bc_jet.H"
#include "bc_nullfill.H"

#include <AMReX_BC_TYPES.H>
#include <AMReX_LO_BCTYPES.H>

using namespace amrex;

bamrexBCData::bamrexBCData()
    : bndryfunc_consv(nullfill)
    , bndryfunc_p(nullfill)
    , bndryfunc_consv_sd(nullfill)
    , bndryfunc_p_sd(nullfill)
    , bndryfunc_ls_sd(nullfill)
    , consv_bcrecs(CONSV_NCOMP)
    , m_rb_enabled(false)
{
}

void bamrexBCData::build(const amrex::Geometry *top_geom)
{
    ParmParse pp("bc");
    if (pp.contains("x"))
    {
        std::array<std::string, 2> bc_in;
        pp.get("x", bc_in);
        bcs[0] = BAMReXBCType::enum_from_string(bc_in[0]);
        bcs[1] = BAMReXBCType::enum_from_string(bc_in[1]);
    }
    else
    {
        bcs[0] = BAMReXBCType::transmissive;
        bcs[1] = BAMReXBCType::transmissive;
    }
#if AMREX_SPACEDIM >= 2
    if (pp.contains("y"))
    {
        std::array<std::string, 2> bc_in;
        pp.get("y", bc_in);
        bcs[2] = BAMReXBCType::enum_from_string(bc_in[0]);
        bcs[3] = BAMReXBCType::enum_from_string(bc_in[1]);
    }
    else
    {
        bcs[2] = BAMReXBCType::transmissive;
        bcs[3] = BAMReXBCType::transmissive;
    }
#endif
#if AMREX_SPACEDIM == 3
    if (pp.contains("z"))
    {
        std::array<std::string, 2> bc_in;
        pp.get("z", bc_in);
        bcs[4] = BAMReXBCType::enum_from_string(bc_in[0]);
        bcs[5] = BAMReXBCType::enum_from_string(bc_in[1]);
    }
    else
    {
        bcs[4] = BAMReXBCType::transmissive;
        bcs[5] = BAMReXBCType::transmissive;
    }
#endif
    if (pp.contains("rb"))
    {
        std::string rb_bc_in;
        pp.get("rb", rb_bc_in);
        rb_bc = RigidBodyBCType::enum_from_string(rb_bc_in);
        if (rb_bc == RigidBodyBCType::no_slip)
            amrex::Abort(
                "No slip boundary conditions don't work without viscosity!");
    }
    else
    {
        rb_bc = RigidBodyBCType::reflective;
    }

    // Check if we need to use a non-null boundary fill function
    m_jet    = false;
    m_gresho = false;
    for (int id = 0; id < 2 * AMREX_SPACEDIM; ++id)
    {
        m_jet    = m_jet || bcs[id] == BAMReXBCType::coaxial_jet;
        m_gresho = m_gresho || bcs[id] == BAMReXBCType::gresho;
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        !(m_jet && m_gresho),
        "Cannot have runtime-specified jet boundaries as well as gresho "
        "vortex boundaries!");

    //
    // Construct boundary functions
    //
    // By default bndryfunc_consv and bndryfunc_p are nullfill
    if (m_jet)
    {
        // Construct jet BC function. Transmissive except for the jet
        Real exit_v_ratio, primary_exit_M, p_amb, T_amb, T_p, T_s, R_spec,
            inner_radius, outer_radius, scale_factor;
        ParmParse ppinit("init");
        ppinit.get("exit_v_ratio", exit_v_ratio);
        ppinit.get("primary_exit_M", primary_exit_M);
        ppinit.get("p_ambient", p_amb);
        ppinit.get("T_ambient", T_amb);
        ppinit.get("T_p", T_p);
        ppinit.get("T_s", T_s);

        ppinit.get("R_spec", R_spec);

        RealArray center;
        ParmParse ppls("ls");
        ppls.get("inner_radius", inner_radius);
        ppls.get("outer_radius", outer_radius);
        ppls.get("nozzle_center", center);
        ppls.get("scale_factor", scale_factor);
        inner_radius *= scale_factor;
        outer_radius *= scale_factor;
        // center doesn't get scaled

        Real      adia, eps = 1;
        ParmParse ppprob("prob");
        ppprob.get("adiabatic", adia);
        ppprob.query("epsilon", eps);

        const Real dens_amb = p_amb / (R_spec * T_amb);
        const Real c        = sqrt(p_amb * adia / dens_amb);
        const Real v_p      = primary_exit_M * c;
        const Real v_s      = exit_v_ratio * v_p;

        // wall = 0 => on x lo face
        bndryfunc_consv = make_coax_jetfill_bcfunc(
            center, inner_radius, outer_radius, 0, v_p, v_s, T_p, T_s, R_spec,
            p_amb, adia, eps);
        bndryfunc_consv_sd = StateDescriptor::BndryFunc(bndryfunc_consv);
        // Nullfill for pressure - all boundary filling must be done by
        // fill_pressure_boundary, using the below object since we have to use
        // the conservative variables
        bndryfunc_jet_p
            = make_coax_jetfill_bcfunc_p(center, inner_radius, outer_radius, 0,
                                         T_p, T_s, R_spec, p_amb, adia, eps);
    }
    else if (m_gresho)
    {
        Real M;
        {
            ParmParse pp("init");
            pp.get("M", M);
        }
        // By the time this is called h_prob_parm has not been populated. We
        // read adia and eps straight from the input file
        Real adia;
        Real eps = 1;
        {
            ParmParse pp("prob");
            pp.get("adiabatic", adia);
            pp.query("epsilon", eps);
        }
        AMREX_ALWAYS_ASSERT(eps == 1);
        AMREX_ASSERT(CONSV_NCOMP == EULER_NCOMP);

        const Real                        dens    = 1;
        const Real                        p_0     = dens / (adia * M * M);
        const Real                        p_infty = p_0 - 2 + 4 * log(2);
        const GpuArray<Real, CONSV_NCOMP> primv{ dens, AMREX_D_DECL(0, 0, 0),
                                                 p_infty };
        const GpuArray<Real, 1>           p_fill{ p_infty };
        const auto &consv  = consv_from_primv(primv, adia, eps);
        bndryfunc_consv    = make_extfill(consv);
        bndryfunc_p        = make_extfill(p_fill);
        bndryfunc_consv_sd = StateDescriptor::BndryFunc(bndryfunc_consv);
        bndryfunc_p_sd     = StateDescriptor::BndryFunc(bndryfunc_p);
    }
    // Make sure that the GPU is happy
    bndryfunc_consv_sd.setRunOnGPU(true);
    bndryfunc_p_sd.setRunOnGPU(
        true); // I promise the bc function will launch gpu kernels.

    //
    // Fill BCRecs
    //
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        BAMReXBCType::bctype lo_type = bcs[2 * dir];
        BAMReXBCType::bctype hi_type = bcs[2 * dir + 1];

        if (top_geom->isPeriodic(dir))
        {
            lo_type = BAMReXBCType::periodic;
            hi_type = BAMReXBCType::periodic;
        }

        switch (lo_type)
        {
        case BAMReXBCType::transmissive:
        {
            for (int n = 0; n < CONSV_NCOMP; ++n)
            {
                consv_bcrecs[n].setLo(dir, amrex::BCType::foextrap);
            }
            pressure_bcrec.setLo(dir, amrex::BCType::foextrap);

            // If any pressure variation reaches the boundary when using IMEX
            // this will no longer be transmissive!
            p_linop_lobc[dir] = LinOpBCType::Neumann;
            break;
        }
        case BAMReXBCType::periodic:
        {
            for (int n = 0; n < CONSV_NCOMP; ++n)
            {
                consv_bcrecs[n].setLo(dir, amrex::BCType::int_dir);
            }
            pressure_bcrec.setLo(dir, amrex::BCType::int_dir);

            p_linop_lobc[dir] = LinOpBCType::Periodic;
            break;
        }
        case BAMReXBCType::reflective:
        {
            AMREX_ASSERT(CONSV_NCOMP == EULER_NCOMP);
            for (int n = 0; n < CONSV_NCOMP; ++n)
            {
                consv_bcrecs[n].setLo(dir, amrex::BCType::reflect_even);
            }
            consv_bcrecs[1 + dir].setLo(dir, amrex::BCType::reflect_odd);

            pressure_bcrec.setLo(dir, amrex::BCType::reflect_even);

            p_linop_lobc[dir] = LinOpBCType::Neumann;
            break;
        }
        case BAMReXBCType::gresho:
        {
            AMREX_ASSERT(CONSV_NCOMP == EULER_NCOMP);
            for (int n = 0; n < CONSV_NCOMP; ++n)
            {
                consv_bcrecs[n].setLo(dir, amrex::BCType::ext_dir);
            }
            pressure_bcrec.setLo(dir, amrex::BCType::ext_dir);

            p_linop_lobc[dir] = LinOpBCType::Dirichlet;
            break;
        }
        case BAMReXBCType::coaxial_jet:
        {
            for (int n = 0; n < CONSV_NCOMP; ++n)
            {
                consv_bcrecs[n].setLo(dir, amrex::BCType::user_1);
            }
            // For IMEX, we fill the boundary conditions after the linear solve
            // using the new density
            pressure_bcrec.setLo(dir, amrex::BCType::user_1);

            // pressure outlet is trivially dirichlet
            // inlet with known u & T is dirichlet, *calculated from density
            // and T*
            p_linop_lobc[dir] = LinOpBCType::Dirichlet;
            break;
        }
        default:
        {
            amrex::Abort("Boundary type not implemented!");
        }
        }
        switch (hi_type)
        {
        case BAMReXBCType::transmissive:
        {
            for (int n = 0; n < CONSV_NCOMP; ++n)
            {
                consv_bcrecs[n].setHi(dir, amrex::BCType::foextrap);
            }
            pressure_bcrec.setHi(dir, amrex::BCType::foextrap);

            // If any pressure variation reaches the boundary when using IMEX
            // this will no longer be transmissive!
            p_linop_hibc[dir] = LinOpBCType::Neumann;
            break;
        }
        case BAMReXBCType::periodic:
        {
            for (int n = 0; n < CONSV_NCOMP; ++n)
            {
                consv_bcrecs[n].setHi(dir, amrex::BCType::int_dir);
            }
            pressure_bcrec.setHi(dir, amrex::BCType::int_dir);

            p_linop_hibc[dir] = LinOpBCType::Periodic;
            break;
        }
        case BAMReXBCType::reflective:
        {
            AMREX_ASSERT(CONSV_NCOMP == EULER_NCOMP);
            for (int n = 0; n < CONSV_NCOMP; ++n)
            {
                consv_bcrecs[n].setHi(dir, amrex::BCType::reflect_even);
            }
            consv_bcrecs[1 + dir].setHi(dir, amrex::BCType::reflect_odd);

            pressure_bcrec.setHi(dir, amrex::BCType::reflect_even);

            p_linop_hibc[dir] = LinOpBCType::Neumann;
            break;
        }
        case BAMReXBCType::gresho:
        {
            AMREX_ASSERT(CONSV_NCOMP == EULER_NCOMP);
            for (int n = 0; n < CONSV_NCOMP; ++n)
            {
                consv_bcrecs[n].setHi(dir, amrex::BCType::ext_dir);
            }
            pressure_bcrec.setHi(dir, amrex::BCType::ext_dir);

            p_linop_hibc[dir] = LinOpBCType::Dirichlet;
            break;
        }
        case BAMReXBCType::coaxial_jet:
        {
            for (int n = 0; n < CONSV_NCOMP; ++n)
            {
                consv_bcrecs[n].setHi(dir, amrex::BCType::user_1);
            }
            // For IMEX, we fill the boundary conditions after the linear solve
            // using the new density
            pressure_bcrec.setHi(dir, amrex::BCType::user_1);

            // pressure outlet is trivially dirichlet
            // inlet with known u & T is dirichlet, *calculated from density
            // and T*
            p_linop_hibc[dir] = LinOpBCType::Dirichlet;
            break;
        }
        default:
        {
            amrex::Abort("Boundary type not implemented!");
        }
        }

        // Leveset BCRecs
        ls_bcrec.setLo(dir, amrex::BCType::hoextrapcc);
        ls_bcrec.setHi(dir, amrex::BCType::hoextrapcc);
        norm_bcrec.setLo(dir, amrex::BCType::foextrap);
        norm_bcrec.setHi(dir, amrex::BCType::foextrap);
    }
}

void bamrexBCData::fill_pressure_boundary(const amrex::Geometry &geom,
                                          amrex::MultiFab       &pressure,
                                          const amrex::MultiFab &U,
                                          Real                   time) const
{
    // internal and periodic boundaries
    pressure.FillBoundary(geom.periodicity());
    if (!geom.isAllPeriodic())
    {
        if (m_jet)
        {
            // cursed re-implementation of AMReX's boundary filling routines so
            // that the pressure can be filled with knowledge of the density
            bamrex::MultiBCFunct<bamrex::MultiCCBndryFuncFabDefault> physbcf(
                geom, { pressure_bcrec }, bndryfunc_jet_p);
            physbcf(pressure, U, 0, 1, pressure.nGrowVect(), time, 0);
        }
        else
        {
            // foextrap, reflect_even, reflect_odd, hoextrap, and external
            // boundaries
            PhysBCFunct<BndryFuncFabDefault> physbcf(geom, { pressure_bcrec },
                                                     bndryfunc_p);
            physbcf(pressure, 0, pressure.nComp(), pressure.nGrowVect(), time,
                    0);
        }
    }
}

void bamrexBCData::fill_consv_boundary(const amrex::Geometry &geom,
                                       amrex::MultiFab &U, Real time) const
{
    // internal and periodic boundaries
    U.FillBoundary(geom.periodicity());
    if (!geom.isAllPeriodic())
    {
        // foextrap, reflect_even, reflect_odd, hoextrap, and external
        // boundareis
        PhysBCFunct<BndryFuncFabDefault> physbcf(geom, consv_bcrecs,
                                                 bndryfunc_consv);
        physbcf(U, 0, U.nComp(), U.nGrowVect(), time, 0);
    }
}