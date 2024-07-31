#include "BCs.H"
#include "AmrLevelAdv.H"
#include "Euler/Euler.H"
#include "Euler/NComp.H"
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
{
}

void bamrexBCData::build(const amrex::Geometry *top_geom)
{
    ParmParse pp("bc");
    if (pp.contains("x"))
    {
        std::array<std::string, 2> bc_in;
        pp.get("x", bc_in);
        bcs[0] = bamrexBCData::enum_from_string(bc_in[0]);
        bcs[1] = bamrexBCData::enum_from_string(bc_in[1]);
    }
    else
    {
        bcs[0] = bamrexBCData::transmissive;
        bcs[1] = bamrexBCData::transmissive;
    }
#if AMREX_SPACEDIM >= 2
    if (pp.contains("y"))
    {
        std::array<std::string, 2> bc_in;
        pp.get("y", bc_in);
        bcs[2] = bamrexBCData::enum_from_string(bc_in[0]);
        bcs[3] = bamrexBCData::enum_from_string(bc_in[1]);
    }
    else
    {
        bcs[2] = bamrexBCData::transmissive;
        bcs[3] = bamrexBCData::transmissive;
    }
#endif
#if AMREX_SPACEDIM == 3
    if (pp.contains("z"))
    {
        std::array<std::string, 2> bc_in;
        pp.get("z", bc_in);
        bcs[4] = bamrexBCData::enum_from_string(bc_in[0]);
        bcs[5] = bamrexBCData::enum_from_string(bc_in[1]);
    }
    else
    {
        bcs[4] = bamrexBCData::transmissive;
        bcs[5] = bamrexBCData::transmissive;
    }
#endif

    // Check if we need to use a non-null boundary fill function
    m_jet    = false;
    m_gresho = false;
    for (int id = 0; id < 2 * AMREX_SPACEDIM; ++id)
    {
        m_jet    = m_jet || bcs[id] == bamrexBCData::coaxial_jet;
        m_gresho = m_gresho || bcs[id] == bamrexBCData::gresho;
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
        Real exit_v_ratio, primary_exit_M, ambient_p, ambient_T,
            air_specific_gas_const, inner_radius, outer_radius, scale_factor;
        ParmParse ppinit("init");
        ppinit.get("exit_v_ratio", exit_v_ratio);
        ppinit.get("primary_exit_M", primary_exit_M);
        ppinit.get("ambient_p", ambient_p);
        ppinit.get("ambient_T", ambient_T);
        ppinit.get("air_specific_gas_const", air_specific_gas_const);

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

        const Real dens = ambient_p / (air_specific_gas_const * ambient_T);
        const Real c    = sqrt(ambient_p * adia / dens);
        const Real v_p  = primary_exit_M * c;
        const Real v_s  = exit_v_ratio * v_p;

        // wall = 0 => on x lo face
        bndryfunc_consv
            = make_coax_jetfill_bcfunc(center, inner_radius, outer_radius, 0,
                                       dens, v_p, v_s, ambient_p, adia, eps);
        bndryfunc_consv_sd = StateDescriptor::BndryFunc(bndryfunc_consv);
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
        bamrexBCData::BCType lo_type = bcs[2 * dir];
        bamrexBCData::BCType hi_type = bcs[2 * dir + 1];

        if (top_geom->isPeriodic(dir))
        {
            lo_type = bamrexBCData::periodic;
            hi_type = bamrexBCData::periodic;
        }

        switch (lo_type)
        {
        case bamrexBCData::transmissive:
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
        case bamrexBCData::periodic:
        {
            for (int n = 0; n < CONSV_NCOMP; ++n)
            {
                consv_bcrecs[n].setLo(dir, amrex::BCType::int_dir);
            }
            pressure_bcrec.setLo(dir, amrex::BCType::int_dir);

            p_linop_lobc[dir] = LinOpBCType::Periodic;
            break;
        }
        case bamrexBCData::reflective:
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
        case bamrexBCData::gresho:
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
        case bamrexBCData::coaxial_jet:
        {
            for (int n = 0; n < CONSV_NCOMP; ++n)
            {
                consv_bcrecs[n].setLo(dir, amrex::BCType::user_1);
            }
            // not implemented for IMEX yet
            pressure_bcrec.setLo(dir, amrex::BCType::foextrap);

            p_linop_lobc[dir] = LinOpBCType::Robin;
            break;
        }
        default:
        {
            amrex::Abort("Boundary type not implemented!");
        }
        }
        switch (hi_type)
        {
        case bamrexBCData::transmissive:
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
        case bamrexBCData::periodic:
        {
            for (int n = 0; n < CONSV_NCOMP; ++n)
            {
                consv_bcrecs[n].setHi(dir, amrex::BCType::int_dir);
            }
            pressure_bcrec.setHi(dir, amrex::BCType::int_dir);

            p_linop_hibc[dir] = LinOpBCType::Periodic;
            break;
        }
        case bamrexBCData::reflective:
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
        case bamrexBCData::gresho:
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
        case bamrexBCData::coaxial_jet:
        {
            for (int n = 0; n < CONSV_NCOMP; ++n)
            {
                consv_bcrecs[n].setHi(dir, amrex::BCType::user_1);
            }
            // not implemented for IMEX yet
            pressure_bcrec.setHi(dir, amrex::BCType::foextrap);

            p_linop_hibc[dir] = LinOpBCType::Robin;
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
                                          Real                   time) const
{
    // internal and periodic boundaries
    pressure.FillBoundary(geom.periodicity());
    // foextrap, reflect_even, reflect_odd, hoextrap, and external boundareis
    PhysBCFunct<BndryFuncFabDefault> physbcf(geom, { pressure_bcrec },
                                             bndryfunc_p);
    physbcf(pressure, 0, pressure.nComp(), pressure.nGrowVect(), time, 0);
}

void bamrexBCData::fill_consv_boundary(const amrex::Geometry &geom,
                                       amrex::MultiFab &U, Real time) const
{
    // internal and periodic boundaries
    U.FillBoundary(geom.periodicity());
    // foextrap, reflect_even, reflect_odd, hoextrap, and external boundareis
    PhysBCFunct<BndryFuncFabDefault> physbcf(geom, consv_bcrecs,
                                             bndryfunc_consv);
    physbcf(U, 0, U.nComp(), U.nGrowVect(), time, 0);
}