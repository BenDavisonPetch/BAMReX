#include <gtest/gtest.h>

#include "System/PlasmaSystem.H"

using namespace amrex;

TEST(PlasmaSystem, LimitingParameters)
{
    ParmParse pp("prob");
    pp.add("system", (std::string) "plasma");
    pp.add("adiabatic", 1.6);
    ParmParse ppl("limiting");
    ppl.add("limiter", (std::string) "van_leer");
    ppl.add("density", (std::string) "density");
    ppl.addarr("mom", { (std::string) "mom_x", (std::string) "mom_y",
                        (std::string) "mom_z" });
    ppl.add("energy", (std::string) "energy");
    ppl.addarr("B", { (std::string) "B_x", (std::string) "B_y",
                      (std::string) "B_z" });

    System *system = System::NewSystem(pp);

    Parm *h_parm = new Parm{};
    system->ReadParameters(*h_parm, pp, ppl);

    ASSERT_EQ(h_parm->limiter, Limiter::van_leer);
    ASSERT_FALSE(h_parm->need_primitive_for_limiting);

    ASSERT_EQ(h_parm->limiter_indices[PUInd::RHO], PUInd::RHO);
    ASSERT_EQ(h_parm->limiter_indices[PUInd::MX], PUInd::MX);
    ASSERT_EQ(h_parm->limiter_indices[PUInd::MY], PUInd::MY);
    ASSERT_EQ(h_parm->limiter_indices[PUInd::MZ], PUInd::MZ);
    ASSERT_EQ(h_parm->limiter_indices[PUInd::E], PUInd::E);
    ASSERT_EQ(h_parm->limiter_indices[PUInd::BX], PUInd::BX);
    ASSERT_EQ(h_parm->limiter_indices[PUInd::BY], PUInd::BY);
    ASSERT_EQ(h_parm->limiter_indices[PUInd::BZ], PUInd::BZ);

    ASSERT_FALSE(h_parm->limit_on_primv[PUInd::RHO]);
    ASSERT_FALSE(h_parm->limit_on_primv[PUInd::MX]);
    ASSERT_FALSE(h_parm->limit_on_primv[PUInd::MY]);
    ASSERT_FALSE(h_parm->limit_on_primv[PUInd::MZ]);
    ASSERT_FALSE(h_parm->limit_on_primv[PUInd::E]);
    ASSERT_FALSE(h_parm->limit_on_primv[PUInd::BX]);
    ASSERT_FALSE(h_parm->limit_on_primv[PUInd::BY]);
    ASSERT_FALSE(h_parm->limit_on_primv[PUInd::BZ]);

    delete h_parm;
    delete system;
    h_parm = nullptr;
    system = nullptr;
}