#include "ButcherTableau.H"

using namespace amrex;

IMEXButcherTableau::IMEXButcherTableau(IMEXButcherTableau::TableauType type)
{
    switch (type)
    {
    case SP111:
    {
        m_n_steps_imp = 1;
        m_n_steps_exp = 1;
        m_order       = 1;
        A_imp         = { { 1 } };
        A_exp         = { { 0 } };
        b_imp         = { 1 };
        b_exp         = { 1 };
        break;
    }
    case SSP222:
    {
        m_n_steps_imp    = 2;
        m_n_steps_exp    = 2;
        m_order          = 2;
        const Real alpha = 1 - 1 / sqrt(2);
        A_imp            = {
                       {        alpha,     0},
                       {1 - 2 * alpha, alpha}
        };
        A_exp = {
            {0, 0},
            {1, 0}
        };
        b_imp = { 0.5, 0.5 };
        b_exp = { 0.5, 0.5 };
        break;
    }
    case SASSP322:
    {
        m_n_steps_imp = 3;
        m_n_steps_exp = 2;
        m_order       = 2;
        A_imp         = {
                    { 0.5,   0,   0},
                    {-0.5, 0.5,   0},
                    {   0, 0.5, 0.5}
        };
        A_exp = {
            {0, 0, 0},
            {0, 0, 0},
            {0, 1, 0}
        };
        b_imp = { 0, 0.5, 0.5 };
        b_exp = { 0, 0.5, 0.5 };
        break;
    }
    case SASSP332:
    {
        m_n_steps_imp = 3;
        m_n_steps_exp = 3;
        m_order       = 2;
        A_imp         = {
                    {       0.25,           0,           0},
                    {          0,        0.25,           0},
                    {(Real)1 / 3, (Real)1 / 3, (Real)1 / 3}
        };
        A_exp = {
            {  0,   0, 0},
            {0.5,   0, 0},
            {0.5, 0.5, 0}
        };
        b_imp = { (Real)1 / 3, (Real)1 / 3, (Real)1 / 3 };
        b_exp = { (Real)1 / 3, (Real)1 / 3, (Real)1 / 3 };
        break;
    }
    case SSP433:
    {
        m_n_steps_imp    = 4;
        m_n_steps_exp    = 3;
        m_order          = 3;
        const Real alpha = 0.24169426078821;
        const Real beta  = 0.06042356519705;
        const Real eta   = 0.12915286960590;
        A_imp            = {
                       { alpha,         0,                        0,     0},
                       {-alpha,     alpha,                        0,     0},
                       {     0, 1 - alpha,                    alpha,     0},
                       {  beta,       eta, 0.5 - beta - eta - alpha, alpha}
        };
        A_exp = {
            {0,    0,    0, 0},
            {0,    0,    0, 0},
            {0,    1,    0, 0},
            {0, 0.25, 0.25, 0}
        };
        b_imp = { 0, (Real)1 / 6, (Real)1 / 6, (Real)2 / 3 };
        b_exp = { 0, (Real)1 / 6, (Real)1 / 6, (Real)2 / 3 };
        break;
    }
    }
}

IMEXButcherTableau::TableauType
IMEXButcherTableau::enum_from_string(const std::string &name)
{
    if (name == "SP111")
        return SP111;
    else if (name == "SSP222")
        return SSP222;
    else if (name == "SASSP322")
        return SASSP322;
    else if (name == "SASSP332")
        return SASSP332;
    else if (name == "SSP433")
        return SSP433;
    else
    {
        amrex::Abort("Invalid butcher tableau \"" + name + "\"");
        return SP111;
    }
}