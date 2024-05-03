#include "ButcherTableau.H"

IMEXButcherTableau::IMEXButcherTableau(IMEXButcherTableau::TableauType type)
{
    switch (type)
    {
    case SA111:
        m_n_steps_imp = 1;
        m_n_steps_exp = 1;
        m_order       = 1;
        A_imp         = { { 1 } };
        A_exp         = { { 0 } };
        b_imp         = { 1 };
        b_exp         = { 1 };
    }
}