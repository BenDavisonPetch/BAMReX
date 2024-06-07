#ifndef ButcherTableau_H_
#define ButcherTableau_H_

#include <AMReX_Vector.H>

#include <AMReX_REAL.H>

/**
 * Represents a dual Butcher Tableau for an IMEX-RK scheme.
 */
class IMEXButcherTableau
{
  public:
    enum TableauType
    {
        SP111,
        SSP222,
        SASSP322,
        SASSP332,
        SSP433
    };

    static TableauType enum_from_string(const std::string &name);

    IMEXButcherTableau(IMEXButcherTableau::TableauType type);

    int n_steps_imp() const { return m_n_steps_imp; }
    int n_steps_exp() const { return m_n_steps_exp; }
    int order() const { return m_order; }
    const amrex::Vector<amrex::Vector<amrex::Real> > &get_A_imp() const
    {
        return A_imp;
    }
    const amrex::Vector<amrex::Vector<amrex::Real> > &get_A_exp() const
    {
        return A_exp;
    }
    const amrex::Vector<amrex::Real> &get_b_imp() const { return b_imp; }
    const amrex::Vector<amrex::Real> &get_b_exp() const { return b_exp; }
    AMREX_FORCE_INLINE
    amrex::Vector<amrex::Real> get_A_imp_row(int row, amrex::Real mult) const;
    AMREX_FORCE_INLINE
    amrex::Vector<amrex::Real> get_A_exp_row(int row, amrex::Real mult) const;

  protected:
    int                                        m_n_steps_imp;
    int                                        m_n_steps_exp;
    int                                        m_order;
    amrex::Vector<amrex::Vector<amrex::Real> > A_imp, A_exp;
    amrex::Vector<amrex::Real>                 b_imp, b_exp;
};

AMREX_FORCE_INLINE amrex::Vector<amrex::Real>
IMEXButcherTableau::get_A_imp_row(int row, amrex::Real mult) const
{
    amrex::Vector<amrex::Real> r(m_n_steps_imp);
    for (int i = 0; i < m_n_steps_imp; ++i) r[i] = A_imp[row][i] * mult;
    return r;
}

AMREX_FORCE_INLINE amrex::Vector<amrex::Real>
IMEXButcherTableau::get_A_exp_row(int row, amrex::Real mult) const
{
    amrex::Vector<amrex::Real> r(m_n_steps_exp);
    for (int i = 0; i < m_n_steps_exp; ++i) r[i] = A_exp[row][i] * mult;
    return r;
}

#endif