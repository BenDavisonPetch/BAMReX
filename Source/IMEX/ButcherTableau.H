#ifndef ButcherTableau_H_
#define ButcherTableau_H_

#include <vector>

#include <AMReX_REAL.H>

/**
 * Represents a dual Butcher Tableau for an IMEX-RK scheme.
 */
class IMEXButcherTableau {
    public:

    enum TableauType {
        SA111
    };

    IMEXButcherTableau(IMEXButcherTableau::TableauType type);

    unsigned int n_steps_imp() const;
    unsigned int n_steps_exp() const;
    unsigned int order() const;
    const std::vector<amrex::Real>& get_A_imp() const;
    const std::vector<amrex::Real>& get_A_exp() const;
    const std::vector<amrex::Real>& get_b_imp() const;
    const std::vector<amrex::Real>& get_b_exp() const;
};

#endif