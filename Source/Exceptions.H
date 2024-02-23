#ifndef Exceptions_H_
#define Exceptions_H_

#define ASSERT_NOT_NAN(x)                                                     \
    AMREX_ASSERT_WITH_MESSAGE(x == x, "Quantity " #x " is NaN.");

#endif