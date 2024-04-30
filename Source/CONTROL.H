#ifndef CONTROL_H_
#define CONTROL_H_

// 0 : MUSCL-Hancock
// 1 : P-IMEX O1
// 2 : HLLC
// 3 : IMEX Boscheri and Pareshi
#define SCHEME 3

// used in IMEX Boscheri and Pareshi
#define REUSE_PRESSURE true
#define USE_EXPLICIT_ENTH false
#define USE_ALT_LAPLACIAN false
#if REUSE_PRESSURE && USE_EXPLICIT_ENTH
#error "Cannot have both!"
#endif

#endif