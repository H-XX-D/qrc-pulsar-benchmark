#ifndef LUNAR_HARMONICS_H
#define LUNAR_HARMONICS_H
// Native CPU reference accelerator; no FPGA synthesis/performance claim.
// Fully normalized (4pi, no Condon-Shortley phase) coefficients, packed by
// index n*(n+1)/2+m. Include C00=1. Position and output use the same body axes.
// SI; degree 0..360; arrays require (degree+1)*(degree+2)/2 doubles each.
// Caller-owned buffers, no retained pointers, no heap or global mutable state.
// Returns 0 success, 1 invalid input. Outputs are unchanged on failure.
extern "C" int lunar_harmonics_acceleration(
    const double position[3], int degree, double gm, double radius,
    const double cosine[], const double sine[], double acceleration[3], double *potential);
#endif
