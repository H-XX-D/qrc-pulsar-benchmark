#ifndef QRC_ORBIT_PID_CORE_H
#define QRC_ORBIT_PID_CORE_H
#include <stdint.h>
// Portable integer-only HLS candidate. No device timing/resources are asserted.
// Compile all participants with identical width/fraction settings.
#ifndef QRC_PID_BITS
#define QRC_PID_BITS 32
#endif
#ifndef QRC_PID_FRAC
#define QRC_PID_FRAC 24
#endif
// Inputs are normalized tracking errors, not large absolute orbit coordinates.
// config: Kp_step,Kd_step,Ki_step,derivative_alpha,axis_dv_limit,integral_limit.
// state: integral contribution[3], filtered velocity error[3].
// Error sign is actual-reference; output is ff-Kp*e-Kd*vf-integral.
// Round to nearest, ties away from zero at EACH product assignment; saturate.
// Status: 1 invalid configuration/state, 2 unavailable fix, 4 arithmetic range
// fault (all three inhibit output and preserve state), 8 ordinary output limit.
// No allocation, iterative solve, transcendental function, or variable trip loop.
extern "C" int orbit_pid_step(const int32_t error[3], const int32_t velocity_error[3],
    const int32_t feedforward[3], const int32_t config[6], int valid,
    int32_t state[6], int32_t command[3]);
#endif
