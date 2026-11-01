#ifndef QRC_ORBIT_MPC_CORE_H
#define QRC_ORBIT_MPC_CORE_H
#include <stdint.h>
#define QRC_MPC_N 36
#ifndef QRC_MPC_BITS
#define QRC_MPC_BITS 32
#endif
#ifndef QRC_MPC_FRAC
#define QRC_MPC_FRAC 24
#endif
#ifndef QRC_MPC_ITERATIONS
#define QRC_MPC_ITERATIONS 256
#endif
// Bounded accelerated projected-gradient QP: min .5*x'H*x + g'x,
// -1 <= x <= 1. Host supplies symmetric positive-definite H scaled so
// max row absolute sum <= 1, and beta in [0,1). No PID fallback.
// Arrays are fixed size, row-major. Unused rows/columns are zero padded.
// Status: 1 invalid input, 2 stale navigation, 4 arithmetic range,
// 16 fixed iteration budget did not meet projected residual tolerance.
// Only status 0 publishes a solution. All failures zero the output.
// Hardware timing/resources require actual synthesis and RTL verification.
// Host bridge identifies ABI, arithmetic format and bounded work budget.
extern "C" int orbit_mpc_metadata(int field);
extern "C" int orbit_mpc_solve(const int32_t h[QRC_MPC_N*QRC_MPC_N],
    const int32_t g[QRC_MPC_N], const int32_t warm[QRC_MPC_N],
    int n, int32_t beta, int32_t tolerance, int valid,
    int32_t solution[QRC_MPC_N], int32_t *residual);
#endif
