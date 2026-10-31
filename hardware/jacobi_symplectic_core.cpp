#include "jacobi_symplectic_core.h"

/**
 * Pipelined hardware acceleration sub-step.
 * Evaluates pairwise inverse-cube forces across the 3 bodies.
 */
static void compute_accelerations_hls(
    const double p[3][3],
    const double m[3],
    double a[3][3]
) {
    #pragma HLS INLINE
    #pragma HLS ARRAY_PARTITION variable=p complete dim=0
    #pragma HLS ARRAY_PARTITION variable=m complete dim=1
    #pragma HLS ARRAY_PARTITION variable=a complete dim=0

    for (int i = 0; i < 3; i++) {
        #pragma HLS UNROLL
        a[i][0] = 0.0;
        a[i][1] = 0.0;
        a[i][2] = 0.0;
    }

    // Unroll all 3 pairwise interactions in parallel hardware
    // Pair (0, 1)
    double dx01 = p[1][0] - p[0][0];
    double dy01 = p[1][1] - p[0][1];
    double dz01 = p[1][2] - p[0][2];
    double rsq01 = dx01*dx01 + dy01*dy01 + dz01*dz01;
    double inv_r01 = 1.0 / std::sqrt(rsq01);
    double inv_r3_01 = inv_r01 * inv_r01 * inv_r01;
    double f01 = G_GRAV * inv_r3_01;

    // Pair (0, 2)
    double dx02 = p[2][0] - p[0][0];
    double dy02 = p[2][1] - p[0][1];
    double dz02 = p[2][2] - p[0][2];
    double rsq02 = dx02*dx02 + dy02*dy02 + dz02*dz02;
    double inv_r02 = 1.0 / std::sqrt(rsq02);
    double inv_r3_02 = inv_r02 * inv_r02 * inv_r02;
    double f02 = G_GRAV * inv_r3_02;

    // Pair (1, 2)
    double dx12 = p[2][0] - p[1][0];
    double dy12 = p[2][1] - p[1][1];
    double dz12 = p[2][2] - p[1][2];
    double rsq12 = dx12*dx12 + dy12*dy12 + dz12*dz12;
    double inv_r12 = 1.0 / std::sqrt(rsq12);
    double inv_r3_12 = inv_r12 * inv_r12 * inv_r12;
    double f12 = G_GRAV * inv_r3_12;

    // Accumulate accelerations on body 0
    a[0][0] = f01 * m[1] * dx01 + f02 * m[2] * dx02;
    a[0][1] = f01 * m[1] * dy01 + f02 * m[2] * dy02;
    a[0][2] = f01 * m[1] * dz01 + f02 * m[2] * dz02;

    // Accumulate accelerations on body 1
    a[1][0] = -f01 * m[0] * dx01 + f12 * m[2] * dx12;
    a[1][1] = -f01 * m[0] * dy01 + f12 * m[2] * dy12;
    a[1][2] = -f01 * m[0] * dz01 + f12 * m[2] * dz12;

    // Accumulate accelerations on body 2
    a[2][0] = -f02 * m[0] * dx02 - f12 * m[1] * dx12;
    a[2][1] = -f02 * m[0] * dy02 - f12 * m[1] * dy12;
    a[2][2] = -f02 * m[0] * dz02 - f12 * m[1] * dz12;
}

/**
 * Top-level HLS synthesizable function.
 * Pipelined with Initiation Interval II = 1.
 */
extern "C" {
void jacobi_symplectic_step(
    double pos[3][3],
    double vel[3][3],
    const double masses[3],
    const double dt_s,
    const double los_vec[3],
    double *roemer_s
) {
    #pragma HLS INTERFACE ap_ctrl_hs port=return
    #pragma HLS INTERFACE m_axi port=pos offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=vel offset=slave bundle=gmem
    #pragma HLS INTERFACE s_axilite port=masses bundle=control
    #pragma HLS INTERFACE s_axilite port=dt_s bundle=control
    #pragma HLS INTERFACE s_axilite port=los_vec bundle=control
    #pragma HLS INTERFACE s_axilite port=roemer_s bundle=control
    #pragma HLS PIPELINE II=1

    double acc[3][3];
    #pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    // Stage 1: Advance positions by c1 * dt
    for (int i = 0; i < 3; i++) {
        #pragma HLS UNROLL
        pos[i][0] += C1 * dt_s * vel[i][0];
        pos[i][1] += C1 * dt_s * vel[i][1];
        pos[i][2] += C1 * dt_s * vel[i][2];
    }
    compute_accelerations_hls(pos, masses, acc);
    for (int i = 0; i < 3; i++) {
        #pragma HLS UNROLL
        vel[i][0] += D1 * dt_s * acc[i][0];
        vel[i][1] += D1 * dt_s * acc[i][1];
        vel[i][2] += D1 * dt_s * acc[i][2];
    }

    // Stage 2: Advance positions by c2 * dt
    for (int i = 0; i < 3; i++) {
        #pragma HLS UNROLL
        pos[i][0] += C2 * dt_s * vel[i][0];
        pos[i][1] += C2 * dt_s * vel[i][1];
        pos[i][2] += C2 * dt_s * vel[i][2];
    }
    compute_accelerations_hls(pos, masses, acc);
    for (int i = 0; i < 3; i++) {
        #pragma HLS UNROLL
        vel[i][0] += D2 * dt_s * acc[i][0];
        vel[i][1] += D2 * dt_s * acc[i][1];
        vel[i][2] += D2 * dt_s * acc[i][2];
    }

    // Stage 3: Advance positions by c3 * dt
    for (int i = 0; i < 3; i++) {
        #pragma HLS UNROLL
        pos[i][0] += C3 * dt_s * vel[i][0];
        pos[i][1] += C3 * dt_s * vel[i][1];
        pos[i][2] += C3 * dt_s * vel[i][2];
    }
    compute_accelerations_hls(pos, masses, acc);
    for (int i = 0; i < 3; i++) {
        #pragma HLS UNROLL
        vel[i][0] += D3 * dt_s * acc[i][0];
        vel[i][1] += D3 * dt_s * acc[i][1];
        vel[i][2] += D3 * dt_s * acc[i][2];
    }

    // Stage 4: Final position drift by c4 * dt
    for (int i = 0; i < 3; i++) {
        #pragma HLS UNROLL
        pos[i][0] += C4 * dt_s * vel[i][0];
        pos[i][1] += C4 * dt_s * vel[i][1];
        pos[i][2] += C4 * dt_s * vel[i][2];
    }

    // Compute Roemer delay on body 0: - (1/c) * (los . pos[0])
    double dot_los = los_vec[0] * pos[0][0] + los_vec[1] * pos[0][1] + los_vec[2] * pos[0][2];
    *roemer_s = - (dot_los / C_LIGHT);
}
}
