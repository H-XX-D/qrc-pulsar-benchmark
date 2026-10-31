#ifndef JACOBI_SYMPLECTIC_CORE_H
#define JACOBI_SYMPLECTIC_CORE_H

/**
 * =====================================================================================
 * QRC-PULSAR-BENCH: Three-body HLS candidate (hardware implementation unverified)
 * =====================================================================================
 * Target: AMD/Xilinx Kintex UltraScale XQRKU060 / Microchip RT PolarFire
 * Toolchain: Vivado HLS / Vitis HLS / Intel HLS Compiler
 * 
 * Computes:
 *   1. 4th-order Yoshida step for three barycentric Cartesian point masses.
 *   2. Instantaneous line-of-sight Roemer timing delay for autonomous pulsar navigation.
 * This API has no external forces or energy watchdog. For lunar J2, Earth/Sun
 * tides and configured SRP, see the separate cislunar_force_core.h API.
 * =====================================================================================
 */

#include <cmath>

// Physical constants in SI units
static const double G_GRAV = 6.67430e-11;
static const double C_LIGHT = 299792458.0;

// Yoshida 4th-order symmetric weights
static const double W1 = 1.3512071919596576340476878;
static const double W0 = -1.7024143839193152680953757;

static const double C1 = 0.5 * W1;
static const double C2 = 0.5 * (W0 + W1);
static const double C3 = 0.5 * (W0 + W1);
static const double C4 = 0.5 * W1;

static const double D1 = W1;
static const double D2 = W0;
static const double D3 = W1;

/**
 * Core synthesizable hardware pipeline function.
 * 
 * @param pos       [in,out] 3x3 array of barycentric positions (m)
 * @param vel       [in,out] 3x3 array of barycentric velocities (m/s)
 * @param masses    [in]     3-element mass vector (kg)
 * @param dt_s      [in]     Fixed symplectic timestep (s)
 * @param los_vec   [in]     Unit line-of-sight vector (dimensionless)
 * @param roemer_s  [out]    Instantaneous Roemer timing delay (s)
 */
extern "C" {
void jacobi_symplectic_step(
    double pos[3][3],
    double vel[3][3],
    const double masses[3],
    const double dt_s,
    const double los_vec[3],
    double *roemer_s
);
}

#endif // JACOBI_SYMPLECTIC_CORE_H
