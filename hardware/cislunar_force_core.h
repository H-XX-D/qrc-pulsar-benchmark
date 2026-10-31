#ifndef CISLUNAR_FORCE_CORE_H
#define CISLUNAR_FORCE_CORE_H

// Host-tested C ABI / HLS candidate, not a synthesized or flight-qualified core.
// SI, Moon-centred nonrotating ICRF. Caller supplies epoch-matched environment:
// env[0:3] Earth relative Moon, [3:6] Sun relative Moon, [6:9] lunar pole.
// params: GM Moon, GM Earth, GM Sun, Moon radius, J2, SRP acceleration at 1 AU,
// Earth radius. SRP coefficient = nominal solar pressure * Cr * area/mass.
// Binary conical umbra; partial eclipses and high-degree gravity not modeled.
// Return 0 success, 1 invalid input/collision. Outputs unchanged on failure.
extern "C" {
int cislunar_acceleration(const double pos[3], const double env[9],
                         const double params[7], double acceleration[3]);
// Second-order velocity Verlet with start/end ephemerides. This is NOT the
// autonomous fourth-order Yoshida scheme in jacobi_symplectic_core.cpp.
int cislunar_verlet_step(double pos[3], double vel[3], double dt_s,
                        const double env_start[9], const double env_end[9],
                        const double params[7]);
}
#endif
