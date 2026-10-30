"""
=====================================================================================
QRC-PULSAR-BENCH: Symmetric Time-Reversible Symplectic Integration Engine
=====================================================================================
Provides high-order symplectic integrators (Yoshida 4th-order, Yoshida 6th-order,
and Velocity Verlet) for long-baseline (giga-year / mega-orbit) dynamical studies,
secular stability analysis, and exact time-reversal experiments.

Key Theoretical Properties:
1. Symplectic Structure Preservation:
   Preserves the Poincaré phase-space 2-form ω = Σ dq ∧ dp exactly.
   Unlike Runge-Kutta / DOP853, there is zero secular energy drift (dE/dt = 0).
   Energy error fluctuates within a strictly bounded envelope O(dt^p) forever.

2. Strict Time-Reversibility (T-Symmetry):
   The operator compositions are strictly palindromic:
       S(-dt) ∘ S(dt) = I
   Flipping velocities v -> -v and integrating for the same duration rewinds the
   gravitational trajectory bit-for-bit back to its primordial initial state,
   limited only by symmetric floating-point roundoff (~ 10^-14 relative closure).

3. Galactic & Giga-Year Secular Evolution:
   Suitable for tracing Kozai-Lidov cycles, planetary formation history, and
   reversing time evolution of gravitational N-body systems.
=====================================================================================
"""

import time
from typing import Dict, Any, Tuple, List
import numpy as np

from .constants import G, c, M_sun, day_s, year_s
from .systems import HierarchicalTripleSystem, PSR_J0337_1715
from .dynamics import RelationalState, _initial_state, _distance, _unit_los


# =====================================================================================
# SYMPLECTIC COEFFICIENTS (Yoshida 1990)
# =====================================================================================

# 4th-order Yoshida coefficients (3 stages, palindromic composition of 2nd-order steps)
_W1_4 = 1.0 / (2.0 - 2.0**(1.0 / 3.0))
_W0_4 = - (2.0**(1.0 / 3.0)) / (2.0 - 2.0**(1.0 / 3.0))
YOSHIDA4_C = np.array([_W1_4 / 2.0, (_W0_4 + _W1_4) / 2.0, (_W0_4 + _W1_4) / 2.0, _W1_4 / 2.0])
YOSHIDA4_D = np.array([_W1_4, _W0_4, _W1_4])

# 6th-order Yoshida coefficients (Solution A, 7 stages, palindromic)
_W1_6 = -1.17767998417887e0
_W2_6 = 0.235573213359357e0
_W3_6 = 0.784513610477560e0
_W0_6 = 1.0 - 2.0 * (_W1_6 + _W2_6 + _W3_6)

_D_6 = np.array([_W3_6, _W2_6, _W1_6, _W0_6, _W1_6, _W2_6, _W3_6])
_C_6 = np.zeros(8)
_C_6[0] = _W3_6 / 2.0
_C_6[1] = (_W3_6 + _W2_6) / 2.0
_C_6[2] = (_W2_6 + _W1_6) / 2.0
_C_6[3] = (_W1_6 + _W0_6) / 2.0
_C_6[4] = (_W0_6 + _W1_6) / 2.0
_C_6[5] = (_W1_6 + _W2_6) / 2.0
_C_6[6] = (_W2_6 + _W3_6) / 2.0
_C_6[7] = _W3_6 / 2.0
YOSHIDA6_C = _C_6
YOSHIDA6_D = _D_6


def compute_newtonian_accelerations(pos: np.ndarray, masses: List[float]) -> np.ndarray:
    """Compute pairwise Newtonian accelerations for N bodies in Cartesian space."""
    N = len(masses)
    acc = np.zeros((N, 3), dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            r_vec = pos[j] - pos[i]
            r = np.linalg.norm(r_vec)
            if r <= 0:
                raise ValueError("Point-mass collision encountered during symplectic integration.")
            force_factor = G / (r**3)
            acc[i] += force_factor * masses[j] * r_vec
            acc[j] -= force_factor * masses[i] * r_vec
    return acc


def compute_system_energy(pos: np.ndarray, vel: np.ndarray, masses: List[float]) -> float:
    """Compute total mechanical energy E = T + V of N-body system."""
    N = len(masses)
    ke = 0.5 * sum(masses[i] * float(np.dot(vel[i], vel[i])) for i in range(N))
    pe = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            r = np.linalg.norm(pos[j] - pos[i])
            pe -= G * masses[i] * masses[j] / r
    return float(ke + pe)


def compute_system_angular_momentum(pos: np.ndarray, vel: np.ndarray, masses: List[float]) -> np.ndarray:
    """Compute total angular momentum vector L = sum m_i (r_i x v_i)."""
    N = len(masses)
    L = np.zeros(3, dtype=float)
    for i in range(N):
        L += masses[i] * np.cross(pos[i], vel[i])
    return L


class SymplecticThreeBodyIntegrator:
    """
    Symmetric time-reversible symplectic integrator suite for 3-body gravitational systems.
    """

    @staticmethod
    def step_verlet(pos: np.ndarray, vel: np.ndarray, masses: List[float], dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """2nd-order Velocity Verlet symplectic step."""
        a = compute_newtonian_accelerations(pos, masses)
        vel_half = vel + 0.5 * dt * a
        pos_new = pos + dt * vel_half
        a_new = compute_newtonian_accelerations(pos_new, masses)
        vel_new = vel_half + 0.5 * dt * a_new
        return pos_new, vel_new

    @staticmethod
    def step_yoshida4(pos: np.ndarray, vel: np.ndarray, masses: List[float], dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """4th-order palindromic time-reversible Yoshida symplectic step."""
        p = pos.copy()
        v = vel.copy()
        for s in range(3):
            p += YOSHIDA4_C[s] * dt * v
            a = compute_newtonian_accelerations(p, masses)
            v += YOSHIDA4_D[s] * dt * a
        p += YOSHIDA4_C[3] * dt * v
        return p, v

    @staticmethod
    def step_yoshida6(pos: np.ndarray, vel: np.ndarray, masses: List[float], dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """6th-order palindromic time-reversible Yoshida symplectic step."""
        p = pos.copy()
        v = vel.copy()
        for s in range(7):
            p += YOSHIDA6_C[s] * dt * v
            a = compute_newtonian_accelerations(p, masses)
            v += YOSHIDA6_D[s] * dt * a
        p += YOSHIDA6_C[7] * dt * v
        return p, v

    @classmethod
    def integrate(
        cls,
        triple: HierarchicalTripleSystem,
        total_time_s: float,
        dt_s: float,
        order: int = 4,
        record_every: int = 10,
        inner_angles_rad: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        outer_angles_rad: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Dict[str, Any]:
        """
        Integrate gravitational system forward with a symmetric symplectic operator.

        Parameters
        ----------
        triple : HierarchicalTripleSystem
            Stellar / planetary 3-body system definition.
        total_time_s : float
            Total physical duration of integration in seconds.
        dt_s : float
            Fixed symplectic timestep in seconds.
        order : int, default=4
            Integration order: 2 (Verlet), 4 (Yoshida4), or 6 (Yoshida6).
        record_every : int, default=10
            Stride between recorded trajectory sample points.
        """
        state, period, inputs = _initial_state(triple, inner_angles_rad, outer_angles_rad)
        cart = state.to_barycentric_cartesian()
        pos = np.array([cart[0], cart[2], cart[4]], dtype=float)
        vel = np.array([cart[1], cart[3], cart[5]], dtype=float)
        masses = [state.m1, state.m2, state.m3]

        step_fn = {
            2: cls.step_verlet,
            4: cls.step_yoshida4,
            6: cls.step_yoshida6,
        }.get(order, cls.step_yoshida4)

        n_steps = int(np.ceil(total_time_s / dt_s))
        e0 = compute_system_energy(pos, vel, masses)
        l0 = compute_system_angular_momentum(pos, vel, masses)

        times = []
        energies = []
        energy_errors = []
        pos_history = []
        vel_history = []

        start_wall = time.perf_counter()
        current_time = 0.0

        for step in range(n_steps + 1):
            if step % record_every == 0 or step == n_steps:
                e_curr = compute_system_energy(pos, vel, masses)
                dE = abs((e_curr - e0) / e0) if e0 != 0 else 0.0
                times.append(current_time)
                energies.append(e_curr)
                energy_errors.append(dE)
                pos_history.append(pos.copy())
                vel_history.append(vel.copy())

            if step < n_steps:
                pos, vel = step_fn(pos, vel, masses, dt_s)
                current_time += dt_s

        wall_time = time.perf_counter() - start_wall
        pos_arr = np.array(pos_history)  # (N_samples, 3, 3)
        vel_arr = np.array(vel_history)

        return {
            "system": triple.name,
            "order": order,
            "n_steps": n_steps,
            "dt_s": dt_s,
            "total_time_s": total_time_s,
            "total_time_years": total_time_s / year_s,
            "wall_time_s": wall_time,
            "times_s": np.array(times),
            "times_days": np.array(times) / day_s,
            "times_years": np.array(times) / year_s,
            "positions_m": pos_arr,
            "velocities_m_s": vel_arr,
            "energies_j": np.array(energies),
            "max_energy_error": float(np.max(energy_errors)),
            "final_energy_error": float(energy_errors[-1]),
            "initial_angular_momentum": l0,
            "final_angular_momentum": compute_system_angular_momentum(pos, vel, masses),
        }

    @classmethod
    def time_reversal_experiment(
        cls,
        triple: HierarchicalTripleSystem,
        n_steps: int = 1000,
        dt_s: float = 100.0,
        order: int = 4,
    ) -> Dict[str, Any]:
        """
        Execute an exact mathematical time-reversal experiment.

        Integrates forward by N steps, reverses physical velocities (v -> -v),
        integrates forward for another N steps (equivalent to running time backward),
        reverses velocities back (v -> -v), and evaluates the return closure error:
            Δx_return = ||x_final - x_initial||

        Demonstrates that symplectic integrators preserve time-reversal symmetry (T-symmetry)
        to double-precision roundoff, allowing reconstruction of primordial dynamical states.
        """
        state, period, _ = _initial_state(triple, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        cart = state.to_barycentric_cartesian()
        pos0 = np.array([cart[0], cart[2], cart[4]], dtype=float)
        vel0 = np.array([cart[1], cart[3], cart[5]], dtype=float)
        masses = [state.m1, state.m2, state.m3]

        step_fn = {
            2: cls.step_verlet,
            4: cls.step_yoshida4,
            6: cls.step_yoshida6,
        }.get(order, cls.step_yoshida4)

        # Forward pass
        pos = pos0.copy()
        vel = vel0.copy()
        for _ in range(n_steps):
            pos, vel = step_fn(pos, vel, masses, dt_s)

        fwd_pos = pos.copy()
        fwd_vel = vel.copy()

        # Time-reversal operation: T(x, v) = (x, -v)
        vel = -vel

        # Backward pass
        for _ in range(n_steps):
            pos, vel = step_fn(pos, vel, masses, dt_s)

        # Restore velocity sign: T(x, -v) = (x, v)
        vel = -vel

        characteristic_scale = float(np.linalg.norm(pos0[1] - pos0[0]))
        pos_error = float(np.max(np.linalg.norm(pos - pos0, axis=1)))
        vel_error = float(np.max(np.linalg.norm(vel - vel0, axis=1)))
        relative_closure = pos_error / characteristic_scale

        return {
            "system": triple.name,
            "order": order,
            "n_steps": n_steps,
            "dt_s": dt_s,
            "duration_days": (n_steps * dt_s) / day_s,
            "characteristic_separation_m": characteristic_scale,
            "return_pos_error_m": pos_error,
            "return_vel_error_m_s": vel_error,
            "relative_closure_precision": relative_closure,
            "is_time_reversible": bool(relative_closure < 1e-12),
        }
