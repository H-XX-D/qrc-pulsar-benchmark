"""
=====================================================================================
QRC-PULSAR-BENCH: Quantum / Dynamical Relational Reservoir Computing Engine
=====================================================================================
Operationalizes the non-linear Jacobi multi-body gravitational system as a
dynamical reservoir computing substrate:
  1. Jacobi Reservoir States: Collects gauge-eliminated 12-dimensional relative
     Jacobi vectors [r1, v1, r2, v2], augmented by scale-invariant Montgomery shape
     coordinates [w1, w2, w3] and hyper-radius R.
  2. Memory Capacity (MC): Evaluates fading memory capacity across delayed target
     reconstructions: MC = sum_{tau} r^2(u(t-tau), y_pred(t)).
  3. Effective Reservoir Dimension: Evaluates spectral participation ratio of the
     reservoir covariance matrix: D_eff = (Tr C)^2 / Tr(C^2).
  4. Out-of-Sample Phase Reconstruction: Trains linear ridge readout W_out to
     predict pulsar phase variations and orbital clock drift.
=====================================================================================
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np

from .constants import G, c, M_sun, day_s
from .dynamics import RelationalThreeBodyIntegrator, _initial_state
from .systems import HierarchicalTripleSystem, PSR_J0337_1715
from .shape_sphere import ShapeSphereReduction


@dataclass(frozen=True)
class QRCBenchmarkResult:
    """Certified metrics from a Relational Reservoir Computing benchmark."""
    system_name: str
    num_samples: int
    duration_days: float
    reservoir_dim: int
    effective_dimension: float
    total_memory_capacity: float
    max_lag_capacity: int
    train_nrmse: float
    test_nrmse: float
    test_r2: float
    execution_time_s: float


class JacobiRelationalReservoir:
    """
    Dynamical reservoir computing interface powered by the 12-variable Jacobi
    three-body equations of motion.
    """

    def __init__(self, system: HierarchicalTripleSystem = PSR_J0337_1715, enable_1pn: bool = False):
        self.system = system
        self.enable_1pn = enable_1pn

    def generate_reservoir_states(
        self,
        duration_days: float = 100.0,
        samples_per_day: int = 10,
        rtol: float = 1e-10,
        atol: float = 1e-11
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrates the Jacobi system and extracts the non-linear reservoir feature matrix.

        Returns:
            times_days: (T,) array of observation timestamps in days.
            states: (T, D) matrix of non-linear reservoir states (D = 18 features:
                    r1 (3), v1 (3), r2 (3), v2 (3), |r1|, |r2|, R, w1, w2, w3).
        """
        duration_s = duration_days * day_s
        num_orbits = duration_days / self.system.Pb1_days
        total_samples = int(duration_days * samples_per_day)

        integrator_out = RelationalThreeBodyIntegrator.integrate(
            self.system,
            num_inner_orbits=num_orbits,
            num_output_points=total_samples,
            rtol=rtol,
            atol=atol,
            enable_1pn=self.enable_1pn
        )

        times_days = integrator_out['t_eval_s'] / day_s
        # sol_y shape: (12, T) containing [r1 (0:3), v1 (3:6), r2 (6:9), v2 (9:12)]
        sol_y = integrator_out['sol_y']
        T = sol_y.shape[1]

        r1 = sol_y[0:3, :].T  # (T, 3)
        v1 = sol_y[3:6, :].T  # (T, 3)
        r2 = sol_y[6:9, :].T  # (T, 3)
        v2 = sol_y[9:12, :].T # (T, 3)

        r1_norm = np.linalg.norm(r1, axis=1, keepdims=True)
        r2_norm = np.linalg.norm(r2, axis=1, keepdims=True)

        # Barycentric positions to compute Shape Sphere coordinates
        cart = integrator_out['cartesian_y']  # (18, T)
        pos1 = cart[0:3, :].T
        pos2 = cart[6:9, :].T
        pos3 = cart[12:15, :].T

        w1_arr = np.zeros(T)
        w2_arr = np.zeros(T)
        w3_arr = np.zeros(T)
        R_arr = np.zeros(T)

        for i in range(T):
            d12 = np.linalg.norm(pos1[i] - pos2[i])
            d23 = np.linalg.norm(pos2[i] - pos3[i])
            d31 = np.linalg.norm(pos3[i] - pos1[i])
            w1, w2, w3 = ShapeSphereReduction.compute_shape_coordinates(d12, d23, d31)
            w1_arr[i] = w1
            w2_arr[i] = w2
            w3_arr[i] = w3
            R_arr[i] = np.sqrt(d12**2 + d23**2 + d31**2)

        # Assemble full reservoir feature matrix: 16 non-linear dimensions
        features = np.column_stack([
            r1 / np.mean(r1_norm),
            v1 / np.std(v1),
            r2 / np.mean(r2_norm),
            v2 / np.std(v2),
            r1_norm / np.mean(r1_norm),
            r2_norm / np.mean(r2_norm),
            R_arr[:, None] / np.mean(R_arr),
            w1_arr[:, None],
            w2_arr[:, None],
            w3_arr[:, None],
        ])

        return times_days, features


def compute_effective_dimension(states: np.ndarray) -> float:
    """
    Computes the spectral participation ratio / effective dimension of the reservoir:
        D_eff = (Tr C)^2 / Tr(C^2) = (sum lambda_i)^2 / sum(lambda_i^2)
    """
    states_centered = states - np.mean(states, axis=0)
    cov = (states_centered.T @ states_centered) / len(states)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-14]
    if len(eigvals) == 0:
        return 0.0
    sum_e = np.sum(eigvals)
    sum_e_sq = np.sum(eigvals**2)
    return float((sum_e**2) / sum_e_sq)


def compute_fading_memory_capacity(
    states: np.ndarray,
    input_signal: np.ndarray,
    max_lag: int = 15,
    ridge_alpha: float = 1e-6
) -> Tuple[float, np.ndarray]:
    """
    Computes Jaeger's Memory Capacity (MC):
        MC = sum_{tau=1}^{max_lag} r^2(u(t - tau), y_pred_tau(t))
    where y_pred_tau is trained via linear ridge regression on reservoir states.
    """
    T, D = states.shape
    r2_scores = np.zeros(max_lag)

    # Standardize states
    s_mean = np.mean(states, axis=0)
    s_std = np.std(states, axis=0) + 1e-12
    X = (states - s_mean) / s_std
    # Add bias term
    X_bias = np.column_stack([np.ones(T), X])

    for tau in range(1, max_lag + 1):
        y_target = input_signal[:-tau]
        X_src = X_bias[tau:]

        n_train = int(0.7 * len(y_target))
        X_train, X_test = X_src[:n_train], X_src[n_train:]
        y_train, y_test = y_target[:n_train], y_target[n_train:]

        # Ridge regression
        reg = ridge_alpha * np.eye(X_bias.shape[1])
        reg[0, 0] = 0.0  # Do not regularize bias
        w = np.linalg.solve(X_train.T @ X_train + reg, X_train.T @ y_train)

        y_pred = X_test @ w
        # Pearson correlation squared r^2
        if np.std(y_pred) > 1e-12 and np.std(y_test) > 1e-12:
            corr = np.corrcoef(y_test, y_pred)[0, 1]
            r2 = float(corr**2) if np.isfinite(corr) and corr > 0 else 0.0
        else:
            r2 = 0.0
        r2_scores[tau - 1] = r2

    total_mc = float(np.sum(r2_scores))
    return total_mc, r2_scores


def benchmark_qrc_reservoir(
    system: HierarchicalTripleSystem = PSR_J0337_1715,
    duration_days: float = 150.0,
    samples_per_day: int = 10,
    max_lag: int = 15,
    enable_1pn: bool = False
) -> QRCBenchmarkResult:
    """
    Runs the complete QRC Dynamical Reservoir benchmark.
    """
    import time
    t0 = time.perf_counter()

    res = JacobiRelationalReservoir(system, enable_1pn=enable_1pn)
    times_days, states = res.generate_reservoir_states(
        duration_days=duration_days,
        samples_per_day=samples_per_day
    )

    T, D = states.shape
    d_eff = compute_effective_dimension(states)

    # Synthetic multi-frequency input signal driving pulse phase deviations
    # (inner orbit + outer orbit beat frequency + random fluctuations)
    f1 = 1.0 / system.Pb1_days
    f2 = 1.0 / system.Pb2_days
    rng = np.random.default_rng(42)
    carrier = np.sin(2 * np.pi * f1 * times_days) + 0.3 * np.cos(2 * np.pi * f2 * times_days)
    noise = 0.05 * rng.standard_normal(T)
    input_signal = carrier + noise

    mc_total, mc_curve = compute_fading_memory_capacity(
        states, input_signal, max_lag=max_lag
    )

    # Task: Reconstruct future timing phase (1-step forward prediction)
    target = np.roll(input_signal, -1)
    target[-1] = target[-2]

    n_train = int(0.7 * T)
    X_train, X_test = states[:n_train], states[n_train:]
    y_train, y_test = target[:n_train], target[n_train:]

    reg = 1e-5 * np.eye(D)
    w_out = np.linalg.solve(X_train.T @ X_train + reg, X_train.T @ y_train)

    train_pred = X_train @ w_out
    test_pred = X_test @ w_out

    train_nrmse = float(np.sqrt(np.mean((y_train - train_pred)**2)) / np.std(y_train))
    test_nrmse = float(np.sqrt(np.mean((y_test - test_pred)**2)) / np.std(y_test))

    ss_res = np.sum((y_test - test_pred)**2)
    ss_tot = np.sum((y_test - np.mean(y_test))**2)
    test_r2 = float(1.0 - ss_res / ss_tot)

    elapsed = time.perf_counter() - t0

    return QRCBenchmarkResult(
        system_name=system.name,
        num_samples=T,
        duration_days=duration_days,
        reservoir_dim=D,
        effective_dimension=d_eff,
        total_memory_capacity=mc_total,
        max_lag_capacity=max_lag,
        train_nrmse=train_nrmse,
        test_nrmse=test_nrmse,
        test_r2=test_r2,
        execution_time_s=elapsed
    )
