"""Chaos parameterization via Moment of Inertia and Power Spectral Slopes.

Instead of tuning chaotic systems for artificial stability (e.g. PID damping),
this module extracts invariant physical parameters of the chaos:
1. Polar Moment of Inertia I(t) = sum m_i |r_i - r_cm|^2 and the Lagrange-Jacobi
   identity: d^2 I / dt^2 = 2(2K + U) = 4E - 2U.
2. Power Spectral Density (PSD) and spectral power slope beta: P(f) ~ f^(-beta).
   - beta ~ 0: Regular quasi-periodic (KAM tori, discrete line spectra).
   - beta ~ 1: Pink / flicker noise (edge of chaos, multi-scale memory, QRC optimal).
   - beta ~ 2: Red / Brownian noise (developed Hamiltonian chaos, phase-space wandering).
3. Virial fluctuation index: departure from virial equilibrium 2K + U = 0.
4. Spectral Entropy: degree of frequency dispersion across modes.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy import signal
from .constants import G


@dataclass(frozen=True)
class ChaosFingerprint:
    """Invariant chaos parameters extracted from an N-body trajectory."""
    spectral_slope_beta: float
    spectral_fit_r2: float
    spectral_entropy: float
    virial_ratio_mean: float
    virial_fluctuation_index: float
    moment_of_inertia_mean: float
    moment_of_inertia_rate: float
    chaos_regime: str
    dominant_frequency: float


def compute_moment_of_inertia(
    masses: np.ndarray,
    positions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute polar moment of inertia I(t) and hyper-radius R(t).

    Args:
        masses: (N,) array of masses in kg.
        positions: (T, N, 3) or (N, 3) array of Cartesian coordinates in meters.

    Returns:
        I: (T,) array of moment of inertia (kg * m^2).
        R: (T,) array of mass-weighted hyper-radius (m).
    """
    masses = np.asarray(masses, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    m_tot = np.sum(masses)

    if positions.ndim == 2:
        # Single time slice (N, 3)
        r_cm = np.sum(masses[:, None] * positions, axis=0) / m_tot
        r_rel = positions - r_cm
        I = np.sum(masses * np.sum(r_rel**2, axis=-1))
        R = np.sqrt(I / m_tot)
        return float(I), float(R)

    # Trajectory (T, N, 3)
    r_cm = np.sum(masses[None, :, None] * positions, axis=1) / m_tot  # (T, 3)
    r_rel = positions - r_cm[:, None, :]  # (T, N, 3)
    dist_sq = np.sum(r_rel**2, axis=-1)   # (T, N)
    I = np.sum(masses[None, :] * dist_sq, axis=-1)  # (T,)
    R = np.sqrt(I / m_tot)
    return I, R


def compute_virial_kinetics(
    masses: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    g_const: float = G,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate kinetic energy, potential energy, and the Lagrange-Jacobi identity.

    Lagrange-Jacobi identity:
        (1/2) * d^2 I / dt^2 = 2K + U = 4E - 2U
    In virial equilibrium, <2K + U> = 0.

    Args:
        masses: (N,) array of masses.
        positions: (T, N, 3) Cartesian positions.
        velocities: (T, N, 3) Cartesian velocities.
        g_const: Gravitational constant.

    Returns:
        K: Kinetic energy array (T,).
        U: Potential energy array (T,).
        E: Total energy array (T,).
        I_ddot_analytical: Analytical d^2 I / dt^2 = 2 * (2K + U).
        Q_vir: Virial ratio 2K / |U|.
    """
    masses = np.asarray(masses, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)

    is_single = positions.ndim == 2
    if is_single:
        positions = positions[None, ...]
        velocities = velocities[None, ...]

    T_steps, N_bodies, _ = positions.shape

    # Kinetic energy: K = 1/2 sum m_i |v_i|^2
    v_sq = np.sum(velocities**2, axis=-1)  # (T, N)
    K = 0.5 * np.sum(masses[None, :] * v_sq, axis=-1)  # (T,)

    # Potential energy: U = -G sum_{i<j} m_i m_j / r_ij
    U = np.zeros(T_steps, dtype=np.float64)
    for i in range(N_bodies):
        for j in range(i + 1, N_bodies):
            r_ij = np.linalg.norm(positions[:, i, :] - positions[:, j, :], axis=-1)
            # Avoid division by zero at collision
            r_ij = np.maximum(r_ij, 1e-12)
            U -= g_const * masses[i] * masses[j] / r_ij

    E = K + U
    # Analytical d^2 I / dt^2 = 2 * (2K + U)
    I_ddot = 2.0 * (2.0 * K + U)
    Q_vir = 2.0 * K / np.maximum(np.abs(U), 1e-12)

    if is_single:
        return float(K[0]), float(U[0]), float(E[0]), float(I_ddot[0]), float(Q_vir[0])
    return K, U, E, I_ddot, Q_vir


def compute_power_spectral_slope(
    sig: np.ndarray,
    dt: float,
    f_min: Optional[float] = None,
    f_max: Optional[float] = None,
    nperseg: Optional[int] = None,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Compute power spectral density and fit power-law slope beta: P(f) ~ f^(-beta).

    log10(P(f)) = -beta * log10(f) + C

    Args:
        sig: 1D time series signal (e.g. Moment of Inertia, coordinate, or velocity).
        dt: Sampling time interval (seconds).
        f_min: Lower cutoff frequency for power-law fit. Defaults to 2 / total_time.
        f_max: Upper cutoff frequency. Defaults to Nyquist / 2.
        nperseg: Length of each segment for Welch PSD.

    Returns:
        beta: Spectral power slope exponent (positive means decay with frequency).
        r2: Coefficient of determination of the log-log linear fit.
        freqs: Array of frequency bins (Hz).
        psd: Power spectral density array.
    """
    sig = np.asarray(sig, dtype=np.float64)
    n = len(sig)
    if n < 8:
        raise ValueError("Time series must have at least 8 samples for spectral analysis.")

    # Detrend linear drift (e.g. if system is dispersing ballistically)
    sig_detrend = signal.detrend(sig, type="linear")

    fs = 1.0 / dt
    if nperseg is None:
        nperseg = min(n, 1024)
    if nperseg > n:
        nperseg = n

    freqs, psd = signal.welch(
        sig_detrend,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        scaling="density",
    )

    # Exclude DC zero frequency
    valid = freqs > 0.0
    freqs = freqs[valid]
    psd = psd[valid]

    if len(freqs) < 4:
        # Fall back to standard periodogram
        freqs, psd = signal.periodogram(sig_detrend, fs=fs, window="hann")
        valid = freqs > 0.0
        freqs = freqs[valid]
        psd = psd[valid]

    t_total = n * dt
    if f_min is None:
        f_min = 2.0 / t_total
    if f_max is None:
        f_max = fs * 0.45

    mask = (freqs >= f_min) & (freqs <= f_max) & (psd > 1e-30)
    if np.sum(mask) < 3:
        # Broaden mask if too restrictive
        mask = psd > 1e-30

    log_f = np.log10(freqs[mask])
    log_p = np.log10(psd[mask])

    # Linear regression: log_p = slope * log_f + intercept
    # beta = -slope
    p = np.polyfit(log_f, log_p, 1)
    slope, intercept = p[0], p[1]
    beta = -slope

    # Compute R^2 goodness of fit
    fit_vals = slope * log_f + intercept
    ss_res = np.sum((log_p - fit_vals) ** 2)
    ss_tot = np.sum((log_p - np.mean(log_p)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    return float(beta), float(r2), freqs, psd


def compute_spectral_entropy(psd: np.ndarray) -> float:
    """Compute normalized Shannon spectral entropy H in [0, 1].

    H = 0: Single pure sinusoidal tone (zero disorder).
    H = 1: Uniform white noise distribution (maximal disorder).
    """
    psd_pos = psd[psd > 1e-30]
    if len(psd_pos) <= 1:
        return 0.0
    p = psd_pos / np.sum(psd_pos)
    h = -np.sum(p * np.log(p))
    h_max = np.log(len(p))
    return float(np.clip(h / h_max, 0.0, 1.0))


def extract_chaos_fingerprint(
    times: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    g_const: float = G,
) -> ChaosFingerprint:
    """Extract the invariant chaos parameters from an N-body trajectory.

    Args:
        times: (T,) array of timestamps in seconds.
        positions: (T, N, 3) array of Cartesian coordinates.
        velocities: (T, N, 3) array of Cartesian velocities.
        masses: (N,) array of masses.
        g_const: Gravitational constant.

    Returns:
        ChaosFingerprint with spectral slope beta, virial fluctuation, and regime.
    """
    times = np.asarray(times, dtype=np.float64)
    dt = float(np.median(np.diff(times)))

    I, R = compute_moment_of_inertia(masses, positions)
    K, U, E, I_ddot, Q_vir = compute_virial_kinetics(masses, positions, velocities, g_const)

    # Power slope of the moment of inertia
    beta, r2, freqs, psd = compute_power_spectral_slope(I, dt)
    h_spec = compute_spectral_entropy(psd)

    # Dominant frequency
    dom_freq = float(freqs[np.argmax(psd)])

    # Virial statistics
    virial_diff = 2.0 * K + U
    mean_k = np.mean(K)
    virial_fluct = float(np.std(virial_diff) / np.maximum(mean_k, 1e-12))
    virial_ratio_mean = float(np.mean(Q_vir))

    # Moment of inertia trend
    mean_I = float(np.mean(I))
    # Relative expansion rate: dI/dt / I
    poly = np.polyfit(times, I, 1)
    rate_I = float(poly[0] / np.maximum(mean_I, 1e-12))

    # Chaos classification regime
    if rate_I > 0.05:
        regime = "disrupting_ejection"
    elif beta < 0.4 and h_spec < 0.45:
        regime = "quasi_periodic_stable"
    elif 0.5 <= beta <= 1.4:
        regime = "intermittent_pink_edge_of_chaos"
    elif 1.4 < beta <= 2.6:
        regime = "developed_red_chaos"
    else:
        regime = "diffusive_subrange"

    return ChaosFingerprint(
        spectral_slope_beta=beta,
        spectral_fit_r2=r2,
        spectral_entropy=h_spec,
        virial_ratio_mean=virial_ratio_mean,
        virial_fluctuation_index=virial_fluct,
        moment_of_inertia_mean=mean_I,
        moment_of_inertia_rate=rate_I,
        chaos_regime=regime,
        dominant_frequency=dom_freq,
    )
