"""Observed-arrival experiment using the repository's Jacobi force law.

Initial conditions are Ransom 2014 Table 1 measured elements, not post-fit
residuals. The extra clock state integrates leading proper-time corrections;
the Newtonian trajectory itself uses the unchanged repository right-hand side.
"""

import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from qrc_pulsar_bench.dynamics import _initial_state
from qrc_pulsar_bench.systems import HierarchicalTripleSystem
from qrc_pulsar_bench.constants import G, c, M_sun, day_s
from .relativity import timing_rhs

EPOCH = 55920.0
CUTOFF = (
    56436.5  # End of the Ransom 2014 fitted observation span; fixed before testing.
)
PUBLISHED = dict(
    x1=1.21752844,
    p1=1.629401788,
    eps11=6.8567e-4,
    eps21=-9.171e-5,
    tasc1=55920.407717436,
    acos1=1.4900,
    q1=0.13737,
    x2=74.6727101,
    p2=327.257541,
    eps12=3.5186279e-2,
    eps22=-3.462131e-3,
    tasc2=56233.935815,
    acos2=91.42,
    delta_node_deg=0.0027,
)


def orbit_angles(period, eps1, eps2, tasc):
    """True ascending node -> eccentric anomaly -> mean anomaly -> initial f.

    A linear time phase is mean anomaly, not true anomaly. omega must be carried.
    """
    ecc = float(np.hypot(eps1, eps2))
    omega = float(np.arctan2(eps1, eps2))
    f_node = -omega
    E_node = np.arctan2(np.sqrt(1 - ecc * ecc) * np.sin(f_node), ecc + np.cos(f_node))
    M_node = E_node - ecc * np.sin(E_node)
    M = M_node + 2 * np.pi * (EPOCH - tasc) / period
    E = M
    for _ in range(20):
        E -= (E - ecc * np.sin(E) - M) / (1 - ecc * np.cos(E))
    f = 2 * np.arctan2(
        np.sqrt(1 + ecc) * np.sin(E / 2), np.sqrt(1 - ecc) * np.cos(E / 2)
    )
    return ecc, omega, float(f)


def initial_state(epoch_offsets_s=(0.0, 0.0), parameters=None):
    p = dict(PUBLISHED if parameters is None else parameters)
    # Infer consistent masses from the measured semimajor axes and mass ratio.
    a_p1 = np.hypot(p["x1"], p["acos1"]) * c
    a_cm2 = np.hypot(p["x2"], p["acos2"]) * c
    a_rel1 = a_p1 * (1 + 1 / p["q1"])
    m12 = (2 * np.pi / (p["p1"] * day_s)) ** 2 * a_rel1**3 / G
    mass_function = (2 * np.pi / (p["p2"] * day_s)) ** 2 * a_cm2**3 / G
    qout = brentq(
        lambda q: q**3 / (1 + q) ** 2 - mass_function / m12, 1e-5, 100, xtol=1e-15
    )
    m1 = m12 / (1 + p["q1"])
    m2 = m12 - m1
    m3 = qout * m12
    e1, w1, f1 = orbit_angles(
        p["p1"], p["eps11"], p["eps21"], p["tasc1"] + epoch_offsets_s[0] / day_s
    )
    e2, w2, f2 = orbit_angles(
        p["p2"], p["eps12"], p["eps22"], p["tasc2"] + epoch_offsets_s[1] / day_s
    )
    system = HierarchicalTripleSystem(
        "J0337 Ransom 2014 measured elements",
        "Frozen published initial conditions",
        "https://arxiv.org/abs/1401.0535",
        m1 / M_sun,
        m2 / M_sun,
        m3 / M_sun,
        p["p1"],
        p["p2"],
        e1,
        e2,
        p["acos1"] / np.hypot(p["x1"], p["acos1"]),
        p["acos2"] / np.hypot(p["x2"], p["acos2"]),
        365.953363096,
    )
    # Publisher uses clockwise node rotation; our orbital helper uses counterclockwise.
    state, period, metadata = _initial_state(
        system, (0.0, w1, f1), (-np.deg2rad(p["delta_node_deg"]), w2, f2)
    )
    metadata["osculating_elements"] = p
    metadata["reference_published_elements"] = dict(PUBLISHED)
    metadata["element_source"] = (
        "Ransom 2014 Table 1"
        if parameters is None
        else "caller-supplied elements; may be fitted"
    )
    metadata["epoch_offsets_s"] = list(epoch_offsets_s)
    metadata["note"] = (
        "Measured Ransom 2014 Table 1 osculating elements at MJD 55920; ascending-node epochs converted through Kepler equation, preserving periapsis angles."
    )
    metadata["jacobi_initial_state_si"] = np.concatenate(
        (state.r1, state.v1, state.r2, state.v2)
    ).tolist()
    return state, metadata


def evolve(
    arrival_seconds,
    epoch_offsets_s=(0.0, 0.0),
    rtol=1e-13,
    parameters=None,
    coupled=True,
    post_newtonian=False,
):
    arrival_seconds = np.asarray(arrival_seconds, dtype=float)
    if (
        arrival_seconds.ndim != 1
        or not len(arrival_seconds)
        or not np.all(np.isfinite(arrival_seconds))
        or np.min(arrival_seconds) < 0
    ):
        raise ValueError(
            "Arrival offsets must be a nonempty finite vector at or after the reference epoch"
        )
    if not np.isfinite(rtol) or rtol <= 0:
        raise ValueError("rtol must be finite and positive")
    state, metadata = initial_state(epoch_offsets_s, parameters)
    metadata["force_model"] = (
        "order-reduced first-PN EIH" if post_newtonian else "Newtonian"
    )
    masses = [state.m1, state.m2, state.m3]
    f2 = state.m2 / state.M12
    F3 = state.m3 / state.M_tot
    y0 = np.concatenate((state.r1, state.v1, state.r2, state.v2, np.zeros(7)))
    masses = np.asarray(masses)

    def clock_rate(y):
        r1 = y[:3]
        r2 = y[6:9]
        vp = -F3 * y[9:12] - f2 * y[3:6]
        return (
            0.5 * np.dot(vp, vp)
            + G * state.m2 / np.linalg.norm(r1)
            + G * state.m3 / np.linalg.norm(r2 + f2 * r1)
        ) / c**2

    def rhs(t, y):
        if coupled:
            return timing_rhs(t, y, masses, post_newtonian)
        else:
            if post_newtonian:
                raise ValueError("EIH requires coupled dynamics")
            derivative = np.r_[
                y[3:6],
                -G * state.M12 * y[:3] / np.linalg.norm(y[:3]) ** 3,
                y[9:12],
                -G * state.M_tot * y[6:9] / np.linalg.norm(y[6:9]) ** 3,
            ]
        return np.r_[derivative, clock_rate(y), np.zeros(6)]

    start = time.perf_counter()
    sol = solve_ivp(
        rhs,
        (0.0, float(np.max(arrival_seconds)) + 200),
        y0,
        method="DOP853",
        rtol=rtol,
        atol=np.array([1e-5] * 12 + [1e-13] + [1e-7] * 3 + [1e-12] * 3),
        dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol, state, metadata, time.perf_counter() - start


def observables(sol, state, arrival_seconds, clock=True, shapiro=True):
    t = np.asarray(arrival_seconds)
    te = t.copy()
    f2 = state.m2 / state.M12
    F3 = state.m3 / state.M_tot
    solution = getattr(sol, "solution", sol)
    lower, upper = solution.t[0], solution.t[-1]
    for _ in range(5):
        if not np.all(np.isfinite(te)) or np.any(te < lower) or np.any(te > upper):
            raise ValueError(
                "Emission time is outside the integrated interval; extrapolation is disabled"
            )
        y = sol.sol(te)
        roemer = (F3 * y[8] + f2 * y[2] - y[15]) / c
        sh = np.zeros_like(roemer)
        if shapiro:
            d12 = y[:3]
            d13 = y[6:9] + f2 * y[:3]
            for mass, d, ref in [
                (state.m2, d12, state.r1),
                (state.m3, d13, state.r2 + f2 * state.r1),
            ]:
                radius = np.linalg.norm(d, axis=0)
                sh -= (
                    2 * G * mass / c**3 * np.log((radius - d[2]) / np.linalg.norm(ref))
                )
        te = t - roemer - sh
    if np.any(te < lower) or np.any(te > upper):
        raise ValueError(
            "Emission time is outside the integrated interval; extrapolation is disabled"
        )
    y = sol.sol(te)
    einstein = y[12] if clock else np.zeros_like(t)
    # Return correction directly to avoid subtracting two ~1e8-second numbers.
    total = roemer + sh + einstein
    return dict(
        delay_s=total,
        proper_seconds=t - total,
        emission_seconds=te,
        roemer_s=roemer,
        einstein_s=einstein,
        shapiro_s=sh,
    )


def summarize(residual_s, unc_us, mask):
    r = residual_s[mask] * 1e6
    u = unc_us[mask]
    weights = (u.min() / u) ** 2
    return dict(
        n=int(np.sum(mask)),
        rms_us=float(np.sqrt(np.mean(r * r))),
        weighted_rms_us=float(np.sqrt(np.sum(weights * r * r) / weights.sum())),
        mean_us=float(np.mean(r)),
        p95_absolute_us=float(np.quantile(np.abs(r), 0.95)),
        max_absolute_us=float(np.max(np.abs(r))),
    )
