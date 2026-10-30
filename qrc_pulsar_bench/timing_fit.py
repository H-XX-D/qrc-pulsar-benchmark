"""Early-only variable-projection calibration of physical orbital parameters.

Input arrays must have already been selected for calibration. Prediction and
scoring are separate operations. No stored residual or derivative column is
accepted as a fit target; astrometry derivatives are a separately computed
observation-model input. Covariances are local, conditional white-noise estimates.
"""

from dataclasses import dataclass
import numpy as np
from .observed_timing import PUBLISHED, evolve, observables

PARAMETERS = tuple(PUBLISHED)
# Numerical optimization units based on Table 1; eps11 uses one tenth of its
# reported error. These are NOT priors or a substitute for joint covariance.
SCALES = np.array(
    [
        4e-8,
        5e-9,
        2e-9,
        2e-8,
        17e-9,
        0.0005,
        0.00004,
        8e-7,
        7e-6,
        3e-9,
        11e-9,
        7e-6,
        0.04,
        0.0006,
    ]
)
PUBLISHED_MARGINAL_SIGMAS = SCALES.copy()
PUBLISHED_MARGINAL_SIGMAS[PARAMETERS.index("eps11")] *= 10
REFERENCE_FREQUENCY = 365.953363096
YEAR = 365.25 * 86400


def orbital_parameters(offsets):
    return {
        name: PUBLISHED[name] + float(offsets[i]) * SCALES[i]
        for i, name in enumerate(PARAMETERS)
    }


def predict(data, offsets, post_newtonian=False):
    sol, state, metadata, wall = evolve(
        data["tcb_s"],
        parameters=orbital_parameters(offsets),
        post_newtonian=post_newtonian,
    )
    result = observables(sol, state, data["tcb_s"])
    return result, metadata, wall


@dataclass
class TimingDesign:
    reference_time: float
    reference_pulse: int
    center: float
    astrometry: str = "none"
    dm_order: int = 0
    astrometry_priors: bool = False

    @classmethod
    def from_training(
        cls, data, astrometry="none", dm_order=0, astrometry_priors=False
    ):
        return cls(
            float(data["tcb_s"][0]),
            int(data["pulse_number"][0]),
            float(np.mean(data["tcb_s"])),
            astrometry,
            dm_order,
            astrometry_priors,
        )

    def priors(self, names):
        # Independent 2014 VLBA sky position and optical distance, not orbital
        # marginals from the same TOAs. The distance prior is locally transformed.
        values = {
            "RAJ": (0.0, 1.95),
            "DECJ": (0.0, 2.0),
            "PX": (1000 / 1300, 1000 * 80 / 1300**2),
        }
        return [
            (names.index(name), mean, sigma)
            for name, (mean, sigma) in values.items()
            if self.astrometry_priors and name in names
        ]

    def matrix(self, data, prediction):
        years = (prediction["proper_seconds"] - self.center) / YEAR
        # Keep the original constant-DM column for exact baseline comparison.
        dm = (1400 / data["frequency_mhz"]) ** 2
        cols = [
            np.ones(len(years)),
            years,
            years**2,
            data["observatory"] == "gbt",
            data["observatory"] == "wsrt",
            dm,
        ]
        names = [
            "phase_s",
            "spin_s_per_year",
            "spindown_s_per_year2",
            "gbt_s",
            "wsrt_s",
            "dm0_s_at_1400MHz",
        ]
        if self.astrometry not in ["none", "position", "motion"]:
            raise ValueError("Unknown astrometry specification")
        indices = (
            []
            if self.astrometry == "none"
            else ([0, 1, 4] if self.astrometry == "position" else list(range(5)))
        )
        for i in indices:
            cols.append(data["astrometry_design_s"][:, i])
            names.append(str(data["astrometry_names"][i]))
        if self.dm_order not in [0, 1, 2]:
            raise ValueError("dm_order must be 0, 1 or 2")
        for power in range(1, self.dm_order + 1):
            cols.append(dm * years**power)
            names.append(f"dm{power}_s_at_1400MHz_per_year{power}")
        return np.column_stack(cols), names

    def target(self, data, prediction):
        observed = (data["tcb_s"] - self.reference_time) - (
            data["pulse_number"] - self.reference_pulse
        ) / REFERENCE_FREQUENCY
        return observed - prediction["delay_s"]


def linear_fit(data, prediction, design):
    matrix, names = design.matrix(data, prediction)
    sigma = data["uncertainty_us"] * 1e-6
    weighted = matrix / sigma[:, None]
    scales = np.linalg.norm(weighted, axis=0)
    target = design.target(data, prediction) / sigma
    for index, mean, width in design.priors(names):
        row = np.zeros(matrix.shape[1])
        row[index] = 1 / width
        weighted = np.vstack((weighted, row))
        target = np.r_[target, mean / width]
    coefficient, _, rank, singular = np.linalg.lstsq(
        weighted / scales, target, rcond=1e-12
    )
    if rank != matrix.shape[1]:
        raise ValueError("Rank-deficient nuisance design")
    coefficient /= scales
    residual = design.target(data, prediction) - matrix @ coefficient
    return residual, coefficient, matrix, names


def calibrate(
    data,
    astrometry="none",
    dm_order=0,
    post_newtonian=False,
    start=None,
    max_iterations=5,
    callback=None,
    astrometry_priors=False,
    active_parameters=None,
):
    """Calibrate ONLY the explicitly supplied training arrays; return a frozen fit."""
    design = TimingDesign.from_training(data, astrometry, dm_order, astrometry_priors)
    offsets = (
        np.zeros(len(PARAMETERS)) if start is None else np.array(start, dtype=float)
    )
    active = (
        list(range(len(PARAMETERS)))
        if active_parameters is None
        else [PARAMETERS.index(name) for name in active_parameters]
    )
    if not active or len(set(active)) != len(active):
        raise ValueError("Select distinct active orbital parameters")
    history = []
    sigma = data["uncertainty_us"] * 1e-6

    def evaluate(x):
        prediction, meta, wall = predict(data, x, post_newtonian)
        residual, linear, matrix, names = linear_fit(data, prediction, design)
        prior_residual = np.array(
            [
                (linear[index] - mean) / width
                for index, mean, width in design.priors(names)
            ]
        )
        return (
            np.r_[residual / sigma, prior_residual],
            prediction,
            linear,
            matrix,
            names,
            meta,
        )

    current = evaluate(offsets)
    for iteration in range(max_iterations):
        columns = []
        for index in active:
            step = 2.0 if PARAMETERS[index] == "delta_node_deg" else 10.0
            delta = np.zeros(len(offsets))
            delta[index] = step
            plus = evaluate(offsets + delta)[0]
            minus = evaluate(offsets - delta)[0]
            columns.append((plus - minus) / (2 * step))
        jacobian = np.column_stack(columns)
        active_update, _, rank, singular = np.linalg.lstsq(
            jacobian, -current[0], rcond=1e-9
        )
        if rank != len(active):
            raise ValueError(
                "Rank-deficient orbital calibration; reduce the active parameter set"
            )
        update = np.zeros(len(offsets))
        update[active] = active_update
        chi = float(current[0] @ current[0])
        accepted = False
        for fraction in [1.0, 0.5, 0.25, 0.125]:
            candidate = offsets + fraction * update
            try:
                trial = evaluate(candidate)
            except (ValueError, RuntimeError):
                continue
            if trial[0] @ trial[0] < chi:
                offsets = candidate
                current = trial
                accepted = True
                break
        record = dict(
            iteration=iteration,
            chi_squared=float(current[0] @ current[0]),
            rank=int(rank),
            singular_values=singular.tolist(),
            accepted=accepted,
            offsets_scaled=offsets.tolist(),
            rms_us=float(
                np.sqrt(np.mean((current[0][: len(sigma)] * sigma * 1e6) ** 2))
            ),
        )
        history.append(record)
        if callback:
            callback(record)
        if not accepted or chi - record["chi_squared"] < 0.02:
            break
    return dict(
        offsets=offsets,
        linear=current[2],
        linear_names=current[4],
        design=design,
        history=history,
        metadata=current[5],
        post_newtonian=post_newtonian,
        residual_s=current[0][: len(sigma)] * sigma,
    )


def evaluate_frozen(data, fit):
    prediction, metadata, wall = predict(data, fit["offsets"], fit["post_newtonian"])
    matrix, names = fit["design"].matrix(data, prediction)
    residual = fit["design"].target(data, prediction) - matrix @ fit["linear"]
    return residual, prediction, metadata, wall
