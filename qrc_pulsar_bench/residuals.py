"""Descriptive analysis of post-fit residuals, with explicit data schemas.

No timing predictions or out-of-sample forecasts are generated here.
"""

from pathlib import Path
from numbers import Integral

import numpy as np
import pandas as pd
from scipy.linalg import lstsq


UNIT_TO_US = {"us": 1.0, "ns": 1e-3, "s": 1e6}
# Archibald et al. (2018) Table 1 spin frequency; the Zenodo 1241122 phase
# residuals come from that fit, so convert turns to time with it.
ARCHIBALD_SPIN_FREQUENCY_HZ = 365.953363080

ARCHIBALD_SEP = {
    "value": -1.09e-6,
    "uncertainty_68": 0.74e-6,
    "absolute_bound_95": 2.6e-6,
    "source": "https://arxiv.org/abs/1807.02059v1",
    "origin": "Published result, not inferred by this package",
}


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _vector(values, name):
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a one-dimensional array of finite numbers.")
    return result


def _rms(values):
    scale = np.max(np.abs(values))
    return float(scale * np.sqrt(np.mean((values / scale)**2))) if scale else 0.0


def inspect_residuals(mjd, residual_us, uncertainty_us, num_windows=10):
    """Summarize residuals without assuming the upstream fit's degrees of freedom.

    RMS is about zero; std is about the sample mean (ddof=0). Chi-square/N
    is descriptive, not reduced chi-square: the upstream fit/noise covariance
    and its effective degrees of freedom are unknown. The slope is unweighted
    OLS over all supplied rows, not a forecast or an orbital-decay measurement.
    """
    _positive_integer(num_windows, "num_windows")
    t = _vector(mjd, "MJD")
    r = _vector(residual_us, "Residuals")
    u = _vector(uncertainty_us, "Uncertainties")
    if not (len(t) == len(r) == len(u)) or len(t) < 2:
        raise ValueError("At least two rows with matching times, residuals, and uncertainties are required.")
    if np.any(u <= 0):
        raise ValueError("Uncertainties must be strictly positive.")
    if np.ptp(t) <= 0:
        raise ValueError("At least two distinct observation times are required.")

    start, end = float(t.min()), float(t.max())
    years = (t - start) / 365.25
    centered = years - years.mean()
    slope = float(np.dot(centered, r - r.mean()) / np.dot(centered, centered))
    edges = np.linspace(start, end, num_windows + 1)
    # Internal boundaries belong to the following window; include the final row.
    indices = np.searchsorted(edges[1:-1], t, side="right")
    windows = []
    for index in range(num_windows):
        mask = indices == index
        count = int(mask.sum())
        windows.append({
            "window": index + 1,
            "mjd_start": float(edges[index]), "mjd_end": float(edges[index + 1]),
            "n_obs": count,
            "rms_us": _rms(r[mask]) if count else None,
            "std_us": float(np.std(r[mask])) if count else None,
            "mean_us": float(np.mean(r[mask])) if count else None,
            "chi2_per_observation": float(np.mean((r[mask] / u[mask])**2)) if count else None,
        })
    return {
        "N_total": len(t), "span_days": end - start,
        "span_years": (end - start) / 365.25,
        "overall_rms_us": _rms(r), "overall_std_us": float(np.std(r)),
        "overall_mean_us": float(np.mean(r)), "drift_slope_us_yr": slope,
        "chi2_per_observation": float(np.mean((r / u)**2)),
        "windows": windows,
        "analysis_kind": "post-fit residual inspection",
        "note": "Descriptive statistics of supplied residuals; no independent timing fit or forecast.",
    }


def _read_table(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Residual file not found: {path}")
    comments = []
    first_data = None
    with path.open() as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                comments.append(stripped.lstrip("# "))
            else:
                first_data = stripped
                break
    if first_data is None:
        raise ValueError("Residual file contains no data rows.")
    sep = "," if "," in first_data else r"\s+"
    try:
        frame = pd.read_csv(path, sep=sep, comment="#", header=None)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Cannot parse residual table: {exc}") from exc
    return frame, comments


def linear_parameter_diagnostic(residual, uncertainty, derivative, nuisance, rcond=1e-12):
    """Formal white-noise sensitivity after projecting out nuisance derivatives.

    Input derivatives must be model-phase derivatives. The fitted increment
    minimizes ||(residual + derivative * increment + nuisance * b)/uncertainty||.
    An increment about an existing fit is not an absolute parameter measurement.
    """
    y = _vector(residual, "Residuals")
    sigma = _vector(uncertainty, "Uncertainties")
    derivative = _vector(derivative, "Parameter derivative")
    nuisance = np.asarray(nuisance, dtype=float)
    if len(y) < 2 or not (len(y) == len(sigma) == len(derivative)):
        raise ValueError("Diagnostic arrays must have matching lengths of at least two.")
    if nuisance.ndim != 2 or nuisance.shape[0] != len(y) or not np.all(np.isfinite(nuisance)):
        raise ValueError("Nuisance derivatives must be a finite matrix with one row per observation.")
    if np.any(sigma <= 0) or not np.isfinite(rcond) or not 0 < rcond < 1:
        raise ValueError("Positive uncertainties and 0 < rcond < 1 are required.")
    d = derivative / sigma
    d_norm = np.linalg.norm(d)
    if not np.isfinite(d_norm) or d_norm == 0:
        raise ValueError("Parameter derivative has no finite nonzero sensitivity.")
    # Normalize the target too, so rank tests are invariant to parameter units.
    target = d / d_norm
    matrix = nuisance / sigma[:, None]
    norms = np.linalg.norm(matrix, axis=0)
    if not np.all(np.isfinite(norms)):
        raise ValueError("Weighted nuisance derivatives overflowed.")
    matrix = matrix[:, norms > 0] / norms[norms > 0]
    rank = 0
    if matrix.shape[1]:
        coefficients, _, rank, _ = lstsq(matrix, target, cond=rcond, lapack_driver="gelsd")
        perpendicular = target - matrix @ coefficients
    else:
        perpendicular = target
    remaining = np.linalg.norm(perpendicular)
    identifiable = bool(remaining > rcond)
    increment = None
    marginal_sigma = None
    if identifiable:
        marginal_sigma = float(1 / (d_norm * remaining))
        increment = float(-np.dot(perpendicular, y / sigma) / (d_norm * remaining**2))
    return {
        "identifiable": identifiable,
        "conditional_sigma": float(1 / d_norm),
        "marginal_sigma": marginal_sigma,
        "increment_about_reference_fit": increment,
        "nuisance_rank": int(rank), "nuisance_columns": int(nuisance.shape[1]),
        "relative_svd_cutoff": rcond,
        "remaining_derivative_norm_fraction": float(remaining),
        "note": "Linearized white-noise diagnostic; excludes systematics and correlated noise. "
                "The absolute parameter value requires the original fit's reference point.",
    }


class GenericPulsarDataEvaluator:
    """Inspect numeric residual tables, not raw .tim/TOA files.

    format='standard': MJD, residual, uncertainty (extra columns ignored).
    format='nanograv': MJD, frequency, residual, uncertainty (extras ignored).
    format='auto': accept exactly three columns, or the explicit NANOGrav
    comment header. Ambiguous tables require the caller to choose a format.
    """

    @staticmethod
    def evaluate(file_path, num_windows=10, unit="us", *, format="auto"):
        if unit not in UNIT_TO_US:
            raise ValueError("unit must be one of: us, ns, s.")
        if format not in {"auto", "standard", "nanograv"}:
            raise ValueError("format must be auto, standard, or nanograv.")
        frame, comments = _read_table(file_path)
        if format == "auto":
            header = " ".join(comments).lower()
            if "mjd frequency(mhz) residual(us) uncertainty(us)" in header:
                format = "nanograv"
                if unit != "us":
                    raise ValueError("NANOGrav header specifies us; use unit='us' or choose an explicit format.")
            elif frame.shape[1] == 3:
                format = "standard"
            else:
                raise ValueError("Ambiguous residual columns. Specify format='standard' or 'nanograv'.")
        columns = [0, 1, 2] if format == "standard" else [0, 2, 3]
        if frame.shape[1] <= max(columns):
            raise ValueError(f"Not enough columns for {format} format.")
        try:
            t, r, u = frame.iloc[:, columns].to_numpy(dtype=float).T
        except (TypeError, ValueError) as exc:
            raise ValueError("Expected numeric MJD, residual, and uncertainty columns; raw TOAs are unsupported.") from exc
        result = inspect_residuals(t, r * UNIT_TO_US[unit], u * UNIT_TO_US[unit], num_windows)
        result["input_format"] = format
        return result


class ZenodoResidualEvaluator:
    """Inspect the Archibald 2018 phase-residual CSV (Zenodo 1241122)."""

    @staticmethod
    def evaluate(csv_path, num_windows=10, *, include_sep_diagnostic=False, backend="all"):
        frame, comments = _read_table(csv_path)
        headers = [line for line in comments if line.startswith("MJD")]
        labels = [part.strip() for part in headers[0].split(",")] if headers else []
        if len(labels) != frame.shape[1] or labels[:3] != ["MJD (approx)", "residual", "uncertainty (phase)"]:
            raise ValueError("Expected the annotated Archibald phase-residual CSV; use the custom evaluator for other tables.")
        try:
            data = frame.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Archibald residuals and derivatives must be numeric.") from exc
        if backend not in ("all", "arecibo", "gbt", "wsrt"):
            raise ValueError("backend must be all, arecibo, gbt or wsrt; this release contains no Nancay data.")
        if backend != "all":
            # In Zenodo 1241122's .tim release, -tel AO rows have both jump
            # indicators zero. Resolve named columns, never positional guesses.
            names = ["j_WSRT_1400", "j_GBT_1400"]
            if any(labels.count(name) != 1 for name in names):
                raise ValueError("Backend selection requires named WSRT and GBT jump columns.")
            jumps = data[:, [labels.index(name) for name in names]]
            if not np.all(np.isin(jumps, [0., 1.])) or np.any(jumps.sum(axis=1) > 1):
                raise ValueError("Expected mutually exclusive binary backend jump indicators.")
            expected = {"arecibo": [0., 0.], "gbt": [0., 1.], "wsrt": [1., 0.]}[backend]
            data = data[np.all(jumps == expected, axis=1)]
        phase_to_us = 1e6 / ARCHIBALD_SPIN_FREQUENCY_HZ
        result = inspect_residuals(data[:, 0], data[:, 1] * phase_to_us,
                                   data[:, 2] * phase_to_us, num_windows)
        result["published_sep"] = dict(ARCHIBALD_SEP)
        result["backend"] = backend
        result["note"] = "Archibald et al. (2018) post-fit residuals and derivatives. " \
                         "Their residual scatter is not validation of this package's formula estimates."
        if include_sep_diagnostic:
            if labels.count("delta") != 1 or labels.index("delta") < 3:
                raise ValueError("Exactly one delta derivative column is required.")
            index = labels.index("delta")
            others = [i for i in range(3, data.shape[1]) if i != index]
            result["sep_diagnostic"] = linear_parameter_diagnostic(
                data[:, 1], data[:, 2], data[:, index], data[:, others])
        return result


# Preserve the original import path/name, while documenting its actual scope.
ZenodoWalkForwardEvaluator = ZenodoResidualEvaluator
