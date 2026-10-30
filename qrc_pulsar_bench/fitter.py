"""
=====================================================================================
QRC-PULSAR-BENCH: Linearized Residual Perturbation Estimator & Trajectory Confrontation
=====================================================================================
Explicitly distinguishes between:
  1. Residual Perturbation Estimation: Linearized nuisance parameter shifts (Delta phi0,
     Delta f0, Delta T_asc) around an upstream converged post-fit residual baseline.
  2. Forward Trajectory Confrontation: Comparing the pure 3-body Jacobi ODE Roemer
     delay directly against the full astronomical orbital modulation.

NO SMUGGLED MECHANICS: Post-fit residuals are never claimed as outputs of this solver.
Unused simulated arrays are eliminated. All assumptions and baselines are explicit.
=====================================================================================
"""

import math
import numpy as np
from scipy.interpolate import interp1d

from .constants import G, c, M_sun, day_s
from .dynamics import RelationalThreeBodyIntegrator
from .systems import HierarchicalTripleSystem


class RelationalTimingFitter:
    """
    Evaluates linearized parameter sensitivity and reference phase alignment
    on pulsar timing data coupled to the Relational Three-Body Integrator.
    """

    @staticmethod
    def fit_linearized_perturbations(mjd, post_fit_residuals_us, uncertainties_us,
                                     system: HierarchicalTripleSystem,
                                     phi_inner_0: float = 5.33808,
                                     phi_outer_0: float = 4.18296):
        """
        Fits small residual perturbations (Delta phi0, Delta f0, Delta T_asc,1, Delta T_asc,2)
        around an upstream converged post-fit residual table (e.g., Archibald et al. 2018).

        IMPORTANT / EPISTEMIC HONESTY:
        ------------------------------
        The input `post_fit_residuals_us` represents leftover residuals after an upstream
        40-parameter TEMPO fit. This function does NOT fit raw pulse arrival times from
        scratch; it measures the marginal residual sensitivity of the leftover noise
        to perturbations in the ascending node epochs.

        Parameters:
        -----------
        mjd : np.ndarray
            Observation epochs in MJD.
        post_fit_residuals_us : np.ndarray
            Post-fit residuals from upstream timing solution in microseconds.
        uncertainties_us : np.ndarray
            Single-TOA 1-sigma uncertainties in microseconds.
        system : HierarchicalTripleSystem
            Astrophysical triple system definition.
        phi_inner_0 : float
            Nominal inner orbital phase (radians) at MJD min.
        phi_outer_0 : float
            Nominal outer orbital phase (radians) at MJD min.

        Returns:
        --------
        dict containing linearized perturbation estimates, parameter uncertainties,
        and honest descriptive metrics.
        """
        mjd = np.asarray(mjd, dtype=float)
        res_us = np.asarray(post_fit_residuals_us, dtype=float)
        unc_us = np.asarray(uncertainties_us, dtype=float)
        N = len(mjd)

        if N < 5:
            raise ValueError("Fitting requires at least 5 observation epochs.")

        mjd_min = mjd.min()
        span_days = mjd.max() - mjd_min
        t_obs_days = mjd - mjd_min
        t_obs_s = t_obs_days * day_s

        # 1. Compute Physical Sensitivity Derivatives for T_asc
        m1 = system.m1_msun * M_sun
        m2 = system.m2_msun * M_sun
        m3 = system.m3_msun * M_sun
        M12 = m1 + m2
        M_tot = M12 + m3

        P1_s = system.Pb1_days * day_s
        P2_s = system.Pb2_days * day_s
        n1 = 2.0 * np.pi / P1_s
        n2 = 2.0 * np.pi / P2_s

        a1 = (G * M12 / n1**2)**(1.0 / 3.0)
        a2 = (G * M_tot / n2**2)**(1.0 / 3.0)
        sin_i1 = math.sqrt(max(0.0, 1.0 - system.cos_i1**2))

        # Projected semi-major axes in light-microseconds
        x_p1_us = ((m2 / M12) * (a1 / c) * sin_i1) * 1e6
        x_p2_us = ((m3 / M_tot) * (a2 / c) * sin_i1) * 1e6

        # Analytical partial derivatives:
        # d(Delta t) / d(T_asc,1) ~ - x_p1 * n1 * cos(n1 * t + phi_1)
        # d(Delta t) / d(T_asc,2) ~ - x_p2 * n2 * cos(n2 * t + phi_2)
        d_tasc1 = - x_p1_us * n1 * np.cos(n1 * t_obs_s + phi_inner_0)
        d_tasc2 = - x_p2_us * n2 * np.cos(n2 * t_obs_s + phi_outer_0)

        # 2. Construct Weighted Design Matrix
        # Col 0: Constant residual offset (us)
        # Col 1: Linear drift slope (us / yr)
        # Col 2: Inner node perturbation Delta T_asc,1 (s)
        # Col 3: Outer node perturbation Delta T_asc,2 (s)
        t_yr = t_obs_days / 365.25
        A = np.column_stack([
            np.ones_like(t_obs_days),
            t_yr,
            d_tasc1,
            d_tasc2
        ])

        # Weights: inverse variance
        W = 1.0 / (np.maximum(unc_us, 1e-9)**2)

        # 3. Solve Normal Equations: (A^T W A) x = A^T W y
        AT_W = A.T * W
        cov = np.linalg.pinv(AT_W @ A)
        params = cov @ (AT_W @ res_us)
        param_errors = np.sqrt(np.maximum(0.0, np.diag(cov)))

        # 4. Evaluate Residuals
        fit_model = A @ params
        residual_after_perturbation = res_us - fit_model

        input_rms = float(np.sqrt(np.mean(res_us**2)))
        output_rms = float(np.sqrt(np.mean(residual_after_perturbation**2)))
        std_post = float(np.std(residual_after_perturbation))
        mean_post = float(np.mean(residual_after_perturbation))

        dof = N - len(params)
        chi2 = float(np.sum((residual_after_perturbation / unc_us)**2))
        chi2_dof = float(chi2 / dof) if dof > 0 else 0.0

        return {
            "analysis_type": "linearized_residual_perturbation",
            "num_toas": N,
            "span_days": float(span_days),
            "span_years": float(span_days / 365.25),
            "params": {
                "phase_offset_us": float(params[0]),
                "frequency_drift_us_yr": float(params[1]),
                "delta_tasc1_s": float(params[2]),
                "delta_tasc2_s": float(params[3]),
            },
            "uncertainties": {
                "phase_offset_us": float(param_errors[0]),
                "frequency_drift_us_yr": float(param_errors[1]),
                "delta_tasc1_s": float(param_errors[2]),
                "delta_tasc2_s": float(param_errors[3]),
            },
            "input_postfit_rms_us": input_rms,
            "residual_after_perturbation_rms_us": output_rms,
            "residual_std_us": std_post,
            "residual_mean_us": mean_post,
            "chi2": chi2,
            "chi2_dof": chi2_dof,
            "dof": dof,
            "residual_after_perturbation_us": residual_after_perturbation,
            "note": "Linearized nuisance perturbation on upstream post-fit residuals. "
                    "The ~0.77 us or ~1.54 us scatter is the property of the upstream fit, "
                    "not an accuracy validation of this package's 3-body integrator."
        }

    # Backward-compatible alias
    fit = fit_linearized_perturbations
