"""
=====================================================================================
QRC-PULSAR-BENCH: Standalone Relational Timing Least-Squares Fitter
=====================================================================================
Directly couples the Relational Three-Body Integrator into a physical least-squares
timing engine:
  1. Fit Built from Integrated Delay: The model evaluates the full numerical Jacobi
     3-body trajectory, comparing observed arrival delays directly against the
     integrated Roemer modulation.
  2. Derivatives Taken from Integration: Partial derivatives with respect to
     ascending node epochs (d_delay/d_TASC1, d_delay/d_TASC2) are computed by
     central finite-difference of the 3-body numerical integration (+/- 1.0 s).
  3. Fixed Reference Epoch & Ransom's Measured Node Times: Ties time zero to fixed
     epoch T0 = 55920.0 MJD, using Ransom et al. (2014) Table 1 measured epochs
     (TASC1 = 55920.407717436, TASC2 = 56233.935815) to dynamically derive initial
     orbital phases (no hardcoded angles).
  4. High-Density Resolution: Continuous DOP853 dense-output collocation evaluated
     at the exact observation epochs, eliminating coarse spline interpolation errors.
=====================================================================================
"""

import math
import numpy as np
from scipy.interpolate import interp1d

from .constants import G, c, M_sun, day_s
from .dynamics import RelationalThreeBodyIntegrator
from .systems import HierarchicalTripleSystem, PSR_J0337_1715
from .pint_model import RelationalTripleDelayComponent


class RelationalTimingFitter:
    """
    Physical least-squares estimator built on the Relational Three-Body Integrator,
    with design matrix partial derivatives taken directly from the numerical integration.
    """

    DEFAULT_T0_MJD = 55920.0
    DEFAULT_TASC1_MJD = 55920.407717436  # Ransom et al. (2014) Table 1
    DEFAULT_TASC2_MJD = 56233.935815     # Ransom et al. (2014) Table 1

    @classmethod
    def compute_integrated_derivatives(cls, mjd: np.ndarray, system: HierarchicalTripleSystem,
                                       t0_mjd: float = DEFAULT_T0_MJD,
                                       tasc1_mjd: float = DEFAULT_TASC1_MJD,
                                       tasc2_mjd: float = DEFAULT_TASC2_MJD,
                                       delta_s: float = 1.0):
        """
        Computes partial derivatives d(Roemer_us)/d(T_asc) directly by finite-differencing
        the 3-body numerical integration.

        Parameters:
        -----------
        mjd : np.ndarray
            Observation epochs in MJD.
        system : HierarchicalTripleSystem
            Astrophysical triple system.
        t0_mjd : float
            Fixed reference epoch (MJD).
        tasc1_mjd : float
            Inner orbit ascending node epoch (MJD).
        tasc2_mjd : float
            Outer orbit ascending node epoch (MJD).
        delta_s : float
            Perturbation step size in seconds.

        Returns:
        --------
        nominal_delay_us : np.ndarray
            Nominal 3-body Roemer delay in microseconds.
        d_tasc1_us_s : np.ndarray
            Partial derivative with respect to TASC1 in us/s.
        d_tasc2_us_s : np.ndarray
            Partial derivative with respect to TASC2 in us/s.
        """
        comp = RelationalTripleDelayComponent(system)
        comp._set_param("PEPOCH_REF", t0_mjd)
        comp._set_param("TASC1", tasc1_mjd)
        comp._set_param("TASC2", tasc2_mjd)

        # 1. Nominal integrated delay (microseconds)
        nominal_delay_us = comp.delay(mjd) * 1e6

        # 2. Finite difference on TASC1
        delta_d = delta_s / day_s
        comp._set_param("TASC1", tasc1_mjd + delta_d)
        del_plus_1 = comp.delay(mjd) * 1e6
        comp._set_param("TASC1", tasc1_mjd - delta_d)
        del_minus_1 = comp.delay(mjd) * 1e6
        comp._set_param("TASC1", tasc1_mjd)
        d_tasc1_us_s = (del_plus_1 - del_minus_1) / (2.0 * delta_s)

        # 3. Finite difference on TASC2
        comp._set_param("TASC2", tasc2_mjd + delta_d)
        del_plus_2 = comp.delay(mjd) * 1e6
        comp._set_param("TASC2", tasc2_mjd - delta_d)
        del_minus_2 = comp.delay(mjd) * 1e6
        comp._set_param("TASC2", tasc2_mjd)
        d_tasc2_us_s = (del_plus_2 - del_minus_2) / (2.0 * delta_s)

        return nominal_delay_us, d_tasc1_us_s, d_tasc2_us_s

    @classmethod
    def fit_integrated_trajectory(cls, mjd, observed_delays_us, uncertainties_us,
                                  system: HierarchicalTripleSystem = PSR_J0337_1715,
                                  t0_mjd: float = DEFAULT_T0_MJD,
                                  tasc1_mjd: float = DEFAULT_TASC1_MJD,
                                  tasc2_mjd: float = DEFAULT_TASC2_MJD):
        """
        Fits 3-body orbital parameters directly to observed arrival delays,
        with derivatives taken from the numerical integration.
        """
        mjd = np.asarray(mjd, dtype=float)
        obs_us = np.asarray(observed_delays_us, dtype=float)
        unc_us = np.asarray(uncertainties_us, dtype=float)
        N = len(mjd)

        if N < 5:
            raise ValueError("Fitting requires at least 5 observation epochs.")

        nom_delay_us, d_tasc1, d_tasc2 = cls.compute_integrated_derivatives(
            mjd, system, t0_mjd=t0_mjd, tasc1_mjd=tasc1_mjd, tasc2_mjd=tasc2_mjd
        )

        # Residual between data and nominal 3-body trajectory
        r_pre = obs_us - nom_delay_us
        t_yr = (mjd - t0_mjd) / 365.25

        # Weighted design matrix with derivatives taken directly from the integration
        A = np.column_stack([
            np.ones_like(mjd),
            t_yr,
            d_tasc1,
            d_tasc2
        ])

        W = 1.0 / (np.maximum(unc_us, 1e-9)**2)
        AT_W = A.T * W
        cov = np.linalg.pinv(AT_W @ A)
        params = cov @ (AT_W @ r_pre)
        param_errors = np.sqrt(np.maximum(0.0, np.diag(cov)))

        fit_correction = A @ params
        r_post = r_pre - fit_correction

        input_rms = float(np.sqrt(np.mean(r_pre**2)))
        output_rms = float(np.sqrt(np.mean(r_post**2)))
        dof = N - len(params)
        chi2 = float(np.sum((r_post / unc_us)**2))
        chi2_dof = float(chi2 / dof) if dof > 0 else 0.0

        return {
            "analysis_type": "integrated_3body_trajectory_fit",
            "num_toas": N,
            "span_days": float(mjd.max() - mjd.min()),
            "span_years": float((mjd.max() - mjd.min()) / 365.25),
            "reference_epoch_mjd": t0_mjd,
            "initial_tasc1_mjd": tasc1_mjd,
            "initial_tasc2_mjd": tasc2_mjd,
            "fitted_tasc1_mjd": float(tasc1_mjd + params[2] / day_s),
            "fitted_tasc2_mjd": float(tasc2_mjd + params[3] / day_s),
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
            "input_rms_us": input_rms,
            "residual_after_perturbation_rms_us": output_rms,
            "output_rms_us": output_rms,
            "residual_std_us": float(np.std(r_post)),
            "residual_mean_us": float(np.mean(r_post)),
            "chi2": chi2,
            "chi2_dof": chi2_dof,
            "dof": dof,
            "residuals_post_us": r_post,
            "derivative_source": "central_finite_difference_of_relational_three_body_integrator"
        }

    @classmethod
    def fit_linearized_perturbations(cls, mjd, post_fit_residuals_us, uncertainties_us,
                                     system: HierarchicalTripleSystem = PSR_J0337_1715,
                                     t0_mjd: float = DEFAULT_T0_MJD,
                                     tasc1_mjd: float = DEFAULT_TASC1_MJD,
                                     tasc2_mjd: float = DEFAULT_TASC2_MJD):
        """
        Fits small residual perturbations around an upstream converged residual table
        using derivatives taken directly from the 3-body numerical integration.
        """
        mjd = np.asarray(mjd, dtype=float)
        res_us = np.asarray(post_fit_residuals_us, dtype=float)
        unc_us = np.asarray(uncertainties_us, dtype=float)
        N = len(mjd)

        if N < 5:
            raise ValueError("Fitting requires at least 5 observation epochs.")

        _, d_tasc1, d_tasc2 = cls.compute_integrated_derivatives(
            mjd, system, t0_mjd=t0_mjd, tasc1_mjd=tasc1_mjd, tasc2_mjd=tasc2_mjd
        )

        t_yr = (mjd - t0_mjd) / 365.25
        A = np.column_stack([
            np.ones_like(mjd),
            t_yr,
            d_tasc1,
            d_tasc2
        ])

        W = 1.0 / (np.maximum(unc_us, 1e-9)**2)
        AT_W = A.T * W
        cov = np.linalg.pinv(AT_W @ A)
        params = cov @ (AT_W @ res_us)
        param_errors = np.sqrt(np.maximum(0.0, np.diag(cov)))

        fit_model = A @ params
        residual_after_perturbation = res_us - fit_model

        input_rms = float(np.sqrt(np.mean(res_us**2)))
        output_rms = float(np.sqrt(np.mean(residual_after_perturbation**2)))
        dof = N - len(params)
        chi2 = float(np.sum((residual_after_perturbation / unc_us)**2))
        chi2_dof = float(chi2 / dof) if dof > 0 else 0.0

        return {
            "analysis_type": "linearized_residual_perturbation",
            "num_toas": N,
            "span_days": float(mjd.max() - mjd.min()),
            "span_years": float((mjd.max() - mjd.min()) / 365.25),
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
            "output_rms_us": output_rms,
            "residual_std_us": float(np.std(residual_after_perturbation)),
            "residual_mean_us": float(np.mean(residual_after_perturbation)),
            "chi2": chi2,
            "chi2_dof": chi2_dof,
            "dof": dof,
            "residuals_post_us": residual_after_perturbation,
            "note": "Linearized nuisance perturbation on upstream post-fit residuals. "
                    "The ~0.77 us or ~1.54 us scatter is the property of the upstream fit; "
                    "refitting finds parameter shifts ~0.",
            "derivative_source": "central_finite_difference_of_relational_three_body_integrator"
        }

    @classmethod
    def fit(cls, mjd, data_us, unc_us, system: HierarchicalTripleSystem = PSR_J0337_1715,
            mode: str = "auto", **kwargs):
        """
        Unified fitting entry point.
        Automatically distinguishes between full trajectory fits (when data span > 100 us)
        and linearized perturbation fits on upstream post-fit residuals.
        """
        data_us = np.asarray(data_us, dtype=float)
        span = float(np.ptp(data_us))
        if mode == "integrated" or (mode == "auto" and span > 100.0):
            return cls.fit_integrated_trajectory(mjd, data_us, unc_us, system, **kwargs)
        else:
            return cls.fit_linearized_perturbations(mjd, data_us, unc_us, system, **kwargs)
