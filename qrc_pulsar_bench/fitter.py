"""
=====================================================================================
QRC-PULSAR-BENCH: Standalone Relational Timing Least-Squares Fitter
=====================================================================================
Performs joint linear and non-linear least-squares fitting of reference phase
anchors (Delta phi0, Delta f0, Delta T_asc,1, Delta T_asc,2) coupled to the
Relational Three-Body Integrator.
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
    Joint least-squares estimator fitting reference phase anchors and
    times of ascending node passage (T_asc) coupled to the relational 3-body ODE.
    """

    @staticmethod
    def fit(mjd, residuals_us, uncertainties_us, system: HierarchicalTripleSystem,
            phi_inner_0: float = 5.33808, phi_outer_0: float = 4.18296,
            num_eval_points: int = 25000):
        """
        Fits reference phase parameters to align relational 3-body dynamics
        with observational radio pulse arrival times.

        Parameters:
        -----------
        mjd : np.ndarray
            Observation epochs in Modified Julian Date.
        residuals_us : np.ndarray
            Timing residuals in microseconds.
        uncertainties_us : np.ndarray
            Single-TOA 1-sigma timing uncertainties in microseconds.
        system : HierarchicalTripleSystem
            Astrophysical triple system definition.
        phi_inner_0 : float
            Nominal inner orbital phase (radians) at MJD min.
        phi_outer_0 : float
            Nominal outer orbital phase (radians) at MJD min.

        Returns:
        --------
        dict containing fit parameters, formal uncertainties, post-fit residuals,
        and goodness-of-fit metrics.
        """
        mjd = np.asarray(mjd, dtype=float)
        res_us = np.asarray(residuals_us, dtype=float)
        unc_us = np.asarray(uncertainties_us, dtype=float)
        N = len(mjd)

        if N < 5:
            raise ValueError("Fitting requires at least 5 observation epochs.")

        mjd_min = mjd.min()
        span_days = mjd.max() - mjd_min
        t_obs_days = mjd - mjd_min
        t_obs_s = t_obs_days * day_s

        # 1. Run Relational Three-Body Numerical Integration across span
        num_inner_orbits = span_days / system.Pb1_days
        sim_result = RelationalThreeBodyIntegrator.integrate(
            system,
            num_inner_orbits=num_inner_orbits,
            num_output_points=num_eval_points,
            rtol=1e-10,
            atol=1e-10,
            inner_angles_rad=(0.0, 0.0, phi_inner_0),
            outer_angles_rad=(0.0, 0.0, phi_outer_0),
            line_of_sight=(0.0, 0.0, 1.0)
        )

        t_sim_days = np.linspace(0.0, span_days, num_eval_points)
        roemer_sim_us = sim_result['roemer_delay_us']

        # Interpolate simulated Roemer delay to observation epochs
        interp_fn = interp1d(t_sim_days, roemer_sim_us, kind='cubic')
        roemer_at_toas_us = interp_fn(t_obs_days)

        # 2. Compute Physical Sensitivity Derivatives
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
        # d(Delta t) / d(T_asc,2) ~ - x_p2 * n2 * cos(n2 * t + phi_2)
        # d(Delta t) / d(T_asc,1) ~ - x_p1 * n1 * cos(n1 * t + phi_1)
        d_tasc1 = - x_p1_us * n1 * np.cos(n1 * t_obs_s + phi_inner_0)
        d_tasc2 = - x_p2_us * n2 * np.cos(n2 * t_obs_s + phi_outer_0)

        # 3. Construct Weighted Design Matrix
        # Col 0: Constant phase offset (us)
        # Col 1: Linear frequency drift rate (us / yr)
        # Col 2: Inner ascending node epoch shift Delta T_asc,1 (s)
        # Col 3: Outer ascending node epoch shift Delta T_asc,2 (s)
        t_yr = t_obs_days / 365.25
        A = np.column_stack([
            np.ones_like(t_obs_days),
            t_yr,
            d_tasc1,
            d_tasc2
        ])

        # Weights: inverse variance
        W = 1.0 / (np.maximum(unc_us, 1e-9)**2)

        # 4. Normal Equations Solution: (A^T W A) x = A^T W y
        AT_W = A.T * W
        cov = np.linalg.pinv(AT_W @ A)
        params = cov @ (AT_W @ res_us)
        param_errors = np.sqrt(np.maximum(0.0, np.diag(cov)))

        # 5. Evaluate Post-Fit Residuals & Metrics
        fit_model = A @ params
        post_fit_res = res_us - fit_model

        rms_pre = float(np.sqrt(np.mean(res_us**2)))
        rms_post = float(np.sqrt(np.mean(post_fit_res**2)))
        std_post = float(np.std(post_fit_res))
        mean_post = float(np.mean(post_fit_res))

        dof = N - len(params)
        chi2 = float(np.sum((post_fit_res / unc_us)**2))
        chi2_dof = float(chi2 / dof) if dof > 0 else 0.0

        return {
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
            "pre_fit_rms_us": rms_pre,
            "post_fit_rms_us": rms_post,
            "post_fit_std_us": std_post,
            "post_fit_mean_us": mean_post,
            "chi2": chi2,
            "chi2_dof": chi2_dof,
            "dof": dof,
            "post_fit_residuals_us": post_fit_res,
            "sim_wall_time_s": sim_result["wall_time_s"],
            "sim_energy_err": sim_result["energy_conservation_err"],
            "sim_com_drift_m": sim_result["com_drift_m"]
        }
