"""
=====================================================================================
Relational Timing & Post-Keplerian Benchmarking Engine
=====================================================================================
Computational algorithms:
1. PostKeplerianSolver: Standard textbook relativistic post-Keplerian binary formulas
   (1PN to 2.5PN: periastron advance, Peters-Mathews GW orbital decay, Einstein delay,
   Shapiro delay).
2. RelationalTripleEngine: Standard hierarchical 3-body tidal multipole expansion
   (Harrington 1968, Ford et al. 2000, Archibald et al. 2018) providing analytical
   scaling estimates for quadrupole, octupole, and hexadecapole tidal terms.
3. ZenodoWalkForwardEvaluator: Evaluates 27,110 published post-fit radio residuals
   from Archibald et al. (Nature 2018) for PSR J0337+1715.
4. GenericPulsarDataEvaluator: Residual inspection tool across sequential time windows
   for user-supplied or public pulsar data (e.g., NANOGrav data releases).
5. CircumbinarySolver: 3-body secular nodal precession period from Keplerian torque.
=====================================================================================
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from .constants import G, c, M_sun, AU, day_s, year_s
from .systems import BinaryPulsarSystem, HierarchicalTripleSystem, CircumbinaryExoplanetSystem

class PostKeplerianSolver:
    """Standard Post-Keplerian relativistic binary solver."""

    @staticmethod
    def solve(system: BinaryPulsarSystem) -> Dict[str, Any]:
        m1 = system.m1_msun * M_sun
        m2 = system.m2_msun * M_sun
        M = m1 + m2
        Pb = system.Pb_days * day_s
        n = 2.0 * np.pi / Pb
        e = system.eccentricity

        # 1. 1PN Relativistic Periastron Advance (Robertson 1938, Damour-Deruelle 1986)
        om_dot_rad_s = 3.0 * (n**(5.0 / 3.0)) * ((G * M / c**3)**(2.0 / 3.0)) / (1.0 - e**2)
        om_dot_deg_yr = om_dot_rad_s * (180.0 / np.pi) * year_s

        # 2. 2.5PN Gravitational Radiation Reaction (Peters & Mathews 1963)
        fe = (1.0 + (73.0 / 24.0) * e**2 + (37.0 / 96.0) * e**4) / ((1.0 - e**2)**(7.0 / 2.0))
        eta = (m1 * m2) / (M**2)
        Pdot_b = - (192.0 * np.pi / 5.0) * ((n * G * M / c**3)**(5.0 / 3.0)) * eta * fe

        # 3. Einstein Delay Gamma (Gravitational Redshift + Transverse Doppler)
        gamma_s = e * ((Pb / (2.0 * np.pi))**(1.0 / 3.0)) * ((G / c**3)**(2.0 / 3.0)) * (m2 * (m1 + 2.0 * m2) / (M**(4.0 / 3.0)))

        # 4. Shapiro Delay Range (r) and Shape (s)
        r_s = G * m2 / c**3
        s = system.sin_i

        # 5. Geodetic Spin-Orbit Precession (Barker & O'Connell 1975)
        omega_geod_rad_s = ((n * G * M / c**3)**(2.0 / 3.0)) * n * (m2 * (4.0 * m1 + 3.0 * m2) / (2.0 * M**2)) / (1.0 - e**2)
        omega_geod_deg_yr = omega_geod_rad_s * (180.0 / np.pi) * year_s

        result = {
            "system": system.name,
            "om_dot_pred_deg_yr": om_dot_deg_yr,
            "Pdot_b_pred": Pdot_b,
            "gamma_pred_ms": gamma_s * 1e3,
            "r_pred_us": r_s * 1e6,
            "s_pred": s,
            "omega_geod_deg_yr": omega_geod_deg_yr,
            "targets": {
                "om_dot_obs_deg_yr": system.om_dot_obs_deg_yr,
                "Pdot_b_obs": system.Pdot_b_obs,
                "Pdot_b_int_obs": system.Pdot_b_int_obs,
                "gamma_obs_ms": system.gamma_obs_ms,
                "r_obs_us": system.r_obs_us,
                "s_obs": system.s_obs
            }
        }

        # Compute percentage agreements for available empirical targets
        agreements = {}
        if system.om_dot_obs_deg_yr is not None:
            agreements["om_dot"] = (1.0 - abs(om_dot_deg_yr - system.om_dot_obs_deg_yr) / system.om_dot_obs_deg_yr) * 100.0
        if system.Pdot_b_obs is not None:
            agreements["Pdot_b"] = (1.0 - abs(Pdot_b - system.Pdot_b_obs) / abs(system.Pdot_b_obs)) * 100.0
        if system.Pdot_b_int_obs is not None:
            agreements["Pdot_b_int"] = (1.0 - abs(Pdot_b - system.Pdot_b_int_obs) / abs(system.Pdot_b_int_obs)) * 100.0
        if system.gamma_obs_ms is not None:
            agreements["gamma"] = (1.0 - abs(gamma_s * 1e3 - system.gamma_obs_ms) / system.gamma_obs_ms) * 100.0
        if system.r_obs_us is not None:
            agreements["r"] = (1.0 - abs(r_s * 1e6 - system.r_obs_us) / system.r_obs_us) * 100.0
        result["agreements_pct"] = agreements

        return result

class RelationalTripleEngine:
    """
    Standard hierarchical 3-body tidal multipole scaling estimates (Harrington 1968,
    Ford et al. 2000, Archibald et al. 2018).
    Note: Evaluates closed-form semi-amplitudes for circular hierarchical triples;
    does not numerically integrate the 3-body differential equations of motion.
    """

    @staticmethod
    def solve(triple: HierarchicalTripleSystem) -> Dict[str, Any]:
        m1 = triple.m1_msun * M_sun
        m2 = triple.m2_msun * M_sun
        m3 = triple.m3_msun * M_sun
        M_in = m1 + m2
        M_tot = M_in + m3

        P1 = triple.Pb1_days * day_s
        P2 = triple.Pb2_days * day_s
        n1 = 2.0 * np.pi / P1
        n2 = 2.0 * np.pi / P2

        a1 = (G * M_in / n1**2)**(1.0 / 3.0)
        a2 = (G * M_tot / n2**2)**(1.0 / 3.0)
        epsilon = a1 / a2

        # Mass asymmetry factor and symmetric mass ratio
        delta_m = (m1 - m2) / (m1 + m2)
        eta_in = (m1 * m2) / (M_in**2)

        sin_i1 = np.sqrt(1.0 - triple.cos_i1**2)
        x_p_proj = (m2 / M_in) * (a1 / c) * sin_i1

        # 1. Quadrupole Mode (l=2)
        delta_r_quad = (m2 / M_in) * a1 * (m3 / M_tot) * (n2 / n1)**2
        A_quad_us = (delta_r_quad / c) * sin_i1 * 1e6

        # 2. Octupole Mode (l=3)
        delta_r_oct = delta_r_quad * epsilon * delta_m
        A_oct_ns = (delta_r_oct / c) * sin_i1 * 1e9

        # 3. Hexadecapole Mode (l=4)
        M4 = 1.0 - 3.0 * eta_in
        delta_r_hex = delta_r_quad * (epsilon**2) * M4
        A_hex_ns = (delta_r_hex / c) * sin_i1 * 1e9

        # 4. 2PN Relativistic Cross-term scaling estimate
        v_c_sq = (n1 * a1 / c)**2
        delta_t_2pn_ps = x_p_proj * (v_c_sq**2) * 1e12

        # 5. Peters-Mathews 2.5PN Radiation Reaction
        v_c_cubed = n1 * G * M_in / c**3
        Pdot_b1 = - (192.0 * np.pi / 5.0) * (v_c_cubed)**(5.0 / 3.0) * eta_in

        return {
            "system": triple.name,
            "epsilon_hierarchy": epsilon,
            "mass_asymmetry": delta_m,
            "A_quad_us": A_quad_us,
            "A_oct_ns": A_oct_ns,
            "A_hex_ns": A_hex_ns,
            "delta_t_2pn_ps": delta_t_2pn_ps,
            "Pdot_b1": Pdot_b1,
            "delta_sep_published": triple.delta_sep_published,
            "delta_sep_uncertainty": triple.delta_sep_uncertainty
        }

class ZenodoWalkForwardEvaluator:
    """
    Evaluates published post-fit residuals of Archibald et al. (Nature 2018) for PSR J0337+1715.
    Dataset: Zenodo DOI: 10.5281/zenodo.1241121 (27,110 unbinned residuals).
    """

    @staticmethod
    def evaluate(csv_path: str, num_windows: int = 10) -> Dict[str, Any]:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Zenodo CSV file not found at {csv_path}")

        df = pd.read_csv(csv_path, comment='#', header=None)
        N_total = len(df)

        mjd = df[0].values.astype(float)
        res_phase = df[1].values.astype(float)
        unc_phase = df[2].values.astype(float)

        # Pulsar spin frequency from Archibald et al. 2018
        P0_s = 1.0 / 365.953339305
        res_us = res_phase * P0_s * 1e6
        unc_us = unc_phase * P0_s * 1e6

        mjd_min, mjd_max = mjd.min(), mjd.max()
        span_days = mjd_max - mjd_min
        span_years = span_days / 365.25

        # Sequential Forecast Windows
        win_size = span_days / num_windows
        windows = []
        for w in range(num_windows):
            w_start = mjd_min + w * win_size
            w_end = mjd_min + (w + 1) * win_size
            mask = (mjd >= w_start) & (mjd <= w_end) if w == num_windows - 1 else (mjd >= w_start) & (mjd < w_end)
            w_res = res_us[mask]
            w_unc = unc_us[mask]
            windows.append({
                "window": w + 1,
                "mjd_start": w_start,
                "mjd_end": w_end,
                "n_obs": len(w_res),
                "rms_us": float(np.std(w_res)),
                "mean_drift_us": float(np.mean(w_res)),
                "chi2_dof": float(np.sum((w_res / w_unc)**2) / len(w_res))
            })

        # Global Linear Drift Rate
        t_years = (mjd - mjd_min) / 365.25
        drift_slope_us_yr, _ = np.polyfit(t_years, res_us, 1)

        return {
            "N_total": N_total,
            "span_days": span_days,
            "span_years": span_years,
            "overall_rms_us": float(np.std(res_us)),
            "overall_mean_us": float(np.mean(res_us)),
            "drift_slope_us_yr": float(drift_slope_us_yr),
            "delta_sep_published": -0.6e-6,
            "delta_sep_uncertainty": 4.9e-6,
            "windows": windows,
            "note": (
                "These are Archibald et al. (2018) post-fit residuals from a full 40-parameter TEMPO fit. "
                "Because Delta was already included in the original fit, inspecting these residuals verifies the "
                "fit quality but does not constitute an independent measurement of Delta."
            )
        }

class GenericPulsarDataEvaluator:
    """Evaluates arbitrary pulsar timing datasets (CSV, ASCII, or NANOGrav format)."""

    @staticmethod
    def evaluate(file_path: str, num_windows: int = 10, unit: str = "us") -> Dict[str, Any]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at {file_path}")

        try:
            df = pd.read_csv(file_path, comment='#', header=None)
            if len(df.columns) < 3:
                df = pd.read_csv(file_path, sep=r'\s+', comment='#', header=None)
        except Exception:
            df = pd.read_csv(file_path, sep=r'\s+', comment='#', header=None)

        if df.shape[1] >= 4 and np.nanmedian(df.iloc[:, 1].values.astype(float)) > 100.0:
            mjd = df.iloc[:, 0].values.astype(float)
            res = df.iloc[:, 2].values.astype(float)
            unc = df.iloc[:, 3].values.astype(float)
        else:
            mjd = df.iloc[:, 0].values.astype(float)
            res = df.iloc[:, 1].values.astype(float)
            unc = df.iloc[:, 2].values.astype(float)

        if unit == "s":
            res_us = res * 1e6
            unc_us = unc * 1e6
        elif unit == "ns":
            res_us = res * 1e-3
            unc_us = unc * 1e-3
        else:
            res_us = res
            unc_us = unc

        N_total = len(res_us)
        mjd_min, mjd_max = mjd.min(), mjd.max()
        span_days = mjd_max - mjd_min
        span_years = span_days / 365.25

        win_size = span_days / num_windows
        windows = []
        for w in range(num_windows):
            w_start = mjd_min + w * win_size
            w_end = mjd_min + (w + 1) * win_size
            mask = (mjd >= w_start) & (mjd <= w_end) if w == num_windows - 1 else (mjd >= w_start) & (mjd < w_end)
            w_res = res_us[mask]
            w_unc = unc_us[mask]
            if len(w_res) > 0:
                windows.append({
                    "window": w + 1,
                    "mjd_start": w_start,
                    "mjd_end": w_end,
                    "n_obs": len(w_res),
                    "rms_us": float(np.std(w_res)),
                    "mean_drift_us": float(np.mean(w_res)),
                    "chi2_dof": float(np.sum((w_res / w_unc)**2) / len(w_res))
                })

        t_years = (mjd - mjd_min) / 365.25
        drift_slope_us_yr, _ = np.polyfit(t_years, res_us, 1)

        return {
            "N_total": N_total,
            "span_days": span_days,
            "span_years": span_years,
            "overall_rms_us": float(np.std(res_us)),
            "overall_mean_us": float(np.mean(res_us)),
            "drift_slope_us_yr": float(drift_slope_us_yr),
            "global_chi2_dof": float(np.sum((res_us / unc_us)**2) / N_total),
            "windows": windows
        }

class CircumbinarySolver:
    """Secular 3-body nodal precession for circumbinary exoplanets (Doyle et al. 2011)."""

    @staticmethod
    def solve(system: CircumbinaryExoplanetSystem) -> Dict[str, Any]:
        M_bin = (system.m_starA_msun + system.m_starB_msun) * M_sun
        P_bin = system.P_bin_days * day_s
        P_p = system.P_planet_days * day_s

        n_bin = 2.0 * np.pi / P_bin
        n_p = 2.0 * np.pi / P_p

        a_bin = (G * M_bin / n_bin**2)**(1.0 / 3.0)
        a_p = (G * M_bin / n_p**2)**(1.0 / 3.0)

        axis_ratio_factor = (a_bin / a_p)**2
        mass_ratio_factor = (system.m_starA_msun * system.m_starB_msun) / ((system.m_starA_msun + system.m_starB_msun)**2)

        i_mut = np.radians(0.42)
        ecc_factor = (1.0 + 1.5 * system.e_bin**2) / ((1.0 - system.e_planet**2)**2)

        dOmega_dt = 0.75 * n_p * axis_ratio_factor * mass_ratio_factor * ecc_factor * np.cos(i_mut)
        P_prec_yr_pred = (2.0 * np.pi / dOmega_dt) / year_s

        agreement_pct = (1.0 - abs(P_prec_yr_pred - system.P_nodal_prec_yr_obs) / system.P_nodal_prec_yr_obs) * 100.0

        return {
            "system": system.name,
            "P_nodal_prec_yr_pred": float(P_prec_yr_pred),
            "P_nodal_prec_yr_obs": system.P_nodal_prec_yr_obs,
            "agreement_pct": float(agreement_pct),
            "ttv_semi_amplitude_hours": system.ttv_amplitude_hours
        }

class RelationalTimingEngine:
    """Unified master facade for all pulsar timing benchmarks."""

    @classmethod
    def run_all(cls, zenodo_csv: Optional[str] = None) -> Dict[str, Any]:
        from .systems import PSR_J0337_1715, PSR_J0737_3039, PSR_B1913_16, PSR_J1738_0333, KEPLER_16B

        results = {
            "PSR_J0737-3039": PostKeplerianSolver.solve(PSR_J0737_3039),
            "PSR_B1913+16": PostKeplerianSolver.solve(PSR_B1913_16),
            "PSR_J1738+0333": PostKeplerianSolver.solve(PSR_J1738_0333),
            "PSR_J0337+1715_multipoles": RelationalTripleEngine.solve(PSR_J0337_1715),
            "Kepler-16b": CircumbinarySolver.solve(KEPLER_16B)
        }

        if zenodo_csv and os.path.exists(zenodo_csv):
            results["PSR_J0337+1715_real_data"] = ZenodoWalkForwardEvaluator.evaluate(zenodo_csv)

        return results
