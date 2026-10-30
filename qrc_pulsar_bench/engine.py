"""Post-Keplerian formulas and hierarchical perturbation scale estimates.

These routines do not generate trajectories or a pulsar timing model. Reference
comparisons are numerical differences, not independent tests of gravity.
"""

from typing import Optional

import numpy as np

from .constants import GM_sun, T_sun, c, day_s, year_s
from .systems import BinaryPulsarSystem, HierarchicalTripleSystem, CircumbinaryExoplanetSystem
from .residuals import (
    GenericPulsarDataEvaluator, ZenodoResidualEvaluator, ZenodoWalkForwardEvaluator,
)


def _positive(**values):
    for name, value in values.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")


def _interval(name, value, lower, upper, *, include_upper=True):
    if not np.isfinite(value) or value < lower or (value > upper if include_upper else value >= upper):
        raise ValueError(f"{name} must be in [{lower}, {upper}{']' if include_upper else ')'}.")


def _eccentricity_factor(e):
    return (1 + 73/24 * e**2 + 37/96 * e**4) / (1 - e**2)**3.5


def _percent_difference(prediction, reference):
    if reference is None:
        return None
    if not np.isfinite(reference):
        raise ValueError("Reference values must be finite.")
    return float(100 * abs(prediction - reference) / abs(reference)) if reference != 0 else None


class PostKeplerianSolver:
    """Leading-order GR binary relations; higher-order effects are omitted."""

    @staticmethod
    def solve(system: BinaryPulsarSystem):
        m1, m2 = system.m1_msun, system.m2_msun
        _positive(m1_msun=m1, m2_msun=m2, Pb_days=system.Pb_days)
        _interval("eccentricity", system.eccentricity, 0, 1, include_upper=False)
        _interval("sin_i", system.sin_i, 0, 1)
        _interval("x_p_s", system.x_p_s, 0, np.inf)
        total = m1 + m2
        period = system.Pb_days * day_s
        n = 2 * np.pi / period
        e = system.eccentricity
        eta = m1 * m2 / total**2
        advance = 3 * n**(5/3) * (T_sun * total)**(2/3) / (1 - e**2)
        pdot = -(192 * np.pi / 5) * (n * T_sun * total)**(5/3) * eta * _eccentricity_factor(e)
        gamma = e * n**(-1/3) * T_sun**(2/3) * m2 * (m1 + 2*m2) / total**(4/3)
        geodetic = (n*T_sun*total)**(2/3) * n * m2*(4*m1+3*m2)/(2*total**2)/(1-e**2)
        result = {
            "system": system.name,
            "om_dot_pred_deg_yr": float(advance * 180/np.pi * year_s),
            "Pdot_b_pred": float(pdot), "gamma_pred_ms": float(gamma * 1e3),
            "r_pred_us": float(T_sun * m2 * 1e6),
            "s_input": system.sin_i,
            "omega_geod_deg_yr": float(geodetic * 180/np.pi * year_s),
            "targets": {
                "om_dot_obs_deg_yr": system.om_dot_obs_deg_yr,
                "Pdot_b_obs": system.Pdot_b_obs, "Pdot_b_int_obs": system.Pdot_b_int_obs,
                "gamma_obs_ms": system.gamma_obs_ms, "r_obs_us": system.r_obs_us,
                "s_obs": system.s_obs,
            },
            "note": "Leading-order formulas evaluated with supplied masses and elements. "
                    "Inputs may depend on upstream fits. Differences omit parameter covariance, "
                    "higher-order corrections, and model uncertainty; they are not significance tests.",
        }
        pairs = {
            "om_dot": (result["om_dot_pred_deg_yr"], system.om_dot_obs_deg_yr),
            "Pdot_b": (pdot, system.Pdot_b_obs),
            "Pdot_b_int": (pdot, system.Pdot_b_int_obs),
            "gamma": (result["gamma_pred_ms"], system.gamma_obs_ms),
            "r": (result["r_pred_us"], system.r_obs_us),
        }
        result["relative_difference_pct"] = {key: _percent_difference(*pair)
                                            for key, pair in pairs.items() if pair[1] is not None}
        # Compatibility only: the CLI uses relative differences, not "agreement".
        result["agreements_pct"] = {key: 100 - value if value is not None else None
                                    for key, value in result["relative_difference_pct"].items()}
        return result


class RelationalTripleEngine:
    """Hierarchical tidal amplitude scales, not an exact multipole timing solution.

    Tidal scales use the near-circular approximation and omit orbital phases,
    angular coefficients and mutual-inclination dynamics. Eccentricity enters
    the isolated inner-binary Peters decay estimate, not the tidal amplitudes.
    """

    @staticmethod
    def solve(triple: HierarchicalTripleSystem, *, baseline_years=5.23):
        m1, m2, m3 = triple.m1_msun, triple.m2_msun, triple.m3_msun
        _positive(m1_msun=m1, m2_msun=m2, m3_msun=m3,
                  Pb1_days=triple.Pb1_days, Pb2_days=triple.Pb2_days, f0_hz=triple.f0_hz)
        for name in ("e1", "e2"):
            _interval(name, getattr(triple, name), 0, 1, include_upper=False)
        for name in ("cos_i1", "cos_i2"):
            _interval(name, getattr(triple, name), -1, 1)
        _interval("baseline_years", baseline_years, 0, np.inf)
        inner, total = m1 + m2, m1 + m2 + m3
        p1, p2 = triple.Pb1_days * day_s, triple.Pb2_days * day_s
        n1, n2 = 2*np.pi/p1, 2*np.pi/p2
        a1, a2 = (GM_sun*inner/n1**2)**(1/3), (GM_sun*total/n2**2)**(1/3)
        if a1 * (1 + triple.e1) >= a2 * (1 - triple.e2):
            raise ValueError("Inner apoapsis must be below outer periapsis for hierarchical scale estimates.")
        epsilon = a1 / a2
        asymmetry = (m1 - m2) / inner
        eta = m1*m2/inner**2
        sini = np.sqrt(1-triple.cos_i1**2)
        projected_axis = m2/inner * a1/c * sini
        displacement = m2/inner * a1 * m3/total * (n2/n1)**2
        pdot = -(192*np.pi/5) * (n1*T_sun*inner)**(5/3) * eta * _eccentricity_factor(triple.e1)
        n_dot = -n1 * pdot/p1
        return {
            "system": triple.name,
            "epsilon_hierarchy": float(epsilon), "mass_asymmetry": float(asymmetry),
            "A_quad_us": float(displacement/c * sini * 1e6),
            "A_oct_ns": float(displacement/c * sini * epsilon * asymmetry * 1e9),
            "A_hex_ns": float(displacement/c * sini * epsilon**2 * (1-3*eta) * 1e9),
            "delta_t_2pn_ps": float(projected_axis * (n1*a1/c)**4 * 1e12),
            "Pdot_b1": float(pdot),
            "gw_delay_scale_ns": float(projected_axis * 0.5*n_dot*(baseline_years*year_s)**2 * 1e9),
            "baseline_years": baseline_years,
            "published_sep": {
                "value": triple.delta_sep_published,
                "uncertainty_68": triple.delta_sep_uncertainty,
                "absolute_bound_95": triple.delta_sep_bound95,
                "source": triple.delta_sep_ref,
                "origin": "Reference metadata, not calculated by this engine",
            },
            "note": "Near-circular hierarchical tidal scales; accuracy requires a1/a2 much less than 1 "
                    "and small eccentricities. No certified truncation error, trajectory, orbital phase, "
                    "or mutual-inclination evolution is computed. cos_i2 is reference metadata. "
                    "Octupole scale is signed under inner-body relabeling. GW delay is an amplitude scale.",
        }


class CircumbinarySolver:
    """Leading secular quadrupole nodal-precession estimate for a light planet."""

    @staticmethod
    def solve(system: CircumbinaryExoplanetSystem):
        ma, mb, mp = system.m_starA_msun, system.m_starB_msun, system.m_planet_msun
        _positive(m_starA_msun=ma, m_starB_msun=mb,
                  P_bin_days=system.P_bin_days, P_planet_days=system.P_planet_days)
        _interval("m_planet_msun", mp, 0, np.inf)
        for name in ("e_bin", "e_planet"):
            _interval(name, getattr(system, name), 0, 1, include_upper=False)
        _interval("mutual_inclination_deg", system.mutual_inclination_deg, 0, 180)
        nbin, np_ = 2*np.pi/(system.P_bin_days*day_s), 2*np.pi/(system.P_planet_days*day_s)
        abin = (GM_sun*(ma+mb)/nbin**2)**(1/3)
        ap = (GM_sun*(ma+mb+mp)/np_**2)**(1/3)
        if abin*(1+system.e_bin) >= ap*(1-system.e_planet):
            raise ValueError("Circumbinary approximation requires separated inner and outer orbits.")
        factor = (1+1.5*system.e_bin**2)/(1-system.e_planet**2)**2
        cosine = np.cos(np.radians(system.mutual_inclination_deg))
        rate = 0.75*np_*(abin/ap)**2 * ma*mb/(ma+mb)**2 * factor * cosine
        period = None if abs(cosine) < 1e-14 else float(2*np.pi/abs(rate)/year_s)
        diff = _percent_difference(period, system.P_nodal_prec_yr_obs) if period is not None else None
        return {
            "system": system.name, "P_nodal_prec_yr_pred": period,
            "P_nodal_prec_yr_obs": system.P_nodal_prec_yr_obs,
            "relative_difference_pct": diff,
            "agreement_pct": 100-diff if diff is not None else None,
            "ttv_reference_hours": system.ttv_amplitude_hours,
            "note": "Leading secular quadrupole estimate. Precession and TTV reference values are "
                    "supplied comparison metadata; no transit light curve or TTV signal is computed.",
        }


class RelationalTimingEngine:
    """Convenience facade for the formula calculators and optional residual inspection."""

    @classmethod
    def run_all(cls, zenodo_csv: Optional[str] = None):
        from .systems import PSR_J0337_1715, PSR_J0737_3039, PSR_B1913_16, PSR_J1738_0333, KEPLER_16B
        results = {
            "PSR_J0737-3039": PostKeplerianSolver.solve(PSR_J0737_3039),
            "PSR_B1913+16": PostKeplerianSolver.solve(PSR_B1913_16),
            "PSR_J1738+0333": PostKeplerianSolver.solve(PSR_J1738_0333),
            "PSR_J0337+1715_multipoles": RelationalTripleEngine.solve(PSR_J0337_1715),
            "Kepler-16b": CircumbinarySolver.solve(KEPLER_16B),
        }
        if zenodo_csv is not None:
            results["PSR_J0337+1715_real_data"] = ZenodoResidualEvaluator.evaluate(zenodo_csv)
        return results
