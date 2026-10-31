"""Every catalog number must equal the value in its cited table.

Values below were read from the source papers' text: Ransom et al. (2014)
Table 1, Archibald et al. (2018) Table 1 and main text, Kramer et al. (2021)
Table IV, Weisberg and Huang (2016) Table 2 and Sect. 4, Freire et al. (2012)
Table 1, and the NASA Exoplanet Archive record for Doyle et al. (2011).
A failure here means a catalog value drifted from its source.
"""

import numpy as np
import pytest

from qrc_pulsar_bench.residuals import ARCHIBALD_SPIN_FREQUENCY_HZ
from qrc_pulsar_bench.systems import KEPLER_16B, PSR_B1913_16, PSR_J0337_1715, PSR_J0737_3039, PSR_J1738_0333

CASES = [
    # J0337+1715 orbit and masses: Ransom et al. (2014) Table 1, one fit, so
    # periods, node times and periastron longitudes stay mutually consistent.
    (PSR_J0337_1715, "m1_msun", 1.4378), (PSR_J0337_1715, "m2_msun", 0.19751),
    (PSR_J0337_1715, "m3_msun", 0.4101), (PSR_J0337_1715, "Pb1_days", 1.629401788),
    (PSR_J0337_1715, "Pb2_days", 327.257541), (PSR_J0337_1715, "e1", 6.9178e-4),
    (PSR_J0337_1715, "e2", 3.53561955e-2), (PSR_J0337_1715, "f0_hz", 365.953363096),
    (PSR_J0337_1715, "cos_i1", float(np.cos(np.arctan2(1.21752844, 1.4900)))),
    (PSR_J0337_1715, "cos_i2", float(np.cos(np.arctan2(74.6727101, 91.42)))),
    # SEP: Archibald et al. (2018) main text and abstract.
    (PSR_J0337_1715, "delta_sep_published", -1.09e-6), (PSR_J0337_1715, "delta_sep_uncertainty", 0.74e-6),
    (PSR_J0337_1715, "delta_sep_bound95", 2.6e-6),
    # J0737-3039: Kramer et al. (2021) Table IV and Eq. (42).
    (PSR_J0737_3039, "m1_msun", 1.338185), (PSR_J0737_3039, "m2_msun", 1.248868),
    (PSR_J0737_3039, "Pb_days", 0.1022515592973), (PSR_J0737_3039, "eccentricity", 0.087777023),
    (PSR_J0737_3039, "x_p_s", 1.415028603), (PSR_J0737_3039, "sin_i", 0.999936),
    (PSR_J0737_3039, "om_dot_obs_deg_yr", 16.899323), (PSR_J0737_3039, "Pdot_b_obs", -1.247920e-12),
    (PSR_J0737_3039, "Pdot_b_int_obs", -1.247752e-12), (PSR_J0737_3039, "gamma_obs_ms", 0.384045),
    (PSR_J0737_3039, "r_obs_us", 6.162),
    # B1913+16: Weisberg and Huang (2016) Table 2 and Sect. 4.
    (PSR_B1913_16, "m1_msun", 1.438), (PSR_B1913_16, "m2_msun", 1.390),
    (PSR_B1913_16, "Pb_days", 0.322997448918), (PSR_B1913_16, "eccentricity", 0.6171340),
    (PSR_B1913_16, "x_p_s", 2.341776), (PSR_B1913_16, "sin_i", 0.7327),
    (PSR_B1913_16, "om_dot_obs_deg_yr", 4.226585), (PSR_B1913_16, "Pdot_b_obs", -2.423e-12),
    (PSR_B1913_16, "Pdot_b_int_obs", -2.398e-12), (PSR_B1913_16, "gamma_obs_ms", 4.307),
    # J1738+0333: Freire et al. (2012) Table 1.
    (PSR_J1738_0333, "m1_msun", 1.46), (PSR_J1738_0333, "m2_msun", 0.181),
    (PSR_J1738_0333, "Pb_days", 0.3547907398724), (PSR_J1738_0333, "eccentricity", 3.4e-7),
    (PSR_J1738_0333, "x_p_s", 0.3434), (PSR_J1738_0333, "sin_i", 0.53877),
    (PSR_J1738_0333, "Pdot_b_obs", -17.0e-15), (PSR_J1738_0333, "Pdot_b_int_obs", -25.9e-15),
    # Kepler-16: NASA Exoplanet Archive record for Doyle et al. (2011).
    (KEPLER_16B, "m_starA_msun", 0.6897), (KEPLER_16B, "P_planet_days", 228.776),
    (KEPLER_16B, "e_planet", 0.0069),
]


@pytest.mark.parametrize("system,field,value", CASES, ids=lambda x: getattr(x, "name", x) if not isinstance(x, float) else "")
def test_catalog_value_matches_source(system, field, value):
    assert getattr(system, field) == pytest.approx(value, rel=1e-12, abs=0.0)


def test_kepler16_planet_mass_is_0333_jupiter_masses():
    # Doyle et al. (2011): 0.333 +- 0.016 M_Jup; IAU M_Jup/M_sun = 9.547919e-4.
    assert KEPLER_16B.m_planet_msun == pytest.approx(0.333 * 9.547919e-4, rel=2e-3)


def test_archibald_residual_conversion_uses_archibald_spin_frequency():
    # Archibald et al. (2018) Table 1: f = 365.953363080(4) Hz.
    assert ARCHIBALD_SPIN_FREQUENCY_HZ == 365.953363080
