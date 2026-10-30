"""
=====================================================================================
Precision Astrophysical Reference Systems (Explicitly Hardcoded & Locked)
=====================================================================================
Contains the gold-standard precision relativistic systems in astrophysics.
All parameters are locked strictly from published observational literature.
No floating-point tuners or fitted free parameters are permitted.
=====================================================================================
"""

from dataclasses import dataclass
from typing import Optional
from .constants import M_sun, day_s, year_s

@dataclass(frozen=True)
class BinaryPulsarSystem:
    name: str
    description: str
    literature_ref: str
    # Primary & Secondary Masses (in Solar Masses M_sun)
    m1_msun: float
    m2_msun: float
    # Orbital period in days
    Pb_days: float
    # Orbital eccentricity
    eccentricity: float
    # Projected semi-major axis of pulsar x = a_p sin(i) / c in seconds
    x_p_s: float
    # Orbital inclination parameter sin(i)
    sin_i: float
    # Published Empirical Post-Keplerian Parameters (Targets)
    om_dot_obs_deg_yr: Optional[float] = None # Periastron advance (deg/yr)
    Pdot_b_obs: Optional[float] = None        # GW orbital decay (s/s)
    gamma_obs_ms: Optional[float] = None      # Einstein delay (ms)
    r_obs_us: Optional[float] = None          # Shapiro delay range (us)
    s_obs: Optional[float] = None             # Shapiro delay shape sin(i)

@dataclass(frozen=True)
class HierarchicalTripleSystem:
    name: str
    description: str
    literature_ref: str
    # Masses (M_sun)
    m1_msun: float  # Pulsar
    m2_msun: float  # Inner White Dwarf
    m3_msun: float  # Outer White Dwarf
    # Periods (days)
    Pb1_days: float # Inner orbit period
    Pb2_days: float # Outer orbit period
    # Eccentricities
    e1: float
    e2: float
    # Inclinations
    cos_i1: float
    cos_i2: float
    # Pulsar spin frequency (Hz)
    f0_hz: float
    # Strong Equivalence Principle Invariant
    delta_sep_qrc: float = 0.00000000

@dataclass(frozen=True)
class CircumbinaryExoplanetSystem:
    name: str
    description: str
    literature_ref: str
    # Masses (M_sun)
    m_starA_msun: float
    m_starB_msun: float
    m_planet_msun: float
    # Periods (days)
    P_bin_days: float
    P_planet_days: float
    # Eccentricities
    e_bin: float
    e_planet: float
    # Observed precession period (years)
    P_nodal_prec_yr_obs: float
    # Transit Timing Variation semi-amplitude (hours)
    ttv_amplitude_hours: float

# 1. PSR J0337+1715: Hierarchical Stellar Triple Pulsar
PSR_J0337_1715 = HierarchicalTripleSystem(
    name="PSR J0337+1715",
    description="Hierarchical stellar triple system (millisecond pulsar + two white dwarfs). Ideal test of Strong Equivalence Principle in strong gravity.",
    literature_ref="Archibald et al., Nature 559, 73–76 (2018); Voisin et al., A&A 638, A24 (2020)",
    m1_msun=1.4378,
    m2_msun=0.1975,
    m3_msun=0.4104,
    Pb1_days=1.629401787,
    Pb2_days=327.25754,
    e1=0.00069,
    e2=0.03536,
    cos_i1=0.7694,
    cos_i2=0.7696,
    f0_hz=365.953339305,
    delta_sep_qrc=0.00000000
)

# 2. PSR J0737−3039A/B: The Double Pulsar
PSR_J0737_3039 = BinaryPulsarSystem(
    name="PSR J0737-3039A/B",
    description="The only known double pulsar system. Most relativistic binary known; 16 years of continuous timing validates 7 independent post-Keplerian parameters.",
    literature_ref="Kramer et al., Phys. Rev. X 11, 041050 (2021)",
    m1_msun=1.338185,
    m2_msun=1.248868,
    Pb_days=0.10225156248,
    eccentricity=0.0877775,
    x_p_s=1.415032,
    sin_i=0.99987,
    om_dot_obs_deg_yr=16.89947,
    Pdot_b_obs=-1.2524e-12,
    gamma_obs_ms=0.3856,
    r_obs_us=6.153,
    s_obs=0.99987
)

# 3. PSR B1913+16: The Hulse-Taylor Binary Pulsar
PSR_B1913_16 = BinaryPulsarSystem(
    name="PSR B1913+16",
    description="First binary pulsar discovered (1974 Nobel Prize 1993). Provided first empirical evidence for gravitational wave emission.",
    literature_ref="Weisberg & Huang, Astrophys. J. 829:55 (2016)",
    m1_msun=1.4398,
    m2_msun=1.3886,
    Pb_days=0.322997448911,
    eccentricity=0.6171334,
    x_p_s=2.341776,
    sin_i=0.734,
    om_dot_obs_deg_yr=4.226598,
    Pdot_b_obs=-2.423e-12,
    gamma_obs_ms=4.295
)

# 4. PSR J1738+0333: Asymmetric Pulsar + White Dwarf Binary
PSR_J1738_0333 = BinaryPulsarSystem(
    name="PSR J1738+0333",
    description="Relativistic asymmetric binary (pulsar + low-mass helium white dwarf) in circular orbit. Decisively constrains scalar-dipole gravitational radiation.",
    literature_ref="Freire et al., Mon. Not. R. Astron. Soc. 423, 3301–3311 (2012)",
    m1_msun=1.46,
    m2_msun=0.181,
    Pb_days=0.35479073987,
    eccentricity=0.00000035,
    x_p_s=0.3434,
    sin_i=0.54,
    Pdot_b_obs=-2.59e-14
)

# 5. Kepler-16b: Circumbinary Exoplanet Benchmark
KEPLER_16B = CircumbinaryExoplanetSystem(
    name="Kepler-16b",
    description="First unambiguous circumbinary exoplanet discovered by NASA Kepler. Exhibits large transit timing variations (TTV) and nodal orbital precession.",
    literature_ref="Doyle et al., Science 333, 1602–1606 (2011)",
    m_starA_msun=0.6897,
    m_starB_msun=0.20255,
    m_planet_msun=0.000318, # 0.333 Jupiter masses
    P_bin_days=41.079220,
    P_planet_days=228.776,
    e_bin=0.15944,
    e_planet=0.0069,
    P_nodal_prec_yr_obs=44.1,
    ttv_amplitude_hours=1.52
)

SYSTEMS = {
    "PSR_J0337+1715": PSR_J0337_1715,
    "PSR_J0737-3039": PSR_J0737_3039,
    "PSR_B1913+16": PSR_B1913_16,
    "PSR_J1738+0333": PSR_J1738_0333,
    "Kepler-16b": KEPLER_16B
}
