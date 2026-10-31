"""
=====================================================================================
Astrophysical Example Systems and Comparison References
=====================================================================================
Frozen example inputs. Published timing masses may depend on the same fits as
comparison targets; freezing values does not make them independent evidence.
=====================================================================================
"""

from dataclasses import dataclass
import math
from typing import Optional

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
    om_dot_obs_deg_yr: Optional[float] = None     # Periastron advance (deg/yr)
    Pdot_b_obs: Optional[float] = None            # Observed GW orbital decay (s/s)
    Pdot_b_int_obs: Optional[float] = None        # Intrinsic GW decay after Galactic acceleration correction (s/s)
    gamma_obs_ms: Optional[float] = None          # Einstein delay (ms)
    r_obs_us: Optional[float] = None              # Shapiro delay range (us)
    s_obs: Optional[float] = None                 # Shapiro delay shape sin(i)

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
    # Optional reference metadata, not a computed prediction or a default for all triples.
    delta_sep_published: Optional[float] = None
    delta_sep_uncertainty: Optional[float] = None
    delta_sep_bound95: Optional[float] = None
    delta_sep_ref: Optional[str] = None
    # QRC theory prediction for Delta. Supplied by the theory, compared with
    # the published measurement above; never fitted to residuals.
    delta_sep_qrc_prediction: Optional[float] = None

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
    mutual_inclination_deg: float = 0.0  # Coplanar default; supply the system's value.

    def to_hierarchical_triple(self) -> 'HierarchicalTripleSystem':
        """Convert circumbinary exoplanet configuration to HierarchicalTripleSystem."""
        return HierarchicalTripleSystem(
            name=self.name,
            description=self.description,
            literature_ref=self.literature_ref,
            m1_msun=self.m_starA_msun,
            m2_msun=self.m_starB_msun,
            m3_msun=self.m_planet_msun,
            Pb1_days=self.P_bin_days,
            Pb2_days=self.P_planet_days,
            e1=self.e_bin,
            e2=self.e_planet,
            cos_i1=1.0,
            cos_i2=float(math.cos(math.radians(self.mutual_inclination_deg))),
            f0_hz=1.0,
        )

# 1. PSR J0337+1715: Hierarchical Stellar Triple Pulsar
PSR_J0337_1715 = HierarchicalTripleSystem(
    name="PSR J0337+1715",
    description="Hierarchical stellar triple system (millisecond pulsar + two white dwarfs). Ideal test of Strong Equivalence Principle in strong gravity.",
    literature_ref="Ransom et al., Nature 505, 520 (2014) Table 1 (orbit, masses, spin); Archibald et al., Nature 559, 73 (2018) (SEP Delta)",
    # Ransom et al. (2014) Table 1. Masses, periods and eccentricities come
    # from one fit so they stay consistent with that fit's node times and
    # periastron longitudes (used by pint_model and fitter).
    m1_msun=1.4378,        # mp = 1.4378(13)
    m2_msun=0.19751,       # mcI = 0.19751(15)
    m3_msun=0.4101,        # mcO = 0.4101(3)
    Pb1_days=1.629401788,  # Pb,I = 1.629401788(5) d
    Pb2_days=327.257541,   # Pb,O = 327.257541(7) d
    e1=6.9178e-4,          # eI = 6.9178(2)e-4
    e2=3.53561955e-2,      # eO = 3.53561955(17)e-2
    # tan i = (a sin i)/(a cos i): inner 1.21752844/1.4900 lt-s, outer
    # 74.6727101/91.42 lt-s (i about 39.25 deg).
    cos_i1=float(math.cos(math.atan2(1.21752844, 1.4900))),
    cos_i2=float(math.cos(math.atan2(74.6727101, 91.42))),
    f0_hz=365.953363096,   # f = 365.953363096(11) Hz
    # Archibald 2018 arXiv v1, main text: -1.09 +/- 0.74 (1 sigma), in 10^-6.
    delta_sep_published=-1.09e-6,
    delta_sep_uncertainty=0.74e-6,
    delta_sep_bound95=2.6e-6,
    delta_sep_ref="https://arxiv.org/abs/1807.02059v1",
    delta_sep_qrc_prediction=0.0
)

# 2. PSR J0737−3039A/B: The Double Pulsar (Kramer et al. 2021 PRX 11, 041050)
PSR_J0737_3039 = BinaryPulsarSystem(
    name="PSR J0737-3039A/B",
    description="The only known double pulsar system. Most relativistic binary known; 16 years of continuous timing validates 7 independent post-Keplerian parameters.",
    literature_ref="Kramer et al., Phys. Rev. X 11, 041050 (2021)",
    m1_msun=1.338185,
    m2_msun=1.248868,
    # Kramer 2021 Table IV; intrinsic decay from Eq. (42).
    Pb_days=0.1022515592973,
    eccentricity=0.087777023,
    x_p_s=1.415028603,
    sin_i=0.999936,
    om_dot_obs_deg_yr=16.899323,
    Pdot_b_obs=-1.24792e-12,
    Pdot_b_int_obs=-1.247752e-12,
    gamma_obs_ms=0.384045,
    r_obs_us=6.162,
    s_obs=0.999936
)

# 3. PSR B1913+16: The Hulse-Taylor Binary Pulsar (Weisberg & Huang 2016 ApJ 829:55)
PSR_B1913_16 = BinaryPulsarSystem(
    name="PSR B1913+16",
    description="First binary pulsar discovered (1974 Nobel Prize 1993). Provided first empirical evidence for gravitational wave emission.",
    literature_ref="Weisberg & Huang, Astrophys. J. 829:55 (2016); Damour & Taylor, ApJ 366:501 (1991)",
    # Weisberg & Huang 2016, abstract and Table 2 (rounded published masses).
    m1_msun=1.438,
    m2_msun=1.390,
    Pb_days=0.322997448918,
    eccentricity=0.6171340,
    x_p_s=2.341776,
    sin_i=0.7327,          # (sin i) from <omdot> and gamma, Sect. 4: 0.7327(4)
    om_dot_obs_deg_yr=4.226585,
    Pdot_b_obs=-2.423e-12,           # Observed decay (includes Galactic acceleration)
    Pdot_b_int_obs=-2.398e-12,       # Intrinsic decay (after Galactic acceleration correction of -0.025e-12)
    # Table 2 prints "gamma (ms) 0.004307(4)"; the value is in seconds
    # (4.307 ms), which the published masses reproduce to 0.05 percent.
    gamma_obs_ms=4.307
)

# 4. PSR J1738+0333: Asymmetric Pulsar + White Dwarf Binary
PSR_J1738_0333 = BinaryPulsarSystem(
    name="PSR J1738+0333",
    description="Relativistic asymmetric binary (pulsar + low-mass helium white dwarf) in circular orbit. Decisively constrains scalar-dipole gravitational radiation.",
    literature_ref="Freire et al., MNRAS 423, 3328–3343 (2012), https://arxiv.org/abs/1205.1450",
    # Freire et al. (2012) Table 1.
    m1_msun=1.46,                 # Mp = 1.46(+0.06/-0.05)
    m2_msun=0.181,                # Mc = 0.181(+0.008/-0.007), Antoniadis et al. (2012)
    Pb_days=0.3547907398724,      # Pb = 0.3547907398724(13) d
    eccentricity=3.4e-7,          # e = (3.4 +- 1.1)e-7
    x_p_s=0.3434,                 # x = 0.3434 lt-s (rounded in Table 1)
    sin_i=0.53877,                # s = 0.53877 (i = 32.6(1.0) deg)
    Pdot_b_obs=-17.0e-15,         # Pbdot = -17.0(3.1) fs/s
    Pdot_b_int_obs=-25.9e-15      # Pbdot_int = -25.9(3.2) fs/s
)

# 5. Kepler-16b: Circumbinary Exoplanet Benchmark
KEPLER_16B = CircumbinaryExoplanetSystem(
    name="Kepler-16b",
    description="First unambiguous circumbinary exoplanet discovered by NASA Kepler. Exhibits large transit timing variations (TTV) and nodal orbital precession.",
    literature_ref="Doyle et al., Science 333, 1602–1606 (2011)",
    # Planet and primary: NASA Exoplanet Archive record for Doyle et al.
    # (2011) (mass 0.6897, period 228.776 d, e 0.0069, 0.333 M_Jup).
    m_starA_msun=0.6897,
    m_starB_msun=0.20255,     # Doyle 2011 Table 1 (Science supplement); not text-verified here
    m_planet_msun=0.000318,   # 0.333 M_Jup
    P_bin_days=41.079220,     # Doyle 2011 Table 1 (Science supplement); not text-verified here
    P_planet_days=228.776,
    e_bin=0.15944,            # Doyle 2011 Table 1 (Science supplement); not text-verified here
    e_planet=0.0069,
    # UNSOURCED legacy comparison values. Doyle 2011 states only that the
    # secular timescale is "about 40 years" and the orbits align "within
    # 0.4 deg"; 44.1 yr, 1.52 h and 0.42 deg have no located source.
    P_nodal_prec_yr_obs=44.1,
    ttv_amplitude_hours=1.52,
    mutual_inclination_deg=0.42
)

SYSTEMS = {
    "PSR_J0337+1715": PSR_J0337_1715,
    "PSR_J0737-3039": PSR_J0737_3039,
    "PSR_B1913+16": PSR_B1913_16,
    "PSR_J1738+0333": PSR_J1738_0333,
    "Kepler-16b": KEPLER_16B
}
