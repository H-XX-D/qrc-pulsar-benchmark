"""Physical cislunar forces in Moon-centred, nonrotating ICRF axes, SI units.

Planetary ephemerides are external model inputs; spacecraft reference trajectories
are never force inputs. Epochs are JD TDB. DE440 GMs are from NAIF gm_de440.tpc.
The default J2 is an approximate axisymmetric Moon, not a mascon gravity field.
"""

from dataclasses import dataclass
import numpy as np
from scipy.interpolate import CubicHermiteSpline

GM_MOON = 4.9028001184575496e12
GM_EARTH = 3.9860043550702266e14
GM_SUN = 1.3271244004127942e20
R_MOON = 1737.4e3
R_EARTH = 6378.137e3
R_SUN = 695700e3
AU = 149597870700.0
SOLAR_PRESSURE_1AU = 4.56e-6  # N/m², nominal; not measured mission irradiance


def _vector(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite three-vector")
    return value


def point_mass_acceleration(r, gm):
    r = _vector(r, "position")
    distance = np.linalg.norm(r)
    if distance == 0:
        raise ValueError("Point-mass collision")
    return -gm * r / distance**3


def third_body_tide(r, body_position, gm):
    """Differential acceleration: spacecraft attraction minus Moon attraction."""
    r = _vector(r, "position")
    body = _vector(body_position, "body_position")
    return point_mass_acceleration(r - body, gm) - point_mass_acceleration(-body, gm)


def j2_acceleration(r, gm, radius, j2, pole=(0.0, 0.0, 1.0)):
    """Axisymmetric quadrupole acceleration, allowing any inertial spin axis.

    Also accepts consistently nondimensional r, GM and radius. This conservative
    gravity term is not a non-conservative disturbance or a full mascon model.
    """
    r = _vector(r, "position")
    pole = _vector(pole, "pole")
    rn, kn = np.linalg.norm(r), np.linalg.norm(pole)
    if rn == 0 or kn == 0:
        raise ValueError("J2 requires nonzero position and pole")
    k = pole / kn
    z = np.dot(r, k)
    return 1.5 * j2 * gm * radius**2 / rn**5 * ((5 * (z / rn)**2 - 1) * r - 2 * z * k)


def lunar_pole_icrf(jd_tdb):
    """IAU Moon pole from NAIF pck00011.tpc (RA/Dec and 13 nutations).

    This analytic orientation is less accurate than a lunar binary PCK. J2 is
    axisymmetric, so the prime meridian is not needed here.
    """
    t = (float(jd_tdb) - 2451545.0) / 36525.0
    angles = np.deg2rad(np.array([
        125.045, 250.089, 260.008, 176.625, 357.529, 311.589, 134.963,
        276.617, 34.226, 15.134, 119.743, 239.961, 25.053,
    ]) + t * np.array([
        -1935.5364525, -3871.072905, 475263.3328725, 487269.629985,
        35999.0509575, 964468.49931, 477198.869325, 12006.300765,
        63863.5132425, -5806.6093575, 131.84064, 6003.1503825, 473327.79642,
    ]))
    ra = np.deg2rad(269.9949 + 0.0031 * t + np.dot(
        [-3.8787, -.1204, .0700, -.0172, 0, .0072, 0, 0, 0, -.0052, 0, 0, .0043], np.sin(angles)))
    dec = np.deg2rad(66.5392 + 0.0130 * t + np.dot(
        [1.5419, .0239, -.0278, .0068, 0, -.0029, .0009, 0, 0, .0008, 0, 0, -.0009], np.cos(angles)))
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def sunlight_fraction(r, earth, sun, moon_radius=R_MOON, earth_radius=R_EARTH):
    """Binary conical umbra by Moon/Earth; partial eclipses are NOT resolved."""
    r, earth, sun = (_vector(x, n) for x, n in [(r, "position"), (earth, "earth"), (sun, "sun")])
    to_sun = sun - r
    ds = np.linalg.norm(to_sun)
    if ds <= R_SUN:
        raise ValueError("Spacecraft is inside the Sun")
    sun_angle = np.arcsin(R_SUN / ds)
    for centre, radius in ((np.zeros(3), moon_radius), (earth, earth_radius)):
        to_body = centre - r
        db = np.linalg.norm(to_body)
        if db <= radius:
            raise ValueError("Spacecraft is inside an occulting body")
        if db >= ds:
            continue
        separation = np.arccos(np.clip(np.dot(to_body, to_sun) / (db * ds), -1, 1))
        if separation + sun_angle <= np.arcsin(radius / db):
            return 0.0
    return 1.0


@dataclass(frozen=True)
class CislunarForceParameters:
    gm_moon: float = GM_MOON
    gm_earth: float = GM_EARTH
    gm_sun: float = GM_SUN
    moon_radius_m: float = R_MOON
    moon_j2: float = 2.0e-4
    area_mass_m2_kg: float = 0.0  # Unknown mission property: explicit opt-in, no fitted residual.
    reflectivity: float = 1.3
    earth_radius_m: float = R_EARTH

    def __post_init__(self):
        values = list(vars(self).values())
        if not np.all(np.isfinite(values)) or any(x < 0 for x in values):
            raise ValueError("Force parameters must be finite and nonnegative")
        if self.gm_moon == 0 or self.moon_radius_m == 0 or self.earth_radius_m == 0:
            raise ValueError("Central GM and body radii must be positive")

    @property
    def srp_acceleration_1au(self):
        return SOLAR_PRESSURE_1AU * self.reflectivity * self.area_mass_m2_kg

    def native_parameters(self):
        """Stable C ABI order, matching hardware/cislunar_force_core.h."""
        return np.array([self.gm_moon, self.gm_earth, self.gm_sun, self.moon_radius_m,
                         self.moon_j2, self.srp_acceleration_1au, self.earth_radius_m])


def acceleration_components(r, earth, sun, pole, parameters=CislunarForceParameters()):
    p = parameters
    r = _vector(r, "position")
    terms = {
        "moon_point_mass": point_mass_acceleration(r, p.gm_moon),
        "earth_tide": third_body_tide(r, earth, p.gm_earth),
        "sun_tide": third_body_tide(r, sun, p.gm_sun),
        "moon_j2": j2_acceleration(r, p.gm_moon, p.moon_radius_m, p.moon_j2, pole),
        "srp": np.zeros(3),
    }
    if p.srp_acceleration_1au:
        away = r - _vector(sun, "sun")
        distance = np.linalg.norm(away)
        visibility = sunlight_fraction(r, earth, sun, p.moon_radius_m, p.earth_radius_m)
        terms["srp"] = visibility * p.srp_acceleration_1au * AU**2 * away / distance**3
    return terms


def cislunar_acceleration(r, earth, sun, pole, parameters=CislunarForceParameters()):
    return sum(acceleration_components(r, earth, sun, pole, parameters).values())


class PlanetaryEphemeris:
    """Preloaded Earth/Sun positions relative to Moon, independent of flight data.

    ephemeris='builtin' uses Astropy's approximate solar-system model. For measured
    mission comparisons supply a local DE440s BSP path (requires jplephem). No
    mission-trajectory samples are used. The fixed 1-hour Hermite grid uses both
    planetary positions and velocities, and disallows extrapolation.
    """

    def __init__(self, epoch_jd_tdb, duration_s, ephemeris="builtin", grid_step_s=3600.0):
        if not np.all(np.isfinite([epoch_jd_tdb, duration_s, grid_step_s])) or duration_s <= 0 or grid_step_s <= 0:
            raise ValueError("Finite epoch, positive duration and grid step required")
        try:
            from astropy.coordinates import get_body_barycentric_posvel
            from astropy.time import Time, TimeDelta
        except ImportError as exc:
            raise ImportError("Planetary ephemerides require qrc-pulsar-bench[navigation]") from exc
        self.epoch_jd_tdb = float(epoch_jd_tdb)
        self.duration_s = float(duration_s)
        self.source = str(ephemeris)
        grid = np.linspace(0, duration_s, max(2, int(np.ceil(duration_s / grid_step_s)) + 1))
        times = Time(epoch_jd_tdb, format="jd", scale="tdb") + TimeDelta(grid, format="sec")
        moon_p, moon_v = get_body_barycentric_posvel("moon", times, ephemeris=ephemeris)
        self._splines = []
        for body in ("earth", "sun"):
            pos, vel = get_body_barycentric_posvel(body, times, ephemeris=ephemeris)
            self._splines.append(CubicHermiteSpline(
                grid, (pos.xyz - moon_p.xyz).to_value("m").T,
                (vel.xyz - moon_v.xyz).to_value("m/s").T, extrapolate=False))

    def __call__(self, time_s):
        if not np.isfinite(time_s) or not 0 <= time_s <= self.duration_s:
            raise ValueError("Epoch outside preloaded planetary ephemeris")
        earth, sun = (s(time_s) for s in self._splines)
        pole = lunar_pole_icrf(self.epoch_jd_tdb + time_s / 86400.0)
        return earth, sun, pole
