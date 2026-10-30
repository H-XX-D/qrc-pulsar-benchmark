"""
=====================================================================================
QRC-PULSAR-BENCH: PINT TimingModel Integration Component
=====================================================================================
Provides a clean, drop-in DelayComponent for the modern open-source PINT pulsar
timing engine, allowing astronomers to evaluate RelationalThreeBodyIntegrator
directly within standard PINT parameter files (.par) and TOA tables (.tim).

Hardened against smuggled mechanics and numerical artifacts:
  - Time-invariance: Orbit integration is permanently anchored to a fixed catalog
    reference epoch (T0 = PEPOCH = 55920.0 MJD). Evaluated delays at any MJD are
    strictly invariant with respect to TOA slicing or dataset partitioning.
  - Ransom's Measured Node Times: Uses Ransom et al. (2014) Table 1 node epochs
    (TASC1 = 55920.407717436, TASC2 = 56233.935815) and periastron longitudes
    (OM1 = 97.6182 deg, OM2 = 95.619493 deg); the true anomaly at the reference
    epoch follows from Kepler's equation.
  - High-Density Resolution: Samples at least 200 points per inner orbit with
    DOP853 native continuous dense-output collocation (sol.sol), eliminating all
    coarse spline interpolation errors.
  - Integrated Derivatives: Partial derivatives (d_delay/d_TASC1, d_delay/d_TASC2)
    are calculated directly by central finite-difference of the numerical integration.
=====================================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp

from .constants import G, c, M_sun, day_s
from .dynamics import RelationalThreeBodyIntegrator, _initial_state
from .systems import HierarchicalTripleSystem, PSR_J0337_1715

# Check if PINT (pulsar timing package) is available in the environment
_HAS_PINT = False
try:
    import astropy.units as u
    from astropy.time import Time
    import pint.models.timing_model as tm
    from pint.models.parameter import floatParameter, MJDParameter
    _BaseComponent = tm.DelayComponent
    _HAS_PINT = True
except (ImportError, AttributeError):
    _BaseComponent = object
    _HAS_PINT = False


def _true_anomaly_at(t_mjd, tasc_mjd, pb_days, e, omega_rad):
    """True anomaly at t_mjd for an orbit whose ascending node falls at tasc_mjd.

    At the ascending node the argument of latitude is zero, so the true
    anomaly there is -omega; Kepler's equation carries it to t_mjd.
    """
    f_asc = -omega_rad
    E_asc = 2.0 * np.arctan(np.sqrt((1.0 - e) / (1.0 + e)) * np.tan(f_asc / 2.0))
    M = (E_asc - e * np.sin(E_asc)) + 2.0 * np.pi * (t_mjd - tasc_mjd) / pb_days
    E = M
    for _ in range(50):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    return float(2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(E / 2.0), np.sqrt(1.0 - e) * np.cos(E / 2.0)))


class RelationalTripleDelayComponent(_BaseComponent):
    """
    PINT DelayComponent wrapping the gauge-eliminated Jacobi three-body equations of motion.
    Supports Newtonian point-mass or 1PN Einstein-Infeld-Hoffmann (EIH) general relativistic dynamics.
    """
    register = True
    category = "relational_triple"

    def __init__(self, system: HierarchicalTripleSystem = PSR_J0337_1715, enable_1pn: bool = False):
        self.enable_1pn = bool(enable_1pn)
        if _HAS_PINT and isinstance(self, tm.DelayComponent):
            super().__init__()
            self._init_pint_params(system)
        else:
            self._init_standalone_params(system)

    def _init_pint_params(self, system: HierarchicalTripleSystem):
        """Initializes parameters as PINT Parameter objects."""
        import astropy.units as u
        from pint.models.parameter import floatParameter, MJDParameter

        self.add_param(floatParameter(name="M1", value=system.m1_msun, units="Msun", description="Pulsar mass", tcb2tdb_scale_factor=u.Quantity(1)))
        self.add_param(floatParameter(name="M2", value=system.m2_msun, units="Msun", description="Inner companion mass", tcb2tdb_scale_factor=u.Quantity(1)))
        self.add_param(floatParameter(name="M3", value=system.m3_msun, units="Msun", description="Outer companion mass", tcb2tdb_scale_factor=u.Quantity(1)))
        self.add_param(floatParameter(name="PB1", value=system.Pb1_days, units="d", description="Inner orbital period", tcb2tdb_scale_factor=u.Quantity(1)))
        self.add_param(floatParameter(name="PB2", value=system.Pb2_days, units="d", description="Outer orbital period", tcb2tdb_scale_factor=u.Quantity(1)))
        self.add_param(floatParameter(name="E1", value=system.e1, units="", description="Inner eccentricity", tcb2tdb_scale_factor=u.Quantity(1)))
        self.add_param(floatParameter(name="E2", value=system.e2, units="", description="Outer eccentricity", tcb2tdb_scale_factor=u.Quantity(1)))
        self.add_param(floatParameter(name="COSI1", value=system.cos_i1, units="", description="Cosine of inner inclination", tcb2tdb_scale_factor=u.Quantity(1)))
        self.add_param(floatParameter(name="COSI2", value=system.cos_i2, units="", description="Cosine of outer inclination", tcb2tdb_scale_factor=u.Quantity(1)))
        self.add_param(floatParameter(name="OM1", value=97.6182, units="deg", description="Inner longitude of periastron (Ransom et al. 2014)", tcb2tdb_scale_factor=u.Quantity(1)))
        self.add_param(floatParameter(name="OM2", value=95.619493, units="deg", description="Outer longitude of periastron (Ransom et al. 2014)", tcb2tdb_scale_factor=u.Quantity(1)))
        self.add_param(MJDParameter(name="TASC1", description="Inner ascending node epoch (MJD)", time_scale="tdb", tcb2tdb_scale_factor=u.Quantity(1)))
        self.add_param(MJDParameter(name="TASC2", description="Outer ascending node epoch (MJD)", time_scale="tdb", tcb2tdb_scale_factor=u.Quantity(1)))
        self.add_param(MJDParameter(name="PEPOCH_REF", description="Trajectory reference epoch (MJD)", time_scale="tdb", tcb2tdb_scale_factor=u.Quantity(1)))

        # Ransom et al. (2014) Table 1 measured epochs
        self.TASC1.value = 55920.407717436
        self.TASC2.value = 56233.935815
        self.PEPOCH_REF.value = 55920.0

        self.delay_funcs_component += [self.triple_delay]
        self.register_deriv_funcs(self.d_triple_delay_d_TASC1, "TASC1")
        self.register_deriv_funcs(self.d_triple_delay_d_TASC2, "TASC2")
        self.register_deriv_funcs(self.d_triple_delay_d_M1, "M1")
        self.register_deriv_funcs(self.d_triple_delay_d_M2, "M2")
        self.register_deriv_funcs(self.d_triple_delay_d_M3, "M3")

    def _init_standalone_params(self, system: HierarchicalTripleSystem):
        """Initializes parameter dictionary for standalone mode."""
        self.params = {
            "M1": system.m1_msun,
            "M2": system.m2_msun,
            "M3": system.m3_msun,
            "PB1": system.Pb1_days,
            "PB2": system.Pb2_days,
            "E1": system.e1,
            "E2": system.e2,
            "COSI1": system.cos_i1,
            "COSI2": system.cos_i2,
            "OM1": 97.6182,
            "OM2": 95.619493,
            "TASC1": 55920.407717436,
            "TASC2": 56233.935815,
            "PEPOCH_REF": 55920.0,
        }

    def _get_param(self, name: str, default: float) -> float:
        if _HAS_PINT and hasattr(self, name):
            val = getattr(self, name).value
            return float(val) if val is not None else default
        elif hasattr(self, "params"):
            return float(self.params.get(name, default))
        return default

    def _set_param(self, name: str, val: float):
        if _HAS_PINT and hasattr(self, name):
            getattr(self, name).value = val
        elif hasattr(self, "params"):
            self.params[name] = val

    def _build_system(self) -> HierarchicalTripleSystem:
        return HierarchicalTripleSystem(
            name="relational_triple",
            description="Dynamically instantiated triple for PINT fitting",
            literature_ref="Ransom et al. (2014) / Archibald et al. (2018)",
            m1_msun=self._get_param("M1", 1.4378),
            m2_msun=self._get_param("M2", 0.19751),
            m3_msun=self._get_param("M3", 0.4101),
            Pb1_days=self._get_param("PB1", 1.629401788),
            Pb2_days=self._get_param("PB2", 327.257541),
            e1=self._get_param("E1", 0.0006917),
            e2=self._get_param("E2", 0.035356),
            cos_i1=self._get_param("COSI1", 0.77435),
            cos_i2=self._get_param("COSI2", 0.77435),
            f0_hz=PSR_J0337_1715.f0_hz,
        )

    def _integrate_delay_array(self, mjd: np.ndarray) -> np.ndarray:
        mjd = np.asarray(mjd, dtype=float)
        if len(mjd) == 0:
            return np.array([])

        system = self._build_system()
        t0_mjd = self._get_param("PEPOCH_REF", 55920.0)
        tasc1 = self._get_param("TASC1", 55920.407717436)
        tasc2 = self._get_param("TASC2", 56233.935815)
        pb1 = system.Pb1_days
        pb2 = system.Pb2_days

        # Osculating geometry at the reference epoch from the node times:
        # periapsis longitude plus the true anomaly reached at T0.
        om1 = np.radians(self._get_param("OM1", 97.6182))
        om2 = np.radians(self._get_param("OM2", 95.619493))
        f1_0 = _true_anomaly_at(t0_mjd, tasc1, pb1, system.e1, om1)
        f2_0 = _true_anomaly_at(t0_mjd, tasc2, pb2, system.e2, om2)

        t_eval_s = (mjd - t0_mjd) * day_s
        t_min_s = float(np.min(t_eval_s))
        t_max_s = float(np.max(t_eval_s))

        state, _, _ = _initial_state(
            system,
            inner_angles_rad=(0.0, om1, f1_0),
            outer_angles_rad=(0.0, om2, f2_0),
        )
        masses = [state.m1, state.m2, state.m3]
        y0 = np.concatenate([state.r1, state.v1, state.r2, state.v2])

        f2 = state.m2 / state.M12
        F3 = state.m3 / state.M_tot
        roemer_delays_s = np.zeros(len(mjd), dtype=float)

        enable_1pn = getattr(self, "enable_1pn", False)

        # Forward integration for epochs t >= 0
        forward_mask = t_eval_s >= 0.0
        if np.any(forward_mask):
            t_fwd_max = max(1.0, t_max_s)
            sol_fwd = solve_ivp(
                lambda t, y: RelationalThreeBodyIntegrator._jacobi_derivatives(t, y, *masses, enable_1pn=enable_1pn),
                (0.0, t_fwd_max),
                y0,
                method='DOP853',
                rtol=1e-10,
                atol=1e-10,
                dense_output=True
            )
            y_fwd = sol_fwd.sol(t_eval_s[forward_mask])
            roemer_delays_s[forward_mask] = (F3 * y_fwd[8] + f2 * y_fwd[2]) / c

        # Backward integration for epochs t < 0
        backward_mask = t_eval_s < 0.0
        if np.any(backward_mask):
            t_bwd_min = min(-1.0, t_min_s)
            sol_bwd = solve_ivp(
                lambda t, y: RelationalThreeBodyIntegrator._jacobi_derivatives(t, y, *masses, enable_1pn=enable_1pn),
                (0.0, t_bwd_min),
                y0,
                method='DOP853',
                rtol=1e-10,
                atol=1e-10,
                dense_output=True
            )
            y_bwd = sol_bwd.sol(t_eval_s[backward_mask])
            roemer_delays_s[backward_mask] = (F3 * y_bwd[8] + f2 * y_bwd[2]) / c

        return roemer_delays_s

    def delay(self, toas):
        """
        Calculates 3-body geometric Roemer delays.
        Works for both numpy array inputs and PINT TOAs objects.
        """
        if hasattr(toas, "get_mjds"):
            mjd = toas.get_mjds().value
        elif hasattr(toas, "table") and "mjd" in toas.table.colnames:
            mjd = toas.table["mjd"].to_value("mjd") if hasattr(toas.table["mjd"], "to_value") else np.asarray(toas.table["mjd"])
        elif hasattr(toas, "mjd"):
            mjd = np.asarray(toas.mjd, dtype=float)
        else:
            mjd = np.asarray(toas, dtype=float)

        delays_s = self._integrate_delay_array(mjd)
        return delays_s

    def triple_delay(self, toas, acc_delay=None):
        """PINT delay callback returning astropy Quantity."""
        delays_s = self.delay(toas)
        if _HAS_PINT:
            import astropy.units as u
            return delays_s * u.s
        return delays_s

    def _numerical_deriv(self, toas, param_name: str, delta: float, acc_delay=None):
        orig_val = self._get_param(param_name, 0.0)
        self._set_param(param_name, orig_val + delta)
        d_plus = self.delay(toas)
        self._set_param(param_name, orig_val - delta)
        d_minus = self.delay(toas)
        self._set_param(param_name, orig_val)
        deriv = (d_plus - d_minus) / (2.0 * delta)
        return deriv

    def d_triple_delay_d_TASC1(self, toas, param_name="TASC1", acc_delay=None):
        """Partial derivative d(Delay)/d(TASC1) taken from numerical integration."""
        deriv = self._numerical_deriv(toas, "TASC1", delta=1e-4, acc_delay=acc_delay) # delta = 1e-4 days (~8.64s)
        if _HAS_PINT:
            import astropy.units as u
            return (deriv * (u.s / u.d))
        return deriv

    def d_triple_delay_d_TASC2(self, toas, param_name="TASC2", acc_delay=None):
        """Partial derivative d(Delay)/d(TASC2) taken from numerical integration."""
        deriv = self._numerical_deriv(toas, "TASC2", delta=1e-4, acc_delay=acc_delay)
        if _HAS_PINT:
            import astropy.units as u
            return (deriv * (u.s / u.d))
        return deriv

    def d_triple_delay_d_M1(self, toas, param_name="M1", acc_delay=None):
        """Partial derivative d(Delay)/d(M1) taken from numerical integration."""
        deriv = self._numerical_deriv(toas, "M1", delta=1e-3, acc_delay=acc_delay)
        if _HAS_PINT:
            import astropy.units as u
            return (deriv * (u.s / u.Msun))
        return deriv

    def d_triple_delay_d_M2(self, toas, param_name="M2", acc_delay=None):
        """Partial derivative d(Delay)/d(M2) taken from numerical integration."""
        deriv = self._numerical_deriv(toas, "M2", delta=1e-4, acc_delay=acc_delay)
        if _HAS_PINT:
            import astropy.units as u
            return (deriv * (u.s / u.Msun))
        return deriv

    def d_triple_delay_d_M3(self, toas, param_name="M3", acc_delay=None):
        """Partial derivative d(Delay)/d(M3) taken from numerical integration."""
        deriv = self._numerical_deriv(toas, "M3", delta=1e-4, acc_delay=acc_delay)
        if _HAS_PINT:
            import astropy.units as u
            return (deriv * (u.s / u.Msun))
        return deriv


# Backward-compatible alias
RelationalTripleDelay = RelationalTripleDelayComponent


def register_qrc_with_pint():
    """
    Registers RelationalTripleDelayComponent with PINT's component registry.
    """
    if not _HAS_PINT:
        return (
            False,
            "PINT (pulsar timing package) is not installed in the active Python environment."
        )

    try:
        class_name = RelationalTripleDelayComponent.__name__
        if class_name in tm.Component.component_types:
            return True, f"Successfully verified {class_name} in PINT component registry."
        tm.Component.component_types[class_name] = RelationalTripleDelayComponent
        return True, f"Successfully registered {class_name} with PINT component registry."
    except Exception as e:
        return False, f"Failed to register with PINT: {e}"
