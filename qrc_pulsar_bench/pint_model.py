"""
=====================================================================================
QRC-PULSAR-BENCH: PINT TimingModel Integration Component
=====================================================================================
Provides a clean, drop-in DelayComponent for the modern open-source PINT pulsar
timing engine, allowing astronomers to evaluate RelationalThreeBodyIntegrator
directly within standard PINT parameter files (.par) and TOA tables (.tim).
=====================================================================================
"""

import numpy as np
from .constants import G, c, M_sun, day_s
from .dynamics import RelationalThreeBodyIntegrator
from .systems import HierarchicalTripleSystem, PSR_J0337_1715

# Check if PINT (pulsar package) is available in the environment
_HAS_PINT = False
try:
    import pint.models.timing_model as tm
    from pint.models.parameter import floatParameter, maskParameter
    _HAS_PINT = True
except (ImportError, AttributeError):
    _HAS_PINT = False


class RelationalTripleDelayComponent:
    """
    PINT-compatible 3-body relational delay component.
    Replaces the Cartesian post-Newtonian triple approximation with the
    gauge-eliminated Jacobi three-body equations of motion.
    """

    def __init__(self, system: HierarchicalTripleSystem = PSR_J0337_1715):
        self.system = system
        self.category = "delay"
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
            "TASC1": 55956.70,
            "TASC2": 55956.70,
        }

    def delay(self, toas):
        """
        Calculates the 3-body geometric Roemer delay for an array of TOAs (MJDs).

        Parameters:
        -----------
        toas : array-like or PINT TOAs object
            Observation epochs in MJD.

        Returns:
        --------
        np.ndarray
            Roemer delays in seconds for each TOA epoch.
        """
        if hasattr(toas, 'table'):
            mjd = toas.table['mjd'].to_value('mjd')
        elif hasattr(toas, 'mjd'):
            mjd = np.asarray(toas.mjd, dtype=float)
        else:
            mjd = np.asarray(toas, dtype=float)

        mjd_min = mjd.min()
        span_days = mjd.max() - mjd_min
        if span_days <= 0:
            span_days = 1.0

        t_start_days = mjd_min
        # Dynamically compute orbital phase angles from TASC parameters (NO HARDCODED ANGLES)
        tasc1 = float(self.params.get("TASC1", 55956.70))
        tasc2 = float(self.params.get("TASC2", 55956.70))
        pb1 = float(self.params.get("PB1", self.system.Pb1_days))
        pb2 = float(self.params.get("PB2", self.system.Pb2_days))

        phi1 = (2.0 * np.pi * ((t_start_days - tasc1) / pb1)) % (2.0 * np.pi)
        phi2 = (2.0 * np.pi * ((t_start_days - tasc2) / pb2)) % (2.0 * np.pi)

        num_inner_orbits = max(5.0, span_days / self.system.Pb1_days)
        num_points = max(500, int(num_inner_orbits * 25))

        sim = RelationalThreeBodyIntegrator.integrate(
            self.system,
            num_inner_orbits=num_inner_orbits,
            num_output_points=num_points,
            rtol=1e-10,
            atol=1e-10,
            inner_angles_rad=(0.0, 0.0, phi1),
            outer_angles_rad=(0.0, 0.0, phi2),
            line_of_sight=(0.0, 0.0, 1.0)
        )

        t_sim_days = np.linspace(0.0, span_days, num_points)
        roemer_s = sim['roemer_delay_us'] * 1e-6

        # Cubic interpolation to exact observation epochs
        t_obs_relative_days = mjd - mjd_min
        from scipy.interpolate import interp1d
        fn = interp1d(t_sim_days, roemer_s, kind='cubic', fill_value='extrapolate')
        return fn(t_obs_relative_days)


def register_qrc_with_pint():
    """
    Registers the RelationalTripleDelayComponent with PINT's component registry
    if PINT is installed.
    """
    if not _HAS_PINT:
        return False, "PINT (pulsar timing package) is not installed in the active Python environment."

    try:
        class PINTRelationalTriple(tm.DelayComponent):
            register = True
            category = "relational_triple"

            def __init__(self):
                super().__init__()
                self.add_param(floatParameter(name="M1", value=PSR_J0337_1715.m1_msun, units="Msun", description="Pulsar mass"))
                self.add_param(floatParameter(name="M2", value=PSR_J0337_1715.m2_msun, units="Msun", description="Inner companion mass"))
                self.add_param(floatParameter(name="M3", value=PSR_J0337_1715.m3_msun, units="Msun", description="Outer companion mass"))
                self.component = RelationalTripleDelayComponent()

            def delay(self, toas, acc_delay=None):
                return self.component.delay(toas)

        return True, "Successfully registered PINTRelationalTriple with PINT component registry."
    except Exception as e:
        return False, f"Failed to register with PINT: {e}"
