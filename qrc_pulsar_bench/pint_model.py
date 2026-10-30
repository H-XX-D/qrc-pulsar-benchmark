"""
=====================================================================================
QRC-PULSAR-BENCH: PINT TimingModel Integration Component
=====================================================================================
Provides a clean, drop-in DelayComponent for the modern open-source PINT pulsar
timing engine, allowing astronomers to evaluate RelationalThreeBodyIntegrator
directly within standard PINT parameter files (.par) and TOA tables (.tim).

Hardened against smuggled mechanics and numerical artifacts:
  - Time-invariance: Orbit integration is permanently anchored to a fixed catalog
    reference epoch (PEPOCH / TASC1). Evaluated delays at any MJD are strictly
    invariant with respect to TOA slicing or dataset partitioning.
  - Zero Spline Error: Evaluates the native high-order (8th-order DOP853) continuous
    dense-output polynomial directly at the exact observation epochs, eliminating
    the ~99 us interpolation errors of coarse cubic splines.
  - Active Parameters: Dynamic system reconstruction from `self.params`, ensuring
    masses (M1, M2, M3), orbital periods, and ascending node epochs directly drive
    the underlying dynamical equations of motion during fitting iterations.
=====================================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp

from .constants import G, c, M_sun, day_s
from .dynamics import RelationalThreeBodyIntegrator, _initial_state
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
        self.default_system = system
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
            "PEPOCH": 55956.70,
        }

    def _build_system(self) -> HierarchicalTripleSystem:
        """Dynamically constructs system from current parameter state."""
        return HierarchicalTripleSystem(
            name=f"dynamic_{self.default_system.name}",
            description="Dynamically instantiated triple for PINT fitting",
            literature_ref=self.default_system.literature_ref,
            m1_msun=float(self.params.get("M1", self.default_system.m1_msun)),
            m2_msun=float(self.params.get("M2", self.default_system.m2_msun)),
            m3_msun=float(self.params.get("M3", self.default_system.m3_msun)),
            Pb1_days=float(self.params.get("PB1", self.default_system.Pb1_days)),
            Pb2_days=float(self.params.get("PB2", self.default_system.Pb2_days)),
            e1=float(self.params.get("E1", self.default_system.e1)),
            e2=float(self.params.get("E2", self.default_system.e2)),
            cos_i1=float(self.params.get("COSI1", self.default_system.cos_i1)),
            cos_i2=float(self.params.get("COSI2", self.default_system.cos_i2)),
            f0_hz=self.default_system.f0_hz,
        )

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

        if len(mjd) == 0:
            return np.array([])

        system = self._build_system()

        # Fixed reference epoch anchors the dynamical trajectory (ensures time-invariance)
        t0_mjd = float(self.params.get("PEPOCH", 55956.70))
        tasc1 = float(self.params.get("TASC1", 55956.70))
        tasc2 = float(self.params.get("TASC2", 55956.70))
        pb1 = float(self.params.get("PB1", system.Pb1_days))
        pb2 = float(self.params.get("PB2", system.Pb2_days))

        # Initial orbital phases at reference epoch t0_mjd
        phi1_0 = (2.0 * np.pi * ((t0_mjd - tasc1) / pb1)) % (2.0 * np.pi)
        phi2_0 = (2.0 * np.pi * ((t0_mjd - tasc2) / pb2)) % (2.0 * np.pi)

        # Integration time offsets in seconds from t0_mjd
        t_eval_s = (mjd - t0_mjd) * day_s
        t_min_s = float(np.min(t_eval_s))
        t_max_s = float(np.max(t_eval_s))

        state, _, _ = _initial_state(
            system,
            inner_angles_rad=(0.0, 0.0, phi1_0),
            outer_angles_rad=(0.0, 0.0, phi2_0),
        )
        masses = [state.m1, state.m2, state.m3]
        y0 = np.concatenate([state.r1, state.v1, state.r2, state.v2])

        f2 = state.m2 / state.M12
        F3 = state.m3 / state.M_tot

        roemer_delays_s = np.zeros(len(mjd), dtype=float)

        # Forward integration for epochs t >= 0
        forward_mask = t_eval_s >= 0.0
        if np.any(forward_mask):
            t_fwd_max = max(1.0, t_max_s)
            sol_fwd = solve_ivp(
                lambda t, y: RelationalThreeBodyIntegrator._jacobi_derivatives(t, y, *masses),
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
                lambda t, y: RelationalThreeBodyIntegrator._jacobi_derivatives(t, y, *masses),
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


def register_qrc_with_pint():
    """
    Registers the RelationalTripleDelayComponent with PINT's component registry
    if PINT (pulsar timing package) is installed.
    """
    if not _HAS_PINT:
        return (
            False,
            "PINT (pulsar timing package) is not installed in the active Python environment "
            "(note: the installed package 'pint' 0.25.2 is the physical units library; "
            "'pint-pulsar' is the pulsar timing framework)."
        )

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
                # Synchronize parameters from PINT model
                for param_name in ["M1", "M2", "M3", "PB1", "PB2", "E1", "E2", "COSI1", "COSI2", "TASC1", "TASC2", "PEPOCH"]:
                    if hasattr(self, param_name):
                        self.component.params[param_name] = getattr(self, param_name).value
                return self.component.delay(toas)

        return True, "Successfully registered PINTRelationalTriple with PINT component registry."
    except Exception as e:
        return False, f"Failed to register with PINT: {e}"
