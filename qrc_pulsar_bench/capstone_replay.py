"""Initial-state propagation and explicitly labelled offline flight comparisons.

Bundled Horizons vectors are interpreted as geometric Moon-centred ICRF states
at JD TDB epochs (km, km/s). Planetary ephemerides drive physical forces. Sampled
spacecraft residuals are diagnostics only and are NEVER fed into propagation.
The legacy controller uses interpolated reference truth: it is an ideal-feedback
experiment, not an onboard estimator or proof of autonomous flight capability.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline
from .cislunar_forces import (
    GM_MOON, GM_EARTH, CislunarForceParameters, PlanetaryEphemeris,
    cislunar_acceleration, third_body_tide,
)


@dataclass(frozen=True)
class FlightReplayReport:
    """Metrics for offline reference-feedback experiments, not flight certification."""
    dataset_name: str
    duration_days: float
    total_delta_v_mps: float
    delta_v_per_month_mps: float
    burn_count: int
    mean_burn_mps: float
    max_tracking_error_km: float
    rms_tracking_error_km: float
    mean_disturbance_mps2: float
    median_disturbance_mps2: float
    causal_integrity_verified: bool
    control_mode: str = "offline_reference_feedback"
    ground_connection_required: Optional[bool] = None
    uses_reference_feedback: bool = True


# Backward compatibility alias
CapstoneReplayReport = FlightReplayReport


class MultiDatasetFlightReplay:
    """Physical propagation with reference data isolated from force evaluation."""

    DATASET_FILES = {
        "capstone_nov2022": "capstone_horizons_nrho.npz",
        "capstone_feb2023": "capstone_feb2023_out_of_sample.npz",
        "artemis1_nov2022": "artemis1_nov2022_dro.npz",
        "lro_jan2023": "lro_jan2023_low_orbit.npz",
    }

    def __init__(
        self,
        dataset_name: str = "capstone_nov2022",
        dataset_path: Optional[str] = None,
        *,
        parameters: CislunarForceParameters = CislunarForceParameters(),
        ephemeris: str = "builtin",
        environment=None,
        lunar_gravity=None,
        maneuvers=None,
        maneuver_evidence=(),
        allow_retrospective_maneuvers=False,
        radiation_pressure=None,
        additional_acceleration=None,
    ):
        self.dataset_name = dataset_name
        module_dir = os.path.dirname(__file__)

        if dataset_path is not None:
            filepath = dataset_path
        elif dataset_name in self.DATASET_FILES:
            filename = self.DATASET_FILES[dataset_name]
            filepath = os.path.join(module_dir, filename)
        else:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Available: {list(self.DATASET_FILES.keys())}"
            )

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file {filepath} not found!")

        with np.load(filepath, allow_pickle=False) as data:
            self.times_jd = np.array(data['times_jd'], dtype=float)
            self.pos_km = np.array(data['positions_km'], dtype=float)
            self.vel_km_s = np.array(data['velocities_km_s'], dtype=float)
        n = self.times_jd.size
        if (self.times_jd.ndim != 1 or n < 3
                or self.pos_km.shape != (n, 3) or self.vel_km_s.shape != (n, 3)
                or not all(np.all(np.isfinite(x)) for x in (self.times_jd, self.pos_km, self.vel_km_s))
                or np.any(np.diff(self.times_jd) <= 0)):
            raise ValueError("Expected finite, strictly increasing epochs and Nx3 states")
        self.pos_m = self.pos_km * 1e3
        self.vel_m_s = self.vel_km_s * 1e3
        self.times_s = (self.times_jd - self.times_jd[0]) * 86400.0
        self.total_duration_s = float(self.times_s[-1])
        self.total_duration_days = self.total_duration_s / 86400.0
        self.n_epochs = n
        self.dt_nominal = float(np.median(np.diff(self.times_s)))
        self.parameters = parameters
        self.initial_state = np.r_[self.pos_m[0], self.vel_m_s[0]].copy()
        self.initial_state.setflags(write=False)
        self.environment = environment if environment is not None else PlanetaryEphemeris(
            self.times_jd[0], self.total_duration_s, ephemeris=ephemeris)
        self.ephemeris_source = getattr(self.environment, "source", "caller-supplied")
        self.lunar_gravity = lunar_gravity
        if radiation_pressure is not None and parameters.area_mass_m2_kg:
            raise ValueError("Configure either cannonball or plate SRP, not both")
        self.radiation_pressure = radiation_pressure
        self.additional_acceleration = additional_acceleration
        from .maneuvers import ManeuverSchedule
        self.maneuvers = maneuvers if maneuvers is not None else ManeuverSchedule()
        self.maneuver_evidence = tuple(maneuver_evidence)
        self.allow_retrospective_maneuvers = bool(allow_retrospective_maneuvers)
        self._reference = CubicHermiteSpline(self.times_s, self.pos_m, self.vel_m_s, extrapolate=False)
        self._diagnostics = None

    def _get_r_em(self, t_s):
        return self.environment(t_s)[0]

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        if not isinstance(value, CislunarForceParameters):
            raise TypeError("Expected CislunarForceParameters")
        if getattr(self,"radiation_pressure",None) is not None and value.area_mass_m2_kg:
            raise ValueError("Configure either cannonball or plate SRP, not both")
        self._parameters = value
        self._diagnostics = None

    def eval_earth_tide(self, r, t_s):
        return third_body_tide(r, self._get_r_em(t_s), self.parameters.gm_earth)

    def eval_total_acceleration(self, r, t_s):
        """Configured lunar gravity, differential Earth/Sun gravity and SRP.

        Commanded thrust is added separately by the segmented propagator.
        """
        environment = self.environment(t_s)
        if self.lunar_gravity is None and self.radiation_pressure is None:
            return cislunar_acceleration(r, *environment, self.parameters)
        from .cislunar_forces import acceleration_components
        terms = acceleration_components(r, *environment, self.parameters)
        # A complete harmonic field replaces both old lunar terms, including GM.
        lunar=(terms["moon_point_mass"]+terms["moon_j2"]) if self.lunar_gravity is None else self.lunar_gravity(r,t_s)
        srp=terms["srp"] if self.radiation_pressure is None else self.radiation_pressure(r,environment[0],environment[1],t_s)
        return lunar+terms["earth_tide"]+terms["sun_tide"]+srp

    @staticmethod
    def _differentiate(times, velocities, width):
        # Local polynomial weights use actual (possibly nonuniform) times.
        out = np.empty_like(velocities)
        for i, time in enumerate(times):
            start = min(max(0, i - width // 2), len(times) - width)
            sl = slice(start, start + width)
            offsets = times[sl] - time
            scale = np.max(np.abs(offsets))
            x = offsets / scale
            rhs = np.zeros(width)
            rhs[1] = 1.0 / scale
            weights = np.linalg.solve(np.vander(x, width, increasing=True).T, rhs)
            out[i] = weights @ velocities[sl]
        return out

    def eval_state_acceleration(self, r, v, t_s):
        """Full state-dependent force, including an explicitly supplied correction.

        Position-only callers of eval_total_acceleration retain their old
        contract. Velocity-dependent first-PN corrections require this method.
        """
        acceleration = self.eval_total_acceleration(r,t_s)
        if self.additional_acceleration is not None:
            extra = np.asarray(self.additional_acceleration(r,v,t_s),dtype=float)
            if extra.shape != (3,) or not np.all(np.isfinite(extra)):
                raise ValueError("Additional acceleration must be a finite three-vector")
            acceleration = acceleration + extra
        return acceleration

    def residual_diagnostics(self):
        """Offline finite-difference residuals, not identified physical forces.

        5-point versus 3-point disagreement diagnoses sampling sensitivity; it is
        not an uncertainty bound. Maneuvers, frame/state errors, differentiation
        error and omitted forces can all contribute. Endpoints are one-sided.
        """
        # Recompute: callers may change force callbacks or supplied commands.
        low = self._differentiate(self.times_s, self.vel_m_s, 3)
        high = self._differentiate(self.times_s, self.vel_m_s, min(5, self.n_epochs))
        modeled = np.array([self.eval_state_acceleration(r, v, t) + self.maneuvers.acceleration(t)
                            for r, v, t in zip(self.pos_m, self.vel_m_s, self.times_s)])
        return {
            "residual_mps2": high - modeled,
            "differencing_disagreement_mps2": high - low,
            "modeled_acceleration_mps2": modeled,
            "interpretation": "offline sampled residual including configured thrust; force attribution unverified",
            "used_by_propagator": False,
        }

    @property
    def a_dist_flight(self):
        """Legacy diagnostic alias; not an acceleration supplied to the solver."""
        return self.residual_diagnostics()["residual_mps2"]

    @property
    def mean_disturbance(self):
        return float(np.mean(np.linalg.norm(self.a_dist_flight, axis=1)))

    @property
    def median_disturbance(self):
        return float(np.median(np.linalg.norm(self.a_dist_flight, axis=1)))

    def eval_disturbance_at_timestamp(self, t_s):
        """Offline diagnostic interpolation ONLY. Never called by propagation."""
        if not 0 <= t_s <= self.total_duration_s:
            raise ValueError("Diagnostic epoch outside reference coverage")
        return np.array([np.interp(t_s, self.times_s, x) for x in self.a_dist_flight.T])

    def propagate(self, initial_state, times_s, *, max_step_s=120.0, rtol=1e-11):
        """Integrate supplied state at times_s[0], using no spacecraft observations.

        DOP853 error control separates integration error from missing-force error.
        External Earth/Sun ephemerides may be preloaded onboard. This is not an
        isolated four-body integration or an onboard state estimator.
        """
        times = np.asarray(times_s, dtype=float)
        state = np.asarray(initial_state, dtype=float)
        if (times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times))
                or np.any(np.diff(times) <= 0) or times[0] < 0 or times[-1] > self.total_duration_s
                or state.shape != (6,) or not np.all(np.isfinite(state))
                or not np.isfinite(max_step_s) or max_step_s <= 0 or not np.isfinite(rtol) or rtol <= 0):
            raise ValueError("Invalid initial state, propagation times or solver settings")
        if np.linalg.norm(state[:3]) <= self.parameters.moon_radius_m:
            raise ValueError("Initial state intersects Moon")
        def impact(t, y):
            return np.linalg.norm(y[:3]) - self.parameters.moon_radius_m
        impact.terminal = True
        impact.direction = -1
        self.maneuvers.validate_available(times[0],times[-1],self.allow_retrospective_maneuvers)
        if self.radiation_pressure is not None and self.radiation_pressure.available_at_s>times[0]:
            raise ValueError("Radiation-pressure inputs were not available at prediction start")
        boundaries=self.maneuvers.boundaries(times[0],times[-1])
        output=np.empty((len(times),6)); output[0]=state
        for left,right in zip(boundaries,boundaries[1:]):
            # Freeze which burn is active on this smooth integration segment.
            # Evaluate the left limit at burn stop so endpoint RK stages do not
            # accidentally use an off-thrust derivative for the powered segment.
            active=[b for b in self.maneuvers.burns if b.start_s<=0.5*(left+right)<b.stop_s]
            def rhs(t,y):
                thrust=sum((b.acceleration(min(t,np.nextafter(b.stop_s,-np.inf))) for b in active),np.zeros(3))
                return np.r_[y[3:], self.eval_state_acceleration(y[:3],y[3:],t)+thrust]
            indices=np.flatnonzero((times>left)&(times<=right))
            evaluate=np.unique(np.r_[times[indices],right])
            solution=solve_ivp(rhs,(left,right),state,t_eval=evaluate,
                method="DOP853",rtol=rtol,atol=np.r_[[1e-4]*3,[1e-7]*3],max_step=max_step_s,events=impact)
            if not solution.success or solution.y.shape[1]!=len(evaluate):
                raise RuntimeError("Propagation stopped: lunar impact or integration failure: "+solution.message)
            output[indices]=solution.y[:,:len(indices)].T
            state=solution.y[:,-1]
        return output

    def run_forward(self, duration_days=None, *, max_step_s=120.0, rtol=1e-11):
        """Initial-only forecast scored against later archived spacecraft epochs.

        The reconstructed initial state and planetary ephemeris are inputs. Held
        out spacecraft states are read only AFTER propagation, for scoring.
        """
        duration = self.total_duration_s if duration_days is None else duration_days * 86400.0
        if not np.isfinite(duration) or not 0 < duration <= self.total_duration_s:
            raise ValueError("Duration must lie within reference coverage")
        times = self.times_s[self.times_s <= duration]
        if len(times) < 2:
            raise ValueError("Duration must include at least one held-out epoch")
        predicted = self.propagate(self.initial_state, times, max_step_s=max_step_s, rtol=rtol)
        errors = np.linalg.norm(predicted[1:, :3] - self.pos_m[1:len(times)], axis=1)
        end_jd=self.times_jd[0]+times[-1]/86400.0
        events=[e for e in self.maneuver_evidence if e.start_jd_tdb<end_jd and e.stop_jd_tdb>self.times_jd[0]]
        return {
            "dataset": self.dataset_name, "times_s": times, "states_si": predicted,
            "position_error_m": errors, "rms_position_error_m": float(np.sqrt(np.mean(errors**2))),
            "max_position_error_m": float(np.max(errors)), "final_position_error_m": float(errors[-1]),
            "ephemeris_source": self.ephemeris_source, "uses_reference_feedback": False,
            "uses_trajectory_derived_forces": False,
            "additional_state_dependent_forces":self.additional_acceleration is not None,
            "srp_enabled": bool(self.parameters.srp_acceleration_1au) or self.radiation_pressure is not None,
            "commanded_burn_count":sum(b.start_s<times[-1] and b.stop_s>times[0] for b in self.maneuvers.burns),
            "retrospective_commands_allowed":self.allow_retrospective_maneuvers,
            "maneuver_screening":"known_event_in_interval" if events else "no_event_in_supplied_catalog",
            "maneuver_catalog_complete":False,
            "maneuver_evidence_sources":[e.source for e in events],
            "gravity_model":getattr(getattr(self.lunar_gravity,"field",None),"source","approximate J2"),
        }

    def get_reference_state(self, t_s: float) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolates target reference state [pos (m), vel (m/s)] at timestamp t_s."""
        if not np.isfinite(t_s) or not 0 <= t_s <= self.total_duration_s:
            raise ValueError("Reference epoch outside coverage")
        return self._reference(t_s), self._reference(t_s, 1)

    def advance_step(self, r, v, t_start, t_end, dt_sub=120.0):
        """Error-controlled physical propagation; no sampled residual injection."""
        state = self.propagate(np.r_[r, v], [t_start, t_end], max_step_s=dt_sub)[-1]
        return state[:3].copy(), state[3:].copy()

    def run_causal_replay(
        self,
        check_interval_h: float = 6.0,
        deadband_km: float = 25.0,
        duration_days: Optional[float] = None,
        autonomous: bool = True
    ) -> FlightReplayReport:
        """Legacy API for offline ideal-state feedback, not a causal flight test.

        The controller reads an interpolated archived reference. Only run_forward
        is an initial-only forecast. No trajectory-derived force is injected here.
        """
        sim_duration_days = duration_days if duration_days is not None else self.total_duration_days
        sim_duration_s = sim_duration_days * 86400.0

        if (not np.all(np.isfinite([sim_duration_s, check_interval_h, deadband_km]))
                or not 0 < sim_duration_s <= self.total_duration_s or check_interval_h <= 0 or deadband_km < 0):
            raise ValueError("Invalid replay interval, duration or deadband")
        dt_step = check_interval_h * 3600.0
        n_steps = int(np.ceil(sim_duration_s / dt_step))

        curr_pos, curr_vel = self.initial_state[:3].copy(), self.initial_state[3:].copy()
        curr_t = 0.0

        burns_dv: List[float] = []
        tracking_errors_km: List[float] = []

        # Characteristic orbital period for frequency tuning
        initial_radius_km = float(np.linalg.norm(curr_pos) / 1e3)
        is_nrho = self.dataset_name.startswith("capstone_")
        is_low_orbit = initial_radius_km < 3000.0

        if is_low_orbit:
            omega_orbit = 2.0 * np.pi / (118.0 * 60.0)
        elif is_nrho:
            omega_orbit = 2.0 * np.pi / (6.54 * 86400.0)
        else: # Distant Retrograde Orbit (DRO)
            omega_orbit = 2.0 * np.pi / (12.0 * 86400.0)

        dt_sub = 30.0 if is_low_orbit else 120.0

        for step in range(n_steps):
            t_next = min(curr_t + dt_step, sim_duration_s)

            # Propagate the configured physical force model
            curr_pos, curr_vel = self.advance_step(curr_pos, curr_vel, curr_t, t_next, dt_sub=dt_sub)
            curr_t = t_next

            # Offline reference truth, not a simulated onboard sensor observation
            ref_p, ref_v = self.get_reference_state(curr_t)
            err_r = curr_pos - ref_p
            err_v = curr_vel - ref_v
            err_km = float(np.linalg.norm(err_r) / 1e3)
            tracking_errors_km.append(err_km)

            if not autonomous:
                continue

            # Apolune gating rule: In high-eccentricity halo orbits, only burn near apolune
            r_curr_km = float(np.linalg.norm(curr_pos) / 1e3)
            is_optimal_window = (r_curr_km > 30000.0) if is_nrho else True

            if is_optimal_window and err_km > deadband_km:
                dv = - (0.35 * omega_orbit) * err_r - 0.45 * err_v
                dv_mag = float(np.linalg.norm(dv))

                # Thruster physical impulse clamp
                max_impulse = 0.5
                if dv_mag > max_impulse:
                    dv = dv * (max_impulse / dv_mag)
                    dv_mag = max_impulse

                curr_vel += dv
                burns_dv.append(dv_mag)

        total_dv = sum(burns_dv)
        dv_month = total_dv * (30.0 / sim_duration_days) if sim_duration_days > 0 else 0.0
        max_err = float(max(tracking_errors_km)) if tracking_errors_km else 0.0
        rms_err = float(np.sqrt(np.mean(np.array(tracking_errors_km)**2))) if tracking_errors_km else 0.0

        return FlightReplayReport(
            dataset_name=self.dataset_name,
            duration_days=sim_duration_days,
            total_delta_v_mps=total_dv,
            delta_v_per_month_mps=dv_month,
            burn_count=len(burns_dv),
            mean_burn_mps=float(np.mean(burns_dv)) if burns_dv else 0.0,
            max_tracking_error_km=max_err,
            rms_tracking_error_km=rms_err,
            mean_disturbance_mps2=self.mean_disturbance,
            median_disturbance_mps2=self.median_disturbance,
            causal_integrity_verified=False,
            control_mode="offline_reference_feedback" if autonomous else "uncontrolled_drift",
            ground_connection_required=None,
            uses_reference_feedback=autonomous
        )


class CapstoneFlightReplay(MultiDatasetFlightReplay):
    """Backward compatibility wrapper for CAPSTONE replay."""
    def __init__(self, dataset_path: Optional[str] = None, **kwargs):
        super().__init__(dataset_name="capstone_nov2022", dataset_path=dataset_path, **kwargs)

    def run_replay(
        self,
        autonomous: bool = True,
        check_interval_h: float = 6.0,
        lookahead_h: float = 36.0,
        deadband_km: float = 25.0,
        duration_days: Optional[float] = None
    ) -> FlightReplayReport:
        return self.run_causal_replay(
            check_interval_h=check_interval_h,
            deadband_km=deadband_km,
            duration_days=duration_days,
            autonomous=autonomous
        )
