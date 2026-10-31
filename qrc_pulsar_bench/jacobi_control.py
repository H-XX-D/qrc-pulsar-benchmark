"""Experimental cislunar guidance: restricted Jacobi propagation and bounded control.

The Earth--Moon inner Jacobi vector is a prescribed planetary input. The outer
Jacobi vector (spacecraft relative to the Earth--Moon barycentre) is integrated.
All force callbacks take Moon-relative ICRF position/velocity, SI, TDB seconds.
The controller receives navigation estimates and a *planned* reference, never
future observed spacecraft states. This is a simulation API, not flight software.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import lsq_linear

from .cislunar_forces import GM_EARTH, GM_MOON, R_EARTH, R_MOON
from .maneuvers import FiniteBurn

G0 = 9.80665


def _array(value, shape, name):
    out = np.asarray(value, dtype=float).copy()
    if out.shape != shape or not np.all(np.isfinite(out)):
        raise ValueError(name + " has invalid dimensions or nonfinite values")
    return out


@dataclass(frozen=True)
class Actuator:
    thrust_n: float = .2
    isp_s: float = 220.
    dry_mass_kg: float = 480.
    reserve_kg: float = .5
    routine_burn_s: float = 600.
    emergency_burn_s: float = 3600.
    minimum_dv_m_s: float = .0001

    def __post_init__(self):
        values = np.array(list(vars(self).values()))
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Nonnegative finite actuator parameters required")
        if min(self.isp_s, self.dry_mass_kg, self.routine_burn_s, self.emergency_burn_s) <= 0:
            raise ValueError("Positive Isp, dry mass and burn durations required")

    @property
    def exhaust_velocity(self):
        return G0 * self.isp_s

    def capacity(self, mass, emergency=False, interval_s=np.inf):
        if not np.isfinite(mass) or mass < self.dry_mass_kg:
            raise ValueError("Mass below dry mass")
        if np.isnan(interval_s) or interval_s <= 0:
            raise ValueError("Positive actuator interval required")
        floor = self.dry_mass_kg + (0 if emergency else self.reserve_kg)
        fuel = max(0., mass - floor)
        duration = min(self.emergency_burn_s if emergency else self.routine_burn_s, interval_s)
        consumed = min(fuel, self.thrust_n * duration / self.exhaust_velocity)
        return self.exhaust_velocity * np.log(mass / (mass - consumed))

    def command(self, dv, time_s, mass, *, emergency=False, interval_s=np.inf):
        dv = _array(dv, (3,), "delta v")
        if not np.isfinite(time_s):
            raise ValueError("Finite command epoch required")
        magnitude = np.linalg.norm(dv)
        allowed = min(magnitude, self.capacity(mass, emergency, interval_s))
        if self.thrust_n == 0 or allowed < self.minimum_dv_m_s or magnitude == 0:
            return None
        flow = self.thrust_n / self.exhaust_velocity
        duration = mass * (-np.expm1(-allowed / self.exhaust_velocity)) / flow
        return FiniteBurn(time_s, time_s + duration, self.thrust_n,
                          tuple(dv / magnitude), mass, flow, time_s,
                          "jacobi_control: simulated onboard command")


@dataclass(frozen=True)
class NavigationEstimate:
    state: np.ndarray                 # outer Jacobi position / velocity
    time_s: float                    # epoch to which estimate is propagated
    last_observation_s: float        # physical measurement epoch, not arrival time
    position_sigma_m: float = 0.
    velocity_sigma_m_s: float = 0.

    def __post_init__(self):
        values = [self.time_s, self.last_observation_s, self.position_sigma_m, self.velocity_sigma_m_s]
        if (not np.all(np.isfinite(values)) or self.last_observation_s > self.time_s
                or min(self.position_sigma_m, self.velocity_sigma_m_s) < 0):
            raise ValueError("Invalid navigation epochs or uncertainties")
        state = _array(self.state, (6,), "navigation state")
        state.setflags(write=False)
        object.__setattr__(self, "state", state)


@dataclass(frozen=True)
class ControlSettings:
    decision_interval_s: float = 6 * 3600.
    planning_horizon_s: float = 2 * 86400.
    linear_step_s: float = 3600.
    response_time_s: float = 12 * 3600.
    position_on_m: float = 300.
    position_off_m: float = 100.
    velocity_on_m_s: float = .01
    velocity_off_m_s: float = .003
    minimum_burn_spacing_s: float = 6 * 3600.
    maximum_observation_age_s: float = 2 * 3600.
    moon_floor_m: float = 50e3
    earth_floor_m: float = 200e3
    uncertainty_multiplier: float = 3.
    position_cost_scale_m: float = 1000.
    velocity_cost_scale_m_s: float = .05
    dv_cost_scale_m_s: float = .05

    def __post_init__(self):
        vals = np.array(list(vars(self).values()))
        if not np.all(np.isfinite(vals)) or np.any(vals < 0):
            raise ValueError("Finite nonnegative settings required")
        if min(self.decision_interval_s, self.planning_horizon_s, self.linear_step_s,
               self.response_time_s, self.position_cost_scale_m,
               self.velocity_cost_scale_m_s, self.dv_cost_scale_m_s) <= 0:
            raise ValueError("Positive time and cost scales required")
        if self.position_off_m >= self.position_on_m or self.velocity_off_m_s >= self.velocity_on_m_s:
            raise ValueError("Hysteresis off thresholds must be below on thresholds")


@dataclass
class Rollout:
    pieces: list
    start_s: float
    stop_s: float
    requested_stop_s: float
    min_clearance_m: float
    completed: bool

    def at(self, time_s):
        if not np.isfinite(time_s) or not self.start_s <= time_s <= self.stop_s + 1e-7:
            raise ValueError("Requested state outside rollout")
        for solution in self.pieces:
            if time_s <= solution.t[-1] + 1e-7:
                return solution.sol(min(time_s, solution.t[-1]))
        raise ValueError("Empty rollout")


class JacobiFlightModel:
    """Ephemeris-driven restricted Jacobi ODE with complete a(r,v,t) callback.

    inner(t) returns Moon-minus-Earth position, velocity and acceleration. These
    MUST be consistent derivatives (a C2 interpolation works). Merely adding a
    position offset while omitting its acceleration produces a fictitious force.
    """
    def __init__(self, acceleration, inner, *, moon_fraction=GM_MOON/(GM_EARTH+GM_MOON),
                 moon_radius_m=R_MOON, earth_radius_m=R_EARTH, max_step_s=1800., rtol=2e-13):
        parameters = [moon_fraction, moon_radius_m, earth_radius_m, max_step_s, rtol]
        if (not np.all(np.isfinite(parameters)) or not 0 < moon_fraction < 1
                or min(moon_radius_m, earth_radius_m, max_step_s, rtol) <= 0):
            raise ValueError("Invalid model geometry or integration tolerances")
        self.acceleration = acceleration
        self.inner = inner
        self.factor = 1 - moon_fraction
        self.moon_radius_m, self.earth_radius_m = moon_radius_m, earth_radius_m
        self.max_step_s, self.rtol = max_step_s, rtol

    def to_moon(self, state, t):
        q = _array(state, (6,), "Jacobi state")
        r, v, _ = self.inner(t)
        return q - self.factor * np.r_[r, v]

    def from_moon(self, state, t):
        x = _array(state, (6,), "Moon-relative state")
        r, v, _ = self.inner(t)
        return x + self.factor * np.r_[r, v]

    def clearances(self, state, t, moon_floor=0., earth_floor=0.):
        r = self.to_moon(state, t)[:3]
        inner, _, _ = self.inner(t)
        return np.array([np.linalg.norm(r)-self.moon_radius_m-moon_floor,
                         np.linalg.norm(r+inner)-self.earth_radius_m-earth_floor])

    def propagate(self, state, start, stop, mass, burn=None, *, floors=(0., 0.)):
        state = _array(state, (6,), "Jacobi state")
        floors = _array(floors, (2,), "clearance floors")
        if np.any(floors < 0):
            raise ValueError("Nonnegative clearance floors required")
        if not np.all(np.isfinite([start, stop, mass])) or stop <= start or mass <= 0:
            raise ValueError("Positive interval and mass required")
        if burn is not None:
            if burn.available_at_s > start or burn.start_s < start or burn.stop_s > stop + 1e-7:
                raise ValueError("Burn must be available and contained in propagation interval")
            if not np.isclose(burn.mass_start_kg, mass, atol=1e-8, rtol=1e-12):
                raise ValueError("Burn mass and propagated mass disagree")
        initial_clearance = float(min(self.clearances(state, start, *floors)))
        if initial_clearance <= 0:
            raise ValueError("Initial state violates body clearance")
        edges = sorted(set([start, stop] + ([] if burn is None else [burn.start_s, burn.stop_s])))
        y = np.r_[state, mass]
        pieces = []
        min_clearance = initial_clearance
        for left, right in zip(edges[:-1], edges[1:]):
            if right <= left:
                continue
            firing = burn is not None and left >= burn.start_s and right <= burn.stop_s + 1e-7
            force = np.asarray(burn.direction_icrf)*burn.thrust_n if firing else np.zeros(3)
            flow = burn.mass_flow_kg_s if firing else 0.
            def rhs(t, y):
                r, v = np.split(self.to_moon(y[:6], t), 2)
                _, _, inner_a = self.inner(t)
                a = _array(self.acceleration(r, v, t), (3,), "acceleration")
                return np.r_[y[3:6], a + self.factor*inner_a + force/y[6], -flow]
            def collision(t, y):
                return float(min(self.clearances(y[:6], t, *floors)))
            collision.terminal = True
            collision.direction = -1
            sol = solve_ivp(rhs, (left, right), y, method="DOP853", rtol=self.rtol,
                            atol=np.r_[[1e-6]*3, [1e-10]*3, 1e-10],
                            max_step=self.max_step_s, dense_output=True, events=collision)
            if not sol.success:
                raise RuntimeError(sol.message)
            pieces.append(sol)
            # Events detect crossings; node+midpoint minima are diagnostic only.
            # They do not certify arbitrary sub-step grazing encounters.
            sample_t = np.sort(np.r_[sol.t, (sol.t[:-1]+sol.t[1:])/2])
            min_clearance = min(min_clearance, min(float(min(self.clearances(sol.sol(t)[:6], t, *floors))) for t in sample_t))
            y = sol.y[:, -1]
            if sol.status == 1:
                break
        end = pieces[-1].t[-1]
        return Rollout(pieces, start, end, stop, min_clearance, end >= stop-1e-7)


@dataclass
class ControlDecision:
    mode: str
    reason: str
    burn: object = None
    planned_times_s: np.ndarray = field(default_factory=lambda: np.empty(0))
    planned_dv_m_s: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    predicted_clearance_m: Optional[float] = None
    candidates_evaluated: int = 0


class JacobiPredictiveController:
    """LTV receding-horizon planner, damped PD candidate and safety supervisor.

    Future impulsive controls are advisory. Only the first finite burn can be
    issued, after nonlinear validation. No integral accumulator: saturation and
    long sensor outages cannot wind up an integral term. No general nonlinear
    stability, collision-avoidance or optimal-fuel guarantee is asserted.
    """
    def __init__(self, model, reference, reference_end_s, *, actuator=Actuator(), settings=ControlSettings()):
        if not np.isfinite(reference_end_s) or reference_end_s <= 0:
            raise ValueError("Finite positive mission reference end required")
        self.model, self.reference, self.reference_end_s = model, reference, reference_end_s
        self.actuator, self.settings = actuator, settings
        self.active = False
        self.last_burn_s = -np.inf
        self._transition_cache = {}

    def _transition(self, t, h):
        key = (float(t), float(h))
        if key not in self._transition_cache:
            middle = t + h/2
            x = self.model.to_moon(self.reference(middle), middle)
            a = np.zeros((6, 6)); a[:3, 3:] = np.eye(3)
            for j in range(6):
                delta = np.zeros(6); delta[j] = 10. if j < 3 else .001
                plus, minus = x+delta, x-delta
                a[3:, j] = (self.model.acceleration(plus[:3], plus[3:], middle)
                            -self.model.acceleration(minus[:3], minus[3:], middle))/(2*delta[j])
            # Includes velocity-dependent forces. Frozen-gradient approximation
            # is used only to propose burns, then the first burn is checked.
            self._transition_cache[key] = expm(h*a)
        return self._transition_cache[key]

    def _plan(self, estimate, mass, end):
        s = self.settings; t = estimate.time_s
        horizon = min(s.planning_horizon_s, self.reference_end_s-t)
        nodes = np.arange(t, t+horizon-1e-6, s.decision_interval_s)
        n = len(nodes)
        response = np.zeros((6, 3*n))
        error = estimate.state-self.reference(t)
        rows, offsets = [], []
        scale = np.r_[[1/s.position_cost_scale_m]*3, [1/s.velocity_cost_scale_m_s]*3]
        for i, left in enumerate(nodes):
            response[3:, 3*i:3*i+3] += np.eye(3)
            right = min(left+s.decision_interval_s, t+horizon)
            steps = np.linspace(left, right, max(2, int(np.ceil((right-left)/s.linear_step_s))+1))
            for a, b in zip(steps[:-1], steps[1:]):
                phi = self._transition(a, b-a)
                response, error = phi@response, phi@error
            rows.append(scale[:, None]*response); offsets.append(scale*error)
        # Quadratic fuel proxy, not true minimum-L1 delta-v. The box constraint
        # is inscribed in the actuator's Euclidean ball, preserving its limit.
        matrix = np.vstack([*rows, np.eye(3*n)/s.dv_cost_scale_m_s])
        target = -np.r_[np.concatenate(offsets), np.zeros(3*n)]
        cap = self.actuator.capacity(mass, interval_s=end-t)
        reserve_floor = min(mass, self.actuator.dry_mass_kg+self.actuator.reserve_kg)
        total = self.actuator.exhaust_velocity*np.log(mass/reserve_floor)
        cap = min(cap, total/n) / np.sqrt(3)
        if cap <= 0:
            return nodes, np.zeros((n, 3))
        fit = lsq_linear(matrix, target, bounds=(-cap, cap), tol=1e-9, max_iter=100)
        if not fit.success or not np.all(np.isfinite(fit.x)):
            raise RuntimeError("Predictive least-squares planner failed")
        return nodes, fit.x.reshape(n, 3)

    def _damped_dv(self, estimate):
        s = self.settings; t = estimate.time_s
        target = self.reference(t)
        x = self.model.to_moon(estimate.state, t)
        ref = self.model.to_moon(target, t)
        error = estimate.state-target
        # With exact unsaturated continuous actuation: e''+2e'/tau+e/tau²=0.
        # Finite burns / sampled observations do not inherit that guarantee.
        a = (self.model.acceleration(ref[:3], ref[3:], t)
             -self.model.acceleration(x[:3], x[3:], t)
             -error[:3]/s.response_time_s**2-2*error[3:]/s.response_time_s)
        return s.decision_interval_s*a

    def _cost(self, rollout, start, end, burn):
        s = self.settings
        if not rollout.completed:
            return np.inf
        errors = [rollout.at(t)[:6]-self.reference(t) for t in np.linspace(start, end, 5)[1:]]
        cost = np.mean([np.dot(e[:3], e[:3])/s.position_cost_scale_m**2
                        +np.dot(e[3:], e[3:])/s.velocity_cost_scale_m_s**2 for e in errors])
        if burn is not None:
            final_mass = burn.mass_start_kg-burn.mass_flow_kg_s*(burn.stop_s-burn.start_s)
            dv = self.actuator.exhaust_velocity*np.log(burn.mass_start_kg/final_mass)
            cost += (dv/s.dv_cost_scale_m_s)**2
        return float(cost)

    def decide(self, estimate, mass):
        s = self.settings; t = estimate.time_s
        if not 0 <= t < self.reference_end_s:
            raise ValueError("Decision outside mission plan")
        self.actuator.capacity(mass)  # validates even when no burn is needed
        end = min(t+s.decision_interval_s, self.reference_end_s)
        margin = s.uncertainty_multiplier*(estimate.position_sigma_m+(end-t)*estimate.velocity_sigma_m_s)
        floors = (s.moon_floor_m+margin, s.earth_floor_m+margin)
        # This inflation is a declared heuristic, not a nonlinear reachable tube.
        try:
            coast = self.model.propagate(estimate.state, t, end, mass, floors=floors)
        except ValueError as exc:
            if "Initial state violates body clearance" not in str(exc):
                raise
            return ControlDecision("CLEARANCE_ALREADY_VIOLATED", "No certified recovery from inside the inflated exclusion zone")
        if not coast.completed:
            return self._emergency(estimate, mass, end, floors, coast)
        if t-estimate.last_observation_s > s.maximum_observation_age_s:
            return ControlDecision("NAVIGATION_STALE", "Routine thrust inhibited pending a fresh navigation solution",
                                   predicted_clearance_m=coast.min_clearance_m, candidates_evaluated=1)
        error = estimate.state-self.reference(t)
        er, ev = np.linalg.norm(error[:3]), np.linalg.norm(error[3:])
        if er < s.position_off_m and ev < s.velocity_off_m_s:
            self.active = False
        if er > s.position_on_m or ev > s.velocity_on_m_s:
            self.active = True
        if not self.active or t-self.last_burn_s < s.minimum_burn_spacing_s:
            return ControlDecision("COAST", "Hysteresis deadband or minimum burn spacing",
                                   predicted_clearance_m=coast.min_clearance_m, candidates_evaluated=1)
        if self.actuator.capacity(mass, interval_s=end-t) < self.actuator.minimum_dv_m_s:
            return ControlDecision("ROUTINE_AUTHORITY_EXHAUSTED", "Thrust/fuel unavailable or emergency reserve reached",
                                   predicted_clearance_m=coast.min_clearance_m, candidates_evaluated=1)
        try:
            times, schedule = self._plan(estimate, mass, end)
            candidates = [(schedule[0], "PREDICTIVE_BURN"), (self._damped_dv(estimate), "DAMPED_BURN")]
        except (RuntimeError, np.linalg.LinAlgError):
            times, schedule = np.empty(0), np.empty((0, 3))
            candidates = [(self._damped_dv(estimate), "DAMPED_FALLBACK")]
        best = ControlDecision("COAST", "No candidate improves validated tracking/fuel cost", None,
                               times, schedule, coast.min_clearance_m, 1)
        cost = self._cost(coast, t, end, None)
        for dv, mode in candidates:
            burn = self.actuator.command(dv, t, mass, interval_s=end-t)
            if burn is None:
                continue
            trial = self.model.propagate(estimate.state, t, end, mass, burn, floors=floors)
            best.candidates_evaluated += 1
            new_cost = self._cost(trial, t, end, burn)
            if new_cost < cost*(1-1e-6):
                cost = new_cost
                best.mode, best.reason, best.burn = mode, "Finite-burn rollout improves cost and retains predicted clearance", burn
                best.predicted_clearance_m = trial.min_clearance_m
        return best

    def confirm_execution(self, decision):
        """Update dwell state only after a command is actually accepted/executed."""
        if decision.burn is not None:
            self.last_burn_s = decision.burn.start_s

    def _emergency(self, estimate, mass, end, floors, coast):
        t = estimate.time_s
        r, v = np.split(self.model.to_moon(estimate.state, t), 2)
        inner, _, _ = self.model.inner(t)
        earth_r = r+inner
        directions = [r, -r, v, -v, earth_r, -earth_r, *np.eye(3), *(-np.eye(3))]
        cap = self.actuator.capacity(mass, True, end-t)
        chosen = ControlDecision("NO_FEASIBLE_EMERGENCY", "No tested finite burn maintains clearance; recovery is not established",
                                 predicted_clearance_m=coast.min_clearance_m, candidates_evaluated=1)
        best_dv = np.inf
        for direction in directions:
            norm = np.linalg.norm(direction)
            if norm == 0:
                continue
            for fraction in (.25, .5, 1.):
                dv = fraction*cap
                burn = self.actuator.command(direction/norm*dv, t, mass, emergency=True, interval_s=end-t)
                if burn is None:
                    continue
                trial = self.model.propagate(estimate.state, t, end, mass, burn, floors=floors)
                chosen.candidates_evaluated += 1
                if trial.completed and dv < best_dv:
                    best_dv = dv
                    chosen.mode = "EMERGENCY_BURN"
                    chosen.reason = "Tested finite burn avoids predicted exclusion-zone entry over this horizon"
                    chosen.burn = burn
                    chosen.predicted_clearance_m = trial.min_clearance_m
        return chosen
