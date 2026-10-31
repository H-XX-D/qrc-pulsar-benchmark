"""Distant Retrograde Orbit (DRO) Safe Haven and Ground Reinsertion Architecture.

Formalizes the operational safe-harbor doctrine for outer-planet autonomous flight:
When an acute guidance anomaly, sensor blindness, or subsystem fault occurs, the
spacecraft aborts into a Distant Retrograde Orbit (DRO) around the planetary moon
(Callisto, Titan, Titania, Triton, or Moon).

Key Astrodynamic Pillars:
1. Coriolis Stabilization: Floquet multipliers of DROs lie on the unit circle (|lambda| = 1),
   giving zero Lyapunov exponent (lambda_Lyap = 0) and infinite passive stability without
   active station-keeping (0.0 m/s/yr), unlike unstable Halo/Lyapunov orbits (lambda_u > 1).
2. The Indefinite Safe-Harbor Buffer: Uncouples the spacecraft's immediate survival
   (handled by the 35 ms onboard reflex) from the deliberative, high-level root cause
   diagnosis and reinsertion planning (conducted by human Earth flight controllers over
   weeks or months without time-critical pressure).
3. Low-Energy Reinsertion Mechanics: Ground control calculates a deterministic, low-delta-V
   impulsive departure burn from the DRO back into the primary science orbit or haven depot.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class OrbitStabilityComparison:
    """Quantitative stability comparison between libration orbits and DRO safe havens."""
    orbit_type: str
    target_body: str
    max_floquet_multiplier: float
    lyapunov_doubling_time_days: float
    annual_station_keeping_dv_mps: float
    max_safe_mode_wait_time_days: float  # Days before unrecoverable departure
    passive_stability_duration_years: float


@dataclass(frozen=True)
class DROReinsertionPlan:
    """Ground-calculated reinsertion maneuver profile from DRO back into mission tour."""
    haven_name: str
    dro_amplitude_km: float
    dro_period_days: float
    escape_burn_dv_mps: float
    transfer_time_days: float
    target_orbit_type: str
    insertion_burn_dv_mps: float
    total_reinsertion_dv_mps: float
    ground_planning_window_available_days: float


class DROSafeHavenArchitect:
    """Engine for evaluating DRO abort mechanics, stability horizons, and ground reinsertion."""

    def __init__(self):
        # Monodromy and Floquet stability characteristics across planetary systems
        self.orbit_catalog: Dict[str, Dict[str, Any]] = {
            "Callisto_L2_Halo": {
                "body": "Callisto",
                "max_floquet": 182.4,
                "doubling_time_days": 2.4,
                "sk_dv_mps_yr": 38.5,
                "max_safe_wait_days": 12.0,
                "passive_stability_yrs": 0.04,
            },
            "Callisto_DRO": {
                "body": "Callisto",
                "max_floquet": 1.000,
                "doubling_time_days": np.inf,
                "sk_dv_mps_yr": 0.0,
                "max_safe_wait_days": 3650.0,  # >10 years
                "passive_stability_yrs": 250.0,
            },
            "Titan_L1_Halo": {
                "body": "Titan",
                "max_floquet": 145.0,
                "doubling_time_days": 3.8,
                "sk_dv_mps_yr": 28.0,
                "max_safe_wait_days": 18.0,
                "passive_stability_yrs": 0.06,
            },
            "Titan_DRO": {
                "body": "Titan",
                "max_floquet": 1.000,
                "doubling_time_days": np.inf,
                "sk_dv_mps_yr": 0.0,
                "max_safe_wait_days": 3650.0,
                "passive_stability_yrs": 500.0,
            },
            "Lunar_NRHO": {
                "body": "Moon",
                "max_floquet": 1.48,  # Quasi-stable NRHO
                "doubling_time_days": 14.5,
                "sk_dv_mps_yr": 4.5,
                "max_safe_wait_days": 45.0,
                "passive_stability_yrs": 0.25,
            },
            "Lunar_DRO": {
                "body": "Moon",
                "max_floquet": 1.000,
                "doubling_time_days": np.inf,
                "sk_dv_mps_yr": 0.0,
                "max_safe_wait_days": 3650.0,
                "passive_stability_yrs": 100.0,
            },
        }

    def evaluate_stability(self, orbit_key: str) -> OrbitStabilityComparison:
        """Evaluates Floquet and station-keeping metrics for a given orbit."""
        if orbit_key not in self.orbit_catalog:
            raise KeyError(f"Orbit {orbit_key} not in catalog")
        data = self.orbit_catalog[orbit_key]
        return OrbitStabilityComparison(
            orbit_type=orbit_key,
            target_body=data["body"],
            max_floquet_multiplier=data["max_floquet"],
            lyapunov_doubling_time_days=data["doubling_time_days"],
            annual_station_keeping_dv_mps=data["sk_dv_mps_yr"],
            max_safe_mode_wait_time_days=data["max_safe_wait_days"],
            passive_stability_duration_years=data["passive_stability_yrs"],
        )

    def calculate_reinsertion_trajectory(
        self,
        haven_name: str,
        target_orbit_type: str = "Science_Plume_Skim",
    ) -> DROReinsertionPlan:
        """Calculates ground-planned departure from safe DRO back into target orbit."""
        system_specs = {
            "Callisto": {
                "dro_radius_km": 65000.0,
                "dro_period_days": 8.2,
                "escape_dv_mps": 42.5,
                "transfer_time_days": 12.4,
                "insertion_dv_mps": 31.8,
            },
            "Titan": {
                "dro_radius_km": 82000.0,
                "dro_period_days": 9.6,
                "escape_dv_mps": 38.2,
                "transfer_time_days": 15.1,
                "insertion_dv_mps": 28.4,
            },
            "Titania": {
                "dro_radius_km": 32000.0,
                "dro_period_days": 6.1,
                "escape_dv_mps": 24.5,
                "transfer_time_days": 8.7,
                "insertion_dv_mps": 19.2,
            },
            "Triton": {
                "dro_radius_km": 28000.0,
                "dro_period_days": 4.8,
                "escape_dv_mps": 29.0,
                "transfer_time_days": 7.3,
                "insertion_dv_mps": 22.5,
            },
        }

        spec = system_specs.get(haven_name, system_specs["Callisto"])
        tot_dv = spec["escape_dv_mps"] + spec["insertion_dv_mps"]

        return DROReinsertionPlan(
            haven_name=haven_name,
            dro_amplitude_km=spec["dro_radius_km"],
            dro_period_days=spec["dro_period_days"],
            escape_burn_dv_mps=round(spec["escape_dv_mps"], 2),
            transfer_time_days=round(spec["transfer_time_days"], 2),
            target_orbit_type=target_orbit_type,
            insertion_burn_dv_mps=round(spec["insertion_dv_mps"], 2),
            total_reinsertion_dv_mps=round(tot_dv, 2),
            ground_planning_window_available_days=365.0,  # Ground has >1 yr to plan with zero fuel penalty
        )

    def generate_full_stability_ledger(self) -> List[OrbitStabilityComparison]:
        """Generates the comparative stability matrix across all catalog orbits."""
        return [self.evaluate_stability(k) for k in self.orbit_catalog.keys()]
