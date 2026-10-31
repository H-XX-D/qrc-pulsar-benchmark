"""Reproduce harmonic-gravity forecasts and a separately labelled sensor experiment.

Inputs are explicit local files. --download-inputs fetches the four fixed public
gravity/orientation files below, validating their hashes before using them.
The synthetic sensor experiment does not constitute flight-sensor validation.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import urllib.request

import numpy as np

from .capstone_replay import MultiDatasetFlightReplay
from .cislunar_forces import CislunarForceParameters
from .lunar_gravity import LunarHarmonics, LunarOrientation, InertialLunarGravity
from .maneuvers import ManeuverEvidence, read_lro_small_force_metadata
from .navigation_filter import OrbitClockEKF


GRAIL = "https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/shadr/"
NAIF = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"
PHYSICS_INPUTS = {
    "gggrx_0900c_sha.tab": (GRAIL + "gggrx_0900c_sha.tab", "dab6ab06e0d3d7cbc594ea4bd03151a65534ed5fdf4f147ae38662428c04454e"),
    "gggrx_0900c_sha.lbl": (GRAIL + "gggrx_0900c_sha.lbl", "acc623e4b42a19f8feba1e7bda5189e984aa907491db12dede64bd426a1681c0"),
    "moon_pa_de421_1900-2050.bpc": (NAIF + "pck/moon_pa_de421_1900-2050.bpc", "656f90616403d75a75f0cd6c8830fc5b44f8cb4facb5ccb8915e752b397520cf"),
    "moon_080317.tf": (NAIF + "fk/satellites/moon_080317.tf", "78732477b96f9863e7b0d65bcee3c22b8707ca5ed0db56d1173319cb2e8c7993"),
}
CAPSTONE_EVENT = "https://www.nasa.gov/news-release/capstone-forges-new-path-for-nasas-future-artemis-moon-missions/"
ORION_EVENT = "https://www.nasa.gov/blogs/missions/2022/11/25/artemis-i-flight-day-10-orion-enters-distant-retrograde-orbit/"
LRO_OD = "https://ntrs.nasa.gov/api/citations/20140008968/downloads/20140008968.pdf"
SFF_NAME = "fdf25_sk34a_2015124_2015125_01.txt"
SFF_URL = "https://pds-geosciences.wustl.edu/lro/lro-l-rss-1-tracking-v1/lrors_0001/data/sff/" + SFF_NAME


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physics_inputs(directory, download=False):
    directory.mkdir(parents=True, exist_ok=True)
    receipts = []
    for name, (url, expected) in PHYSICS_INPUTS.items():
        path = directory / name
        if not path.exists() and download:
            with urllib.request.urlopen(url, timeout=120) as response:
                payload = response.read()
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError("Downloaded input hash mismatch: " + name)
            path.write_bytes(payload)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError("Missing or changed physics input: " + str(path))
        receipts.append(dict(file=name, url=url, sha256=expected, bytes=path.stat().st_size))
    return receipts


def independent_event_windows():
    from astropy.time import Time
    def jd(date):
        return float(Time(date, scale="utc").tdb.jd)
    # Broad evidence windows, deliberately not precise burn times or commands.
    # NASA reports two CAPSTONE cleanup maneuvers during its arrival week.
    return {
        "capstone_nov2022": (ManeuverEvidence(jd("2022-11-14"), jd("2022-11-21"),
            CAPSTONE_EVENT, "Arrival-week cleanup maneuvers; precise vectors unavailable"),),
        "artemis1_nov2022": (ManeuverEvidence(jd("2022-11-25"), jd("2022-11-26"),
            ORION_EVENT, "Documented distant-retrograde insertion burn; vector unavailable"),),
    }


def save_forecast(replay, label, days, output, **settings):
    started = time.perf_counter()
    result = replay.run_forward(days, **settings)
    row = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
    row.update(model=label, duration_days=float(result["times_s"][-1] / 86400),
               heldout_epochs=len(result["position_error_m"]), wall_seconds=time.perf_counter()-started)
    np.savez_compressed(output / f"{replay.dataset_name}-{label}.npz",
                        **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
    print(f"{replay.dataset_name:20s} {label:22s} RMS {row['rms_position_error_m']:.3f} m", flush=True)
    return row, result


def synthetic_navigation(replay, output, *, seed=20261031):
    """Four idealized unambiguous directions; first four hours sensed, then coast.

    Reference states generate *synthetic* noisy measurements only as their epoch
    is reached. The estimator receives that measurement, never the truth state.
    The final two-hour interval contains no measurement updates.
    """
    indices = np.flatnonzero(replay.times_s <= 6 * 3600 + 1e-4)[::5]
    times = replay.times_s[indices]
    if times[-1] < 6 * 3600 - 1e-3:
        raise ValueError("Synthetic navigation experiment requires six hours of reference")
    directions = np.array([[1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1]]) / np.sqrt(3)
    sigma, bias, drift, last_sensor = 300.0, 6000.0, 0.03, 4 * 3600
    perturbation = np.array([1000., -1000., 500., .5, -.5, .2])
    initial = replay.initial_state + perturbation
    # Assumptions specified before scoring; no tuning against held-out errors.
    covariance = np.diag(np.square([2000.,2000.,2000.,2.,2.,2.,10000.,.1]))
    estimator = OrbitClockEKF(np.r_[initial,0.,0.], covariance, replay.eval_total_acceleration,
                             acceleration_noise_psd=1e-10, clock_drift_noise_psd=1e-8,
                             state_acceleration=replay.eval_state_acceleration if replay.additional_acceleration is not None else None)
    rng = np.random.default_rng(seed)
    states, covariances, innovations, used = [], [], [], []
    started = time.perf_counter()
    for index, t in zip(indices, times):
        sensing = t <= last_sensor + 1e-4
        if sensing:
            measured = directions @ replay.pos_m[index] + bias + drift*t + rng.normal(0, sigma, 4)
            receipt = estimator.observe(t, measured, directions, np.eye(4)*sigma**2)
            innovations.append(receipt["normalized_innovation_squared"])
        else:
            estimator.predict_to(t)
        states.append(estimator.state.copy())
        covariances.append(estimator.covariance.copy())
        used.append(sensing)
    states, covariances, used = np.array(states), np.array(covariances), np.array(used)
    open_loop = replay.propagate(initial, times)
    # Truth is used here for evaluation, after the estimator has finished.
    errors = np.linalg.norm(states[:,:3] - replay.pos_m[indices], axis=1)
    open_errors = np.linalg.norm(open_loop[:,:3] - replay.pos_m[indices], axis=1)
    clock_errors = states[:,6] - (bias + drift*times)
    np.savez_compressed(output / "synthetic-navigation.npz", times_s=times, states_si=states,
                        covariance=covariances, sensor_available=used, position_error_m=errors,
                        open_loop_error_m=open_errors, clock_bias_error_m=clock_errors)
    rms = lambda values: float(np.sqrt(np.mean(values**2)))
    return {
        "classification": "Synthetic idealized pseudorange observations generated from reconstructed LRO positions; not flight sensor data",
        "measurement_equation": "z = direction dot r + clock_bias + independent Gaussian noise",
        "seed": seed, "directions_icrf": directions.tolist(), "noise_sigma_m": sigma,
        "state_dependent_extra_forces":replay.additional_acceleration is not None,
        "true_clock_bias_m": bias, "true_clock_drift_m_s": drift,
        "initial_orbit_error_si": perturbation.tolist(), "initial_covariance": covariance.tolist(),
        "acceleration_noise_psd_m2_s3": 1e-10, "clock_drift_noise_psd_m2_s3": 1e-8,
        "duration_s": float(times[-1]), "sensor_stop_s": last_sensor,
        "observations_per_epoch": 4, "sensed_epochs": int(used.sum()),
        "sensor_interval_s": float(np.median(np.diff(times))),
        "sensed_position_rms_m": rms(errors[used]), "outage_position_rms_m": rms(errors[~used]),
        "open_loop_sensed_position_rms_m": rms(open_errors[used]),
        "open_loop_outage_position_rms_m": rms(open_errors[~used]),
        "final_position_error_m": float(errors[-1]), "open_loop_final_position_error_m": float(open_errors[-1]),
        "clock_bias_rms_m": rms(clock_errors), "final_clock_bias_error_m": float(clock_errors[-1]),
        "final_clock_drift_error_m_s": float(states[-1,7]-drift),
        "mean_nis_four_observations": float(np.mean(innovations)),
        "wall_seconds": time.perf_counter()-started,
        "limitations": ["Synthetic simultaneous directions; no actual pulsar catalog/visibility/phase ambiguity",
                        "No raw photon/TOA adapter, relativistic delays, dispersion or detector noise model",
                        "A single fixed seed and six-hour arc; not a Monte Carlo consistency or flight test"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ephemeris", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--days", type=float, default=3.0)
    parser.add_argument("--download-inputs", action="store_true")
    parser.add_argument("--run-navigation", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.days) or not 0 < args.days <= 3:
        parser.error("--days must be positive and at most three (common archive coverage)")
    if not args.ephemeris.is_file() or not args.native_library.is_file():
        parser.error("Explicit existing ephemeris and native library required")
    args.output.mkdir(parents=True, exist_ok=True)
    receipts = physics_inputs(args.data_dir, args.download_inputs)
    kernels = [args.data_dir / n for n in ("moon_pa_de421_1900-2050.bpc", "moon_080317.tf")]
    def field(degree):
        return LunarHarmonics.from_shadr(args.data_dir / "gggrx_0900c_sha.tab", degree,
            frame="MOON_PA_DE421", source=GRAIL+"gggrx_0900c_sha.tab").use_native(args.native_library.resolve())
    field120 = field(120)
    report = dict(
        protocol="Same first spacecraft state; no fitting/resets/reference-derived forces; future spacecraft samples only scored after propagation",
        reference_kind="Reconstructed Horizons ephemerides, not raw navigation measurements",
        planetary_ephemeris_is_preloaded=True, frame="Moon-centred ICRF, metres, seconds, JD TDB",
        physics_inputs=receipts, ephemeris_sha256=sha256(args.ephemeris),
        native_library_sha256=sha256(args.native_library),
        lunar_model=dict(degree=120, gm_m3_s2=field120.gm, radius_m=field120.radius_m, frame=field120.frame),
        results=[], limitations=["No mission burn vectors supplied; evidence catalog incomplete",
            "Default SRP disabled; no verified epoch-specific spacecraft properties",
            "No lunar solid-body tides/thermal recoil/albedo, Earth oblateness, other planets or relativity",
            "Gravity parameters and reconstructed initial state have no propagated covariance in forecasts"],
    )
    source_root = Path(__file__).resolve().parent
    sources = [source_root / n for n in ("capstone_replay.py", "cislunar_forces.py", "lunar_gravity.py",
        "maneuvers.py", "radiation_pressure.py", "navigation_filter.py", "cislunar_refinement.py")]
    report["source_sha256"] = {p.name: sha256(p) for p in sources}
    def checkpoint():
        (args.output / "results.json").write_text(json.dumps(report, indent=2)+"\n")
    events = independent_event_windows()
    for dataset in MultiDatasetFlightReplay.DATASET_FILES:
        replay = MultiDatasetFlightReplay(dataset, ephemeris=str(args.ephemeris.resolve()),
                                         maneuver_evidence=events.get(dataset, ()))
        row, _ = save_forecast(replay, "sun-j2", args.days, args.output)
        row["reference_sha256"] = sha256(source_root / replay.DATASET_FILES[dataset])
        report["results"].append(row)
        orientation = LunarOrientation(replay.times_jd[0], replay.total_duration_s, kernels)
        replay.lunar_gravity = InertialLunarGravity(field120, orientation)
        row, nominal = save_forecast(replay, "grgm900c-degree120", args.days, args.output)
        row["reference_sha256"] = report["results"][-1]["reference_sha256"]
        tight = replay.run_forward(args.days, max_step_s=60, rtol=2e-12)
        row["solver_refinement_max_position_change_m"] = float(np.max(np.linalg.norm(
            nominal["states_si"][:,:3]-tight["states_si"][:,:3], axis=1)))
        report["results"].append(row)
        checkpoint()
        if dataset != "lro_jan2023":
            continue
        report["lro_degree_sensitivity"] = []
        for degree in (2, 20, 40, 80, 180):
            replay.lunar_gravity = InertialLunarGravity(field(degree), orientation)
            row, result = save_forecast(replay, f"grgm900c-degree{degree}", args.days, args.output)
            row["max_position_change_from_degree120_m"] = float(np.max(np.linalg.norm(
                nominal["states_si"][:,:3]-result["states_si"][:,:3], axis=1)))
            report["lro_degree_sensitivity"].append(row)
        replay.lunar_gravity = InertialLunarGravity(field120, orientation)
        report["lro_historical_srp_sensitivity"] = []
        for mass in (1000., 1057.105, 1100.):
            replay.parameters = CislunarForceParameters(area_mass_m2_kg=14/mass, reflectivity=1.67)
            row, _ = save_forecast(replay, f"historical-srp-{mass:g}kg", args.days, args.output)
            row.update(mass_kg=mass, area_m2=14, reflectivity=1.67, source=LRO_OD,
                classification="Historical cannonball area/Cr; mass sensitivity, not verified January 2023 inputs")
            report["lro_historical_srp_sensitivity"].append(row)
        replay.parameters = CislunarForceParameters()
        if args.run_navigation:
            report["synthetic_navigation"] = synthetic_navigation(replay, args.output)
        checkpoint()
    sff = args.data_dir / SFF_NAME
    if sff.is_file():
        metadata = read_lro_small_force_metadata(sff, source=SFF_URL)
        metadata["event"] = asdict(metadata["event"])
        metadata["sha256"] = sha256(sff)
        report["historical_maneuver_metadata_example"] = metadata
    checkpoint()


if __name__ == "__main__":
    main()
