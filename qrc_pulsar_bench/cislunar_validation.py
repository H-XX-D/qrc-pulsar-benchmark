"""Reproducible initial-only force ablations against bundled Horizons archives.

python -m qrc_pulsar_bench.cislunar_validation --ephemeris /path/de440s.bsp \
    --days 3 --output /path/results

No maneuvers or spacecraft properties are inferred from held-out data. An
ablation identifies model sensitivity, not the unique cause of a real residual.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import numpy as np
from .capstone_replay import MultiDatasetFlightReplay
from .cislunar_forces import CislunarForceParameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ephemeris", required=True, help="Local JPL DE440s BSP; never downloaded implicitly")
    parser.add_argument("--days", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    kernel = Path(args.ephemeris).resolve()
    if not kernel.is_file():
        parser.error("--ephemeris must be an existing local BSP file")
    if not np.isfinite(args.days) or args.days <= 0:
        parser.error("--days must be positive and finite")
    args.output.mkdir(parents=True, exist_ok=True)
    base = CislunarForceParameters()
    models = {
        "legacy_gm_moon_earth": replace(base, gm_moon=4.9048695e12, gm_earth=3.986004418e14, gm_sun=0, moon_j2=0),
        "de440_moon_earth": replace(base, gm_sun=0, moon_j2=0),
        "plus_sun": replace(base, moon_j2=0),
        "plus_sun_j2": base,
    }
    report = {
        "protocol": "Initial reconstructed spacecraft state only; preloaded planetary ephemerides; held-out spacecraft states only scored after propagation",
        "reference_kind": "Reconstructed Horizons ephemerides, not raw navigation measurements",
        "frame": "Moon-centred ICRF; SI; JD TDB epochs",
        "kernel_sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(),
        "models": {name: asdict(p) for name,p in models.items()},
        "limitations": ["No mission maneuvers", "SRP disabled: no verified mission area/mass/attitude",
                        "Approximate lunar J2 only, no high-degree gravity or tides", "No relativity or Earth oblateness",
                        "Analytic IAU lunar pole; no binary lunar orientation kernel"],
        "results": [],
    }
    for dataset in MultiDatasetFlightReplay.DATASET_FILES:
        replay = MultiDatasetFlightReplay(dataset, ephemeris=str(kernel))
        duration = min(args.days, replay.total_duration_days)
        source = Path(__file__).parent / replay.DATASET_FILES[dataset]
        for model, parameters in models.items():
            replay.parameters = parameters
            result = replay.run_forward(duration, max_step_s=600.0)
            np.savez_compressed(args.output / f"{dataset}-{model}.npz",
                                times_s=result["times_s"], states_si=result["states_si"],
                                position_error_m=result["position_error_m"])
            row = {k:v for k,v in result.items() if not isinstance(v,np.ndarray)}
            row.update(model=model, duration_days=float(result["times_s"][-1]/86400),
                       reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                       heldout_epochs=len(result["position_error_m"]))
            if model == "plus_sun_j2":
                tight = replay.run_forward(duration, max_step_s=300.0, rtol=2e-12)
                row["solver_refinement_max_position_change_m"] = float(np.max(np.linalg.norm(
                    result["states_si"][:,:3] - tight["states_si"][:,:3], axis=1)))
                diag = replay.residual_diagnostics()
                row["median_sampled_residual_mps2"] = float(np.median(np.linalg.norm(diag["residual_mps2"],axis=1)))
                row["median_differencing_disagreement_mps2"] = float(np.median(np.linalg.norm(diag["differencing_disagreement_mps2"],axis=1)))
            report["results"].append(row)
            (args.output / "results.json").write_text(json.dumps(report,indent=2)+"\n")
            print(f"{dataset:20s} {model:24s}: RMS {row['rms_position_error_m']/1e3:.6f} km, final {row['final_position_error_m']/1e3:.6f} km",flush=True)


if __name__ == "__main__":
    main()
