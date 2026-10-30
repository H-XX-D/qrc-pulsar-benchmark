"""Fit a declared early interval and score a later interval without refitting."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from qrc_pulsar_bench.observed_timing import summarize
from qrc_pulsar_bench.timing_fit import (
    calibrate,
    evaluate_frozen,
    predict,
    TimingDesign,
    linear_fit,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoff-mjd", type=float, required=True)
    parser.add_argument(
        "--fit-orbit", choices=["none", "epochs", "all"], default="none"
    )
    parser.add_argument(
        "--astrometry", choices=["none", "position", "motion"], default="none"
    )
    parser.add_argument("--astrometry-priors", action="store_true")
    parser.add_argument("--dm-order", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--post-newtonian", action="store_true")
    args = parser.parse_args()
    data = dict(np.load(args.input, allow_pickle=False))
    n = len(data["tcb_s"])
    train = 55920 + data["tcb_s"] / 86400 <= args.cutoff_mjd
    if not np.any(train) or not np.any(~train):
        parser.error("Both calibration and later intervals must contain observations")
    early = {
        key: (value[train] if len(value) == n else value) for key, value in data.items()
    }
    if args.fit_orbit == "none":
        prediction, metadata, wall = predict(early, np.zeros(14), args.post_newtonian)
        design = TimingDesign.from_training(
            early, args.astrometry, args.dm_order, args.astrometry_priors
        )
        residual, linear, matrix, names = linear_fit(early, prediction, design)
        fit = dict(
            offsets=np.zeros(14),
            linear=linear,
            linear_names=names,
            design=design,
            history=[],
            post_newtonian=args.post_newtonian,
        )
    else:
        fit = calibrate(
            early,
            args.astrometry,
            args.dm_order,
            args.post_newtonian,
            astrometry_priors=args.astrometry_priors,
            active_parameters=(
                ["tasc1", "tasc2"] if args.fit_orbit == "epochs" else None
            ),
            callback=lambda row: print(json.dumps(row), flush=True),
        )
    residual, prediction, metadata, wall = evaluate_frozen(data, fit)
    report = dict(
        cutoff_mjd=args.cutoff_mjd,
        fit_orbit=args.fit_orbit,
        astrometry=args.astrometry,
        astrometry_priors=args.astrometry_priors,
        dm_order=args.dm_order,
        post_newtonian=args.post_newtonian,
        offsets=fit["offsets"].tolist(),
        linear=fit["linear"].tolist(),
        linear_names=fit["linear_names"],
        design=asdict(fit["design"]),
        history=fit["history"],
        initial_conditions=metadata,
        training=summarize(residual, data["uncertainty_us"], train),
        held_out=summarize(residual, data["uncertainty_us"], ~train),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "fit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    np.savez_compressed(
        args.output / "prediction.npz", residual_s=residual, train=train, **prediction
    )
    print(json.dumps({key: report[key] for key in ["training", "held_out"]}, indent=2))


if __name__ == "__main__":
    main()
