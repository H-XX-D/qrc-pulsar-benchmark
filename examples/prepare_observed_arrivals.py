"""Prepare the publisher-verified J0337 TIM without its residual/derivative flags."""

import argparse
from copy import deepcopy
import hashlib, json
from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.time import Time
from pint.models import get_model
from pint.toa import get_TOAs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tim", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    checksum = hashlib.md5(args.tim.read_bytes()).hexdigest()
    if checksum != "b387a0c83cde7ab85942f4041c67337b":
        parser.error(
            "Input is not the original full-precision Zenodo 1241122 TIM (checksum mismatch)"
        )
    out.mkdir(parents=True, exist_ok=True)
    clean = ["FORMAT 1\n"]
    rows = []
    for line in args.tim.read_text().splitlines()[1:]:
        words = line.split()
        if not words:
            continue
        flags = dict(zip(words[5::2], words[6::2]))
        clean.append(" ".join(words[:5]) + f" -pn {flags['-pn']}\n")
        rows.append((words[4], int(flags["-pn"])))
    (out / "observations-only.tim").write_text("".join(clean))
    (out / "barycenter.par").write_text("""PSR J0337+1715
RAJ 03:37:43.82589000
DECJ 17:15:14.8281000
POSEPOCH 56337
F0 365.953363096
F1 -2.3658e-15
PEPOCH 55920
DM 21.3162
EPHEM DE405
CLK TT(TAI)
UNITS TDB
""")
    model = get_model(out / "barycenter.par")
    toas = get_TOAs(
        out / "observations-only.tim",
        model=model,
        ephem="DE405",
        include_bipm=False,
        planets=False,
        usepickle=True,
        picklefilename=str(out / "observations-only.pickle.gz"),
    )
    tdb = Time(list(toas.table["tdb"]))
    seconds = (
        (tdb.jd1 - 2400000.5 - 55920) * 86400
        + tdb.jd2 * 86400
        - model.delay(toas).to_value(u.s)
    )
    days = np.floor(seconds / 86400)
    tcb = Time(
        55920 + days, (seconds - days * 86400) / 86400, format="mjd", scale="tdb"
    ).tcb
    tcb_seconds = (tcb.jd1 - 2400000.5 - 55920) * 86400 + tcb.jd2 * 86400
    indices = np.asarray(toas.table["index"], int)
    names = ["RAJ", "DECJ", "PMRA", "PMDEC", "PX"]
    units = [u.mas, u.mas, u.mas / u.yr, u.mas / u.yr, u.mas]
    columns = []
    for name, unit in zip(names, units):
        delays = []
        for sign in [-1, 1]:
            trial = deepcopy(model)
            parameter = getattr(trial, name)
            parameter.quantity = parameter.quantity + sign * unit
            delays.append(trial.delay(toas).to_value(u.s))
        columns.append((delays[1] - delays[0]) / 2 / (1 - 1.550519768e-8))
    np.savez_compressed(
        out / "timing-inputs.npz",
        tdb_s=seconds,
        tcb_s=tcb_seconds,
        pulse_number=np.array([rows[i][1] for i in indices]),
        observatory=np.array([rows[i][0] for i in indices]),
        input_row=indices,
        uncertainty_us=toas.get_errors().to_value(u.us),
        frequency_mhz=toas.get_freqs().to_value(u.MHz),
        barycentric_frequency_mhz=model.barycentric_radio_freq(toas).to_value(u.MHz),
        astrometry_design_s=np.column_stack(columns),
        astrometry_names=np.array(names),
        astrometry_units=np.array([str(unit) for unit in units]),
    )
    (out / "provenance.json").write_text(
        json.dumps(
            dict(
                source="https://zenodo.org/records/1241122",
                md5=checksum,
                rows=len(toas),
                kept_flags=["pn"],
                computed_astrometry_derivatives="PINT central differences",
                reference_epoch_mjd=55920,
                time_basis="TCB",
                sky_position_epoch_mjd=56337,
                ephemeris="DE405",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
