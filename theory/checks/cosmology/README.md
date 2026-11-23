# Cosmology harness (Section 8 of the preprint)

These scripts fit flat LCDM, CPL (w0waCDM) and the QRC-motivated expansion laws to public
supernova, BAO and cosmic-chronometer data. They produced every number in Section 8 and
Table 3 of the preprint. They were written for an independent audit on 2026-11-22 and are
copied here unchanged.

| File | Role |
|---|---|
| `cosmo_harness.py` | Backgrounds (LCDM, CPL, R_h = ct, agegraphic, causal-memory kernel), data loaders, likelihoods with linear nuisances profiled analytically, fitter |
| `validate.py` | Step 1. Reproduces published flat-LCDM fits, rejects Einstein-de Sitter, runs 11 analytic checks and 5 deliberately wrong harness variants. Exit status 0 only if every reproduction passes |
| `run_models.py` | Step 2. Fits every model to every data combination. Refuses to run unless `validation_results.json` next to it has `ALL_PASS: true` |
| `audit_repo_claims.py` | Compares the hand-typed tables of the old multi-survey script with the real data (correction 6 in Section 7). Needs the QRC repository file `qrc_multi_survey_stress_test_suite.py`, which is not included; it is parsed with `ast` and never executed |
| `expected_output/` | The result files of the run reported in the preprint (`validation_results.json`, `model_results.json`, `repo_claims_results.json`) |

## Data (download them yourself)

The data files are not redistributed here. Put all seven files in one directory under
exactly these names; rename upstream files if their names differ (the Pantheon+ release,
for example, writes `Pantheon+` with a plus sign). Then compare the SHA-256 sums below with
`shasum -a 256 *` to confirm you have the same files.

| File name expected by the scripts | Source | SHA-256 of the file used |
|---|---|---|
| `PantheonPlusSH0ES.dat` | Pantheon+ DataRelease, <https://github.com/PantheonPlusSH0ES/DataRelease>, folder `Pantheon+_Data/4_DISTANCES_AND_COVAR/` | `1cb0fc379ef066afdc2ffd1857681cc478024570d8a3eba284fb645775198cf8` |
| `PantheonPlusSH0ES_STAT+SYS.cov` | same | `abf806d966485e64afdb359c87bffc0ecc00d05eff0a31ced66f247385df0fdc` |
| `DES-Dovekie_HD.csv` | DES-SN5YR, Dovekie recalibrated Hubble diagram, <https://github.com/des-science/DES-SN5YR>, folder `4_DISTANCES_COVMAT/` | `2f57019d783eaa976df80a41b0054171a2d994ee9808d715ce850c2df5720aaf` |
| `STAT+SYS.npz` | same release; packed upper triangle of the inverse covariance | `ffd3124b32148b1372bd95fda9299269f0352a9f8eee02d416c610e38495463b` |
| `desi_2024_gaussian_bao_ALL_GCcomb_mean.txt` | DESI 2024 DR1 BAO, CobayaSampler bao_data, <https://github.com/CobayaSampler/bao_data> | `dd2873a0b88459a491af3c0c0307ba059f62df9211d5b976760f310565a1be68` |
| `desi_2024_gaussian_bao_ALL_GCcomb_cov.txt` | same | `bbafa9074b51cf1a45e0d10e4f37db8c0e80a5d1d1788857abb7fc49fb21abcc` |
| `HzTable_MM_BC03.dat` | Moresco cosmic chronometers (BC03), CCcovariance, <https://gitlab.com/mmoresco/CCcovariance>, file `data/HzTable_MM_BC03.dat` | `32ce92caf251cb60a7a837c71f1856bea2b44fa5c1041f85410d11cb8164da98` |

The repository locations above are given at the level of the repository; check the exact
folder inside each one against its own README.

## Run

```sh
cd checks/cosmology
python3 -I -B validate.py   --data /path/to/data   # about 12 s, writes validation_results.json here
python3 -I -B run_models.py --data /path/to/data   # about 200 s, writes model_results.json here
# optional, needs the QRC repository script:
python3 -I -B audit_repo_claims.py --data /path/to/data --stress /path/to/qrc_multi_survey_stress_test_suite.py
```

Requirements: Python 3 with numpy and scipy (tested with Python 3.13.9, numpy 2.4.2,
scipy 1.16.3). The scripts read the data files as numbers only and write only their JSON
result files next to themselves (or to `--out`). Re-running both steps from this copy on
2026-11-22 reproduced every chi-square in `expected_output/model_results.json` exactly.

## What the validation does and does not catch

`validate.py` reproduces Pantheon+ (Omega_m = 0.3331 +- 0.0183 against the published
0.334 +- 0.018), Pantheon+ with SH0ES (H0 = 73.53 +- 1.02 against 73.6 +- 1.1), DESI DR1
BAO (Omega_m = 0.2946 +- 0.0146, h r_d = 101.90 +- 1.27 Mpc against 0.295 +- 0.015 and
101.8 +- 1.3) and DES-Dovekie (Omega_m = 0.3307 +- 0.0153 against 0.330 +- 0.015), with
tolerances fixed before the first run. Einstein-de Sitter is rejected by Delta chi^2 =
+693 (Pantheon+), +927 (DES) and +482 (DESI).

Of the five deliberately wrong harness variants, four are caught. The fifth (applying the
redshift cut to the inverse covariance instead of to the covariance) moves Omega_m by only
0.002, below tolerance, so it is not caught; the correct order (cut first, then invert)
was confirmed by reading `validate.build_pp`.

## Known limitations

No CMB likelihood. Chronometer errors are diagonal and omit correlated stellar-population
systematics. Omega_r is fixed and neutrinos are massless. DESI compressed BAO assumes a
near-fiducial cosmology, which matters most for R_h = ct. BIC uses N equal to all
supernovae. Pantheon+ and DES are never combined because they share supernovae.
