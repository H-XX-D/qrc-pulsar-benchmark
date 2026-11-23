#!/usr/bin/env python3
"""Audit the numbers in qrc_multi_survey_stress_test_suite.py against real data.

The stress-test source is parsed with ast (literal tables only); it is not imported
or executed. Usage:
  python3 -I -B audit_repo_claims.py --data DIR --stress PATH [--out repo_claims_results.json]
"""
import argparse
import ast
import itertools
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from scipy.integrate import quad  # noqa: E402
import cosmo_harness as H  # noqa: E402

CLAIMS = {"PP_LCDM": 1.483, "PP_QRC": 0.259, "QSO_LCDM": 3.082, "QSO_QRC": 0.490}


def parse_tables(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            v = node.value
            if isinstance(v, ast.Call) and getattr(v.func, "attr", "") == "array":
                v = v.args[0]
            try:
                out[name] = ast.literal_eval(v)
            except ValueError:
                pass
    return out


def mu_curve(Efun, H0, z):
    c = H.C_KMS
    out = []
    for zi in np.atleast_1d(z):
        val, _ = quad(lambda zp: 1.0 / Efun(zp), 0, zi)
        out.append(5 * math.log10(c / H0 * (1 + zi) * val * 1e6) - 5)
    return np.array(out)


def red_chi2(obs, err, mu, marg):
    w = 1 / err**2
    off = np.sum((obs - mu) * w) / np.sum(w) if marg else 0.0
    return float(np.sum(((obs - mu - off) / err) ** 2) / (len(obs) - 1)), float(off)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--stress", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "repo_claims_results.json"))
    a = ap.parse_args()
    T = parse_tables(a.stress)
    P = np.array(T["PANTHEON"], float); Qs = np.array(T["QUASARS"], float)
    CCt = np.array(T["CHRONOMETERS"], float); DB = T["DESI_BAO"]
    res = {"parsed_constants": {k: T[k] for k in ("H0_l", "rd_l", "Omega_m_l", "H0_q", "rd_q",
                                                  "Omega_m_q", "w0_q", "wa_q") if k in T}}

    # ---- 1. re-implementation of the current script on its own tables
    om_l, H0_l = T["Omega_m_l"], T["H0_l"]
    om_q, H0_q, w0, wa = T["Omega_m_q"], T["H0_q"], T["w0_q"], T["wa_q"]
    E_l = lambda z: math.sqrt(om_l * (1 + z) ** 3 + T["Omega_de_l"])

    def E_q(z, om=om_q, w0=w0, wa=wa, ode=1.0 - om_q - 9.0e-5):
        ap_ = 1 / (1 + z)
        return math.sqrt(om * (1 + z) ** 3 + ode * ap_ ** (-3 * (1 + w0 + wa)) * math.exp(-3 * wa * (1 - ap_)))

    cur = {}
    for lab, tab in (("PP", P), ("QSO", Qs)):
        cur[lab + "_LCDM"] = red_chi2(tab[:, 1], tab[:, 2], mu_curve(E_l, H0_l, tab[:, 0]), True)[0]
        cur[lab + "_QRC"] = red_chi2(tab[:, 1], tab[:, 2], mu_curve(E_q, H0_q, tab[:, 0]), True)[0]
    res["current_script_reimplemented"] = cur

    # ---- 2. try to reconstruct the claimed numbers on the hard-coded tables
    candidates = {
        "LCDM Planck H0=67.4 Om=.315": (lambda z: E_l(z), 67.4),
        "LCDM H0=70 Om=.3": (lambda z: math.sqrt(0.3 * (1 + z) ** 3 + 0.7), 70.0),
        "CPL stress (-0.77,-0.21) H0=68.3": (lambda z: E_q(z), 68.3),
        "CPL empirical (-0.77,+0.20) Om=.31 H0=70": (lambda z: E_q(z, 0.31, -0.77, 0.20, 0.69), 70.0),
        "CPL DESI (-0.827,-0.75) Om=.311 H0=68.3": (lambda z: E_q(z, 0.311, -0.827, -0.75), 68.3),
        "Rh=ct H0=70.92": (lambda z: 1 + z, 70.92),
        "Rh=ct H0=74.70": (lambda z: 1 + z, 74.70),
        "EdS H0=70": (lambda z: (1 + z) ** 1.5, 70.0),
    }
    offsets = {"none": 0.0, "+0.219": 0.2196, "-0.219": -0.2196}
    trials = []
    for (lab, (Ef, H0)), (olab, off), marg in itertools.product(candidates.items(), offsets.items(), (True, False)):
        for tlab, tab in (("PP", P), ("QSO", Qs)):
            mu = mu_curve(Ef, H0, tab[:, 0]) + off
            r, _ = red_chi2(tab[:, 1], tab[:, 2], mu, marg)
            trials.append({"table": tlab, "model": lab, "offset": olab, "zero_point_marginalised": marg, "red_chi2": r})
    matches = []
    for t in trials:
        for ck, cv in CLAIMS.items():
            if ck.startswith(t["table"]) and abs(t["red_chi2"] - cv) < 0.0015:
                matches.append({"claim": ck, **t})
    res["reconstruction_trials"] = len(trials)
    res["reconstruction_matches"] = matches
    res["closest_per_claim"] = {}
    for ck, cv in CLAIMS.items():
        tt = [t for t in trials if ck.startswith(t["table"])]
        best = min(tt, key=lambda t: abs(t["red_chi2"] - cv))
        res["closest_per_claim"][ck] = {"claimed": cv, **best}

    # ---- 3. hard-coded 'Pantheon+' table vs real Pantheon+ binned
    D = a.data
    d = np.genfromtxt(os.path.join(D, "PantheonPlusSH0ES.dat"), names=True, dtype=None, encoding="ascii")
    m = (d["zHD"] > 0.01) & (d["IS_CALIBRATOR"] == 0)
    z, mb, e = d["zHD"][m], d["m_b_corr"][m], d["m_b_corr_err_DIAG"][m]
    zt = P[:, 0]
    edges = np.concatenate([[0.01], 0.5 * (zt[1:] + zt[:-1]), [2.3]])
    rows = []
    for i, zc in enumerate(zt):
        sel = (z >= edges[i]) & (z < edges[i + 1])
        w = 1 / e[sel] ** 2
        mu_bin = np.sum(w * mb[sel]) / np.sum(w) + 19.253  # SH0ES-calibrated M (Brout+22)
        zbar = np.sum(w * z[sel]) / np.sum(w)
        rows.append({"z_table": zc, "n_SN": int(sel.sum()), "z_mean_bin": round(float(zbar), 4),
                     "mu_table": P[i, 1], "err_table": P[i, 2],
                     "mu_real_bin(M=-19.253)": round(float(mu_bin), 3),
                     "err_real_bin_diag": round(float(1 / math.sqrt(np.sum(w))), 3)})
    diff = np.array([r["mu_table"] - r["mu_real_bin(M=-19.253)"] for r in rows])
    # compare at the table's own z using the real-data LCDM best fit shape (removes bin-centroid effect)
    Ebest = lambda zz: math.sqrt(0.333 * (1 + zz) ** 3 + 0.667)
    mu_fit_tab = mu_curve(Ebest, 73.5, zt)
    mu_fit_bin = mu_curve(Ebest, 73.5, np.array([r["z_mean_bin"] for r in rows]))
    corrected = diff - (mu_fit_tab - mu_fit_bin)
    res["pantheon_table_vs_real"] = {
        "rows": rows,
        "table_minus_real_after_centroid_correction": [round(float(x), 3) for x in corrected],
        "mean_offset": float(np.mean(corrected)),
        "rms_after_removing_offset": float(np.std(corrected)),
        "max_abs_after_removing_offset": float(np.max(np.abs(corrected - np.mean(corrected)))),
        "note": "table errors 0.05 to 0.16 mag versus real diagonal bin errors; real data have 1580 SNe and a full covariance"}

    # ---- 4. chronometer table vs downloaded Moresco BC03 table
    zc, Hc, ec = H.load_cc(os.path.join(D, "HzTable_MM_BC03.dat"))
    cmp_ = []
    for zi, hi, ei in zip(zc, Hc, ec):
        j = int(np.argmin(np.abs(CCt[:, 0] - zi)))
        cmp_.append({"z_real": float(zi), "H_real": float(hi), "err_real": float(ei),
                     "z_table": float(CCt[j, 0]), "H_table": float(CCt[j, 1]), "err_table": float(CCt[j, 2]),
                     "matched": bool(abs(CCt[j, 0] - zi) < 0.006)})
    res["cc_table_vs_real"] = {"n_table": int(len(CCt)), "n_downloaded": int(len(zc)), "rows": cmp_,
                               "n_table_rows_not_in_downloaded_file": int(len(CCt) - sum(r["matched"] for r in cmp_))}

    # ---- 5. DESI table vs real DESI DR1 file
    zb, vb, kb, Cb = H.load_desi(os.path.join(D, "desi_2024_gaussian_bao_ALL_GCcomb_mean.txt"),
                                 os.path.join(D, "desi_2024_gaussian_bao_ALL_GCcomb_cov.txt"))
    real = {}
    for zi, vi, ki, si in zip(zb, vb, kb, np.sqrt(np.diag(Cb))):
        real.setdefault(round(float(zi), 3), {})[ki] = (round(float(vi), 3), round(float(si), 3))
    drows = []
    for zr, dm, dme, dh, dhe in DB:
        r = real.get(round(zr, 3), {})
        drows.append({"z": zr, "table_DM": (dm, dme), "table_DH": (dh, dhe),
                      "real_DM": r.get("DM_over_rs"), "real_DH": r.get("DH_over_rs"), "real_DV": r.get("DV_over_rs")})
    res["desi_table_vs_real"] = drows

    # ---- 6. the stress-test chi2 for DESI as coded (diagonal, its own table) vs real full covariance
    bao = H.BAOLike(zb, vb, kb, Cb)
    out6 = {}
    for lab, bg, hrd in (("Planck LCDM prereg", H.Background(H.E_lcdm(0.315)), 0.674 * 147.09),
                         ("stress QRC prereg", H.Background(H.E_cpl(0.311, -0.77, -0.21, orad=9e-5)), 0.683 * 144.7)):
        c, _ = bao.chi2(bg, hrd_fixed=hrd)
        out6[lab] = {"chi2_real_fullcov": c, "N": 12, "chi2_per_N": c / 12}
    res["desi_real_prereg"] = out6

    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps({"current": cur, "matches": matches,
                      "closest": {k: (v["model"], v["offset"], v["zero_point_marginalised"], round(v["red_chi2"], 3))
                                  for k, v in res["closest_per_claim"].items()},
                      "pp_table_vs_real": {k: v for k, v in res["pantheon_table_vs_real"].items() if k != "rows"},
                      "cc_unmatched": res["cc_table_vs_real"]["n_table_rows_not_in_downloaded_file"],
                      "desi_prereg": out6}, indent=1, default=float))


if __name__ == "__main__":
    main()
