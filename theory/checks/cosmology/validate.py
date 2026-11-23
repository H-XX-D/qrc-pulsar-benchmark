#!/usr/bin/env python3
"""Step 1: validate the harness against published flat-LCDM results.

Usage: python3 -I -B validate.py --data DIR [--out validation_results.json]
Exit status 0 only if every reproduction passes and every negative control fails.
Acceptance tolerances were fixed before the first run and are stated in CRITERIA.
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import cosmo_harness as H  # noqa: E402

# Published references and tolerances (fixed before running).
CRITERIA = {
    "PantheonPlus": {"ref": "Brout et al. 2022, Pantheon+ only, flat LCDM: Om = 0.334 +- 0.018",
                     "Om": (0.334, 0.010), "sOm": (0.018, 0.004)},
    "PantheonPlusSH0ES": {"ref": "Brout et al. 2022, Pantheon+ & SH0ES, flat LCDM: H0 = 73.6 +- 1.1",
                          "H0": (73.6, 0.5), "sH0": (1.1, 0.3), "Om": (0.334, 0.010)},
    "DESI-DR1": {"ref": "DESI 2024 VI (arXiv:2404.03002), BAO only flat LCDM: Om = 0.295 +- 0.015, h r_d = 101.8 +- 1.3 Mpc",
                 "Om": (0.295, 0.008), "sOm": (0.015, 0.004), "hrd": (101.8, 1.0), "shrd": (1.3, 0.4)},
    "DES-Dovekie": {"ref": "Popovic et al. 2025 (arXiv:2511.07517) abstract: Om = 0.330 +- 0.015 flat LCDM (SN-only inferred, not stated)",
                    "Om": (0.330, 0.010), "sOm": (0.015, 0.004)},
    "EdS_negative_control": {"ref": "Einstein-de Sitter (Om=1) must be rejected: delta chi2 vs LCDM > 25 on SN and BAO"},
}


def within(val, spec):
    return abs(val - spec[0]) <= spec[1]


# ----------------------------------------------------------------- builders
def build_pp(d, C, mutant=None):
    z = d["zHD"]; cal = d["IS_CALIBRATOR"] == 1
    mask = (z > 0.01) & (~cal)
    if mutant == "subset_of_inverse":
        cinv = np.linalg.inv(C)[np.ix_(mask, mask)]
    elif mutant == "diag_only":
        cinv = np.diag(1.0 / np.diag(C)[mask])
    else:
        cinv = np.linalg.inv(C[np.ix_(mask, mask)])
    cinv = 0.5 * (cinv + cinv.T)
    return H.SNLike("PantheonPlus", z[mask], d["zHEL"][mask], d["m_b_corr"][mask], cinv,
                    one_plus_z=(mutant != "no_1pz"))


def build_shoes(d, C):
    z = d["zHD"]; cal = d["IS_CALIBRATOR"] == 1
    mask = (z > 0.01) | cal
    cinv = np.linalg.inv(C[np.ix_(mask, mask)])
    cinv = 0.5 * (cinv + cinv.T)
    return H.SNLike("PantheonPlusSH0ES", z[mask], d["zHEL"][mask], d["m_b_corr"][mask], cinv,
                    is_cal=cal[mask], cal_mu=d["CEPH_DIST"][mask])


def build_des(data, A, mutant=None):
    cinv = np.linalg.inv(A) if mutant == "treat_as_covariance" else A
    return H.SNLike("DES-Dovekie", data["zHD"], data["zHEL"], data["MU"], cinv)


def build_desi(z, v, kind, C, mutant=None):
    kind = list(kind)
    if mutant == "swap_DM_DH":
        kind = [{"DM_over_rs": "DH_over_rs", "DH_over_rs": "DM_over_rs"}.get(k, k) for k in kind]
    return H.BAOLike(z, v, kind, C)


# ------------------------------------------------------------ reproductions
OM_GRID = np.arange(0.10, 0.6001, 0.0005)


def sn_lcdm(like):
    chi = np.array([like.chi2(H.Background(H.E_lcdm(om)))[0] for om in OM_GRID])
    mean, sd = H.grid_posterior_1d(OM_GRID, chi)
    i = int(np.argmin(chi))
    return {"Om": mean, "sOm": sd, "Om_best": float(OM_GRID[i]), "chi2_min": float(chi[i]),
            "N": like.N, "chi2_per_dof": float(chi[i] / (like.N - 2))}


def shoes_lcdm(like):
    chis, h0s, sh0s = [], [], []
    for om in OM_GRID:
        c, nu = like.chi2(H.Background(H.E_lcdm(om)))
        chis.append(c); h0s.append(nu["H0"]); sh0s.append(nu["sigma_H0_cond"])
    chis, h0s, sh0s = map(np.array, (chis, h0s, sh0s))
    w = np.exp(-0.5 * (chis - chis.min())); w /= w.sum()
    mH0 = float(np.sum(w * h0s))
    sH0 = float(math.sqrt(np.sum(w * (sh0s**2 + h0s**2)) - mH0**2))
    mOm, sOm = H.grid_posterior_1d(OM_GRID, chis)
    i = int(np.argmin(chis))
    return {"H0": mH0, "sH0": sH0, "Om": mOm, "sOm": sOm, "chi2_min": float(chis[i]),
            "N": like.N, "chi2_per_dof": float(chis[i] / (like.N - 3))}


def desi_lcdm(like):
    hrd = np.arange(80.0, 125.0001, 0.01)
    alpha = H.DH100 / hrd
    oms = np.arange(0.15, 0.5001, 0.0005)
    chi2 = np.empty((len(oms), len(hrd)))
    for i, om in enumerate(oms):
        t = like.template(H.Background(H.E_lcdm(om)))
        ct = like.cinv @ t
        chi2[i] = like.dcd - 2 * alpha * (like.d @ ct) + alpha**2 * (t @ ct)
    P = np.exp(-0.5 * (chi2 - chi2.min()))
    P /= P.sum()
    pOm = P.sum(axis=1); phrd = P.sum(axis=0)
    mOm = float(np.sum(oms * pOm)); sOm = float(math.sqrt(np.sum((oms - mOm) ** 2 * pOm)))
    mh = float(np.sum(hrd * phrd)); sh = float(math.sqrt(np.sum((hrd - mh) ** 2 * phrd)))
    i, j = np.unravel_index(np.argmin(chi2), chi2.shape)
    return {"Om": mOm, "sOm": sOm, "hrd": mh, "shrd": sh, "chi2_min": float(chi2[i, j]),
            "Om_best": float(oms[i]), "hrd_best": float(hrd[j]), "N": like.N,
            "chi2_per_dof": float(chi2[i, j] / (like.N - 2))}


def check(name, res):
    crit = CRITERIA[name]
    out = {}
    for key, spec in crit.items():
        if key == "ref":
            continue
        out[key] = {"got": res[key], "ref": spec[0], "tol": spec[1], "pass": within(res[key], spec)}
    return all(v["pass"] for v in out.values()), out


def eds_control(likes):
    rows = {}
    bg_eds = H.Background(H.E_eds())
    for L in likes:
        if isinstance(L, H.BAOLike):
            c_eds = L.chi2(bg_eds)[0]
            r = desi_lcdm(L); c_l = r["chi2_min"]; k_l, k_e = 2, 1
        else:
            c_eds = L.chi2(bg_eds)[0]
            r = sn_lcdm(L); c_l = r["chi2_min"]; k_l, k_e = 2, 1
        from scipy.stats import chi2 as chi2dist
        rows[L.name] = {"chi2_EdS": c_eds, "chi2_LCDM": c_l, "delta_chi2": c_eds - c_l,
                        "p_EdS": float(chi2dist.sf(c_eds, L.N - k_e)),
                        "rejected": (c_eds - c_l) > 25.0}
    return rows


def eds_cc(cc):
    from scipy.stats import chi2 as chi2dist
    c_eds = cc.chi2(H.Background(H.E_eds()))[0]
    chi = np.array([cc.chi2(H.Background(H.E_lcdm(om)))[0] for om in OM_GRID])
    return {"chi2_EdS": c_eds, "chi2_LCDM": float(chi.min()), "delta_chi2": float(c_eds - chi.min()),
            "p_EdS": float(chi2dist.sf(c_eds, cc.N - 1)),
            "note": "CC alone is reported, not used as a pass/fail control"}


# ---------------------------------------------------------- analytic checks
def analytic_checks():
    z = np.linspace(0.01, 2.3, 50)
    out = {}
    bg = H.Background(H.E_lcdm(1.0, orad=0.0))
    out["EdS_DM_vs_closed_form"] = float(np.max(np.abs(bg.dm(z) - 2 * (1 - 1 / np.sqrt(1 + z)))))
    bg = H.Background(np.ones_like(H.ZGRID))
    out["deSitter_DM_vs_z"] = float(np.max(np.abs(bg.dm(z) - z)))
    bg = H.Background(H.E_rhct())
    out["Rhct_DM_vs_ln1pz"] = float(np.max(np.abs(bg.dm(z) - np.log1p(z))))
    # agegraphic with negligible n reproduces exact matter+radiation conformal time
    om = 0.3
    sol = H._age_solve(1e-6, om)
    out["age_eta1_vs_exact"] = float(abs(math.exp(sol.sol(0.0)[0]) - H.eta_matter_radiation(1.0, om)))
    # early-time attractors for n = 2.5
    n = 2.5
    w, oq = H.agegraphic_w_of_a(n, om, [1e-6, 1e-2])
    out["age_rad_era_w_plus_1/3"] = float(w[0] + 1 / 3)
    out["age_rad_era_Oq/(n^2a^2)-1"] = float(oq[0] / (n * n * 1e-12) - 1)
    # First version of this check compared w(a=1e-2) with the pure matter-era value
    # -2/3 and FAILED (diff 0.057): at a=1e-2 radiation still shifts eta by
    # O(sqrt(a_eq/a)). Replaced by the exact matter+radiation expression.
    a2 = 1e-2
    Emr = math.sqrt(om * a2**-3 + H.OMEGA_R * a2**-4)
    w_mr = -1 + 2 / (3 * a2 * Emr * H.eta_matter_radiation(a2, om))
    out["age_a1e-2_w_minus_matrad_exact"] = float(w[1] - w_mr)
    out["info_age_a1e-2_w_plus_2/3"] = float(w[1] + 2 / 3)
    bgq = H.agegraphic_background(2.56)
    out["age_wa_numeric_minus_analytic"] = float(bgq.extra["wa"] - bgq.extra["wa_analytic"])
    out["age_E(z=0)-1"] = float(bgq.E[0] - 1.0)
    # retarded trace: compare ODE implementation with a port of the repo's trapz loop
    out.update(retarded_port_check())
    tol = {"EdS_DM_vs_closed_form": 1e-6, "deSitter_DM_vs_z": 1e-9, "Rhct_DM_vs_ln1pz": 1e-7,
           "age_eta1_vs_exact": 1e-6, "age_rad_era_w_plus_1/3": 1e-3,
           "age_rad_era_Oq/(n^2a^2)-1": 1e-2, "age_a1e-2_w_minus_matrad_exact": 1e-3,
           "age_wa_numeric_minus_analytic": 1e-4, "age_E(z=0)-1": 1e-6,
           "rt_port_rel_diff_a0.5": 2e-2, "rt_port_rel_diff_a0.8": 2e-2}
    passed = {k: abs(out[k]) <= tol[k] for k in tol}
    return out, tol, passed


def retarded_port_check():
    """Re-implementation of qrc_retarded_trace_dark_energy_validation.compute_cosmic_evolution
    (not imported; logic copied) versus the harness ODE (as-written variant)."""
    om, orad, tau = 0.315, 9.2e-5, 1.35
    a = np.geomspace(5e-5, 1.0, 600)
    Hh = np.sqrt(orad * a**-4 + om * a**-3 + (1.0 - om))
    dt_da = 1.0 / (a * Hh)
    t = np.zeros_like(a)
    for i in range(1, len(a)):
        t[i] = np.trapezoid(dt_da[:i + 1], a[:i + 1])
    rm = om * a**-3; rr = orad * a**-4
    src = rm / (rm + 2 * rr) * np.sqrt(rm)
    rho = np.zeros_like(a)
    for i in range(len(a)):
        rho[i] = np.trapezoid(np.exp(-(t[i] - t[:i + 1]) / tau) * src[:i + 1], t[:i + 1])
    rho = rho / rho[-1] * (1 - om)
    # repo's own w0, wa estimate
    m = a >= 0.6
    w_eff = -1.0 - (a[m] / (3.0 * rho[m])) * np.gradient(rho[m], a[m])
    repo_w0 = float(w_eff[-1]); repo_wa = float(-np.gradient(w_eff, a[m])[-1])
    bg = H.retarded_trace_background(om, tau, selfconsistent=False, orad=orad)
    out = {}
    for av in (0.5, 0.8):
        z = 1 / av - 1
        mine = bg.e(z) ** 2 - om * av**-3 - orad * av**-4
        theirs = np.interp(av, a, rho)
        out[f"rt_port_rel_diff_a{av}"] = float(mine / theirs - 1)
    out["rt_repo_algorithm_w0"] = repo_w0
    out["rt_repo_algorithm_wa"] = repo_wa
    out["rt_harness_w0"] = bg.extra["w0"]
    out["rt_harness_wa"] = bg.extra["wa"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "validation_results.json"))
    a = ap.parse_args()
    D = a.data
    t0 = time.time()
    d, C = H.load_pantheon(os.path.join(D, "PantheonPlusSH0ES.dat"),
                           os.path.join(D, "PantheonPlusSH0ES_STAT+SYS.cov"))
    des, A = H.load_des(os.path.join(D, "DES-Dovekie_HD.csv"), os.path.join(D, "STAT+SYS.npz"))
    zb, vb, kb, Cb = H.load_desi(os.path.join(D, "desi_2024_gaussian_bao_ALL_GCcomb_mean.txt"),
                                 os.path.join(D, "desi_2024_gaussian_bao_ALL_GCcomb_cov.txt"))
    cc = H.CCLike(*H.load_cc(os.path.join(D, "HzTable_MM_BC03.dat")))

    report = {"criteria": CRITERIA, "results": {}, "mutants": {}}
    allpass = True

    # data integrity facts
    Cdes = np.linalg.inv(A)
    report["data_facts"] = {
        "pantheon_rows": int(len(d)),
        "pantheon_used_hubbleflow": int(((d["zHD"] > 0.01) & (d["IS_CALIBRATOR"] == 0)).sum()),
        "pantheon_used_shoes_mode": int(((d["zHD"] > 0.01) | (d["IS_CALIBRATOR"] == 1)).sum()),
        "des_rows": int(len(des["MU"])),
        "des_diag_cov_over_MUERR2+MUERR_SYS2_median": float(np.median(np.diag(Cdes) / (des["MUERR"]**2 + des["MUERR_SYS"]**2))),
        "des_invcov_min_eigenvalue": float(np.linalg.eigvalsh(A).min()),
        "desi_points": int(len(vb)), "cc_points": int(cc.N),
    }

    pp = build_pp(d, C)
    r = sn_lcdm(pp); ok, chk = check("PantheonPlus", r)
    report["results"]["PantheonPlus"] = {"fit": r, "checks": chk, "pass": ok}; allpass &= ok

    sh = build_shoes(d, C)
    r = shoes_lcdm(sh); ok, chk = check("PantheonPlusSH0ES", r)
    report["results"]["PantheonPlusSH0ES"] = {"fit": r, "checks": chk, "pass": ok}; allpass &= ok

    bao = build_desi(zb, vb, kb, Cb)
    r = desi_lcdm(bao); ok, chk = check("DESI-DR1", r)
    report["results"]["DESI-DR1"] = {"fit": r, "checks": chk, "pass": ok}; allpass &= ok

    dl = build_des(des, A)
    r = sn_lcdm(dl); ok, chk = check("DES-Dovekie", r)
    report["results"]["DES-Dovekie"] = {"fit": r, "checks": chk, "pass": ok}; allpass &= ok

    eds = eds_control([pp, dl, bao])
    ok = all(v["rejected"] for v in eds.values())
    eds["CC_info"] = eds_cc(cc)
    report["results"]["EdS_negative_control"] = {"rows": eds, "pass": ok}; allpass &= ok

    vals, tol, passed = analytic_checks()
    report["analytic_checks"] = {"values": vals, "tolerances": tol, "pass": passed}
    allpass &= all(passed.values())

    # Mutants: deliberately wrong harnesses. Each must FAIL the same criteria.
    mut = {}
    r = sn_lcdm(build_pp(d, C, "no_1pz")); ok, chk = check("PantheonPlus", r)
    mut["PP_dL_without_(1+z)"] = {"fit": r, "caught": not ok}
    r = sn_lcdm(build_pp(d, C, "diag_only")); ok, chk = check("PantheonPlus", r)
    mut["PP_diagonal_errors_only"] = {"fit": r, "caught": not ok}
    r = sn_lcdm(build_pp(d, C, "subset_of_inverse")); ok, chk = check("PantheonPlus", r)
    mut["PP_cut_applied_to_inverse_cov"] = {"fit": r, "caught": not ok}
    r = desi_lcdm(build_desi(zb, vb, kb, Cb, "swap_DM_DH")); ok, chk = check("DESI-DR1", r)
    mut["DESI_DM_DH_labels_swapped"] = {"fit": r, "caught": not ok}
    r = sn_lcdm(build_des(des, A, "treat_as_covariance")); ok, chk = check("DES-Dovekie", r)
    mut["DES_npz_read_as_covariance"] = {"fit": r, "caught": not ok}
    report["mutants"] = mut
    report["all_mutants_caught"] = all(v["caught"] for v in mut.values())
    report["ALL_PASS"] = bool(allpass)
    report["runtime_s"] = round(time.time() - t0, 1)

    with open(a.out, "w") as f:
        json.dump(report, f, indent=1, default=float)

    # concise console summary
    for k, v in report["results"].items():
        print(f"{k:22s} pass={v['pass']}")
        if "fit" in v:
            print("   ", {kk: round(vv, 4) for kk, vv in v["fit"].items() if isinstance(vv, float)})
        else:
            for kk, vv in v["rows"].items():
                print("   ", kk, {a_: (round(b_, 3) if isinstance(b_, float) else b_) for a_, b_ in vv.items()})
    print("analytic:", {k: f"{v:.2e}" for k, v in vals.items()})
    print("analytic pass:", all(passed.values()), [k for k, v in passed.items() if not v])
    for k, v in mut.items():
        print(f"mutant {k:34s} caught={v['caught']}  Om={v['fit'].get('Om', float('nan')):.4f} sOm={v['fit'].get('sOm', float('nan')):.4f}")
    print("ALL_PASS", allpass, "runtime", report["runtime_s"], "s")
    sys.exit(0 if allpass else 1)


if __name__ == "__main__":
    main()
