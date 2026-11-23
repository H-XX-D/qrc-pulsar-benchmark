#!/usr/bin/env python3
"""Step 2: fit LCDM, CPL and every QRC expansion law to real data.

Usage: python3 -I -B run_models.py --data DIR [--out model_results.json]
Refuses to run unless validation_results.json exists with ALL_PASS true.
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from scipy.stats import chi2 as chi2dist  # noqa: E402
import cosmo_harness as H  # noqa: E402
import validate as V  # noqa: E402

Q_VOID = 0.1066
DMU_BACKREACTION = 5 * math.log10(1 + Q_VOID)       # 0.2196 mag, as in the repo
T0_GYR = 13.787                                      # repo's input age
H0_INV_T0 = H.GYR_PER_INV_H0 / T0_GYR               # 70.92 km/s/Mpc


def make_models():
    lin = np.linspace
    M = H.Model
    bg = H.Background
    models = [
        M("LCDM", ["Om"], [(0.01, 0.99)], lambda om: bg(H.E_lcdm(om)), grid=[lin(0.05, 0.7, 27)],
          description="flat LCDM baseline"),
        M("CPL_w0wa", ["Om", "w0", "wa"], [(0.01, 0.99), (-3, 1), (-8, 4)],
          lambda om, w0, wa: bg(H.E_cpl(om, w0, wa)),
          grid=[lin(0.2, 0.45, 6), lin(-1.4, -0.4, 6), lin(-3.0, 1.5, 6)],
          description="flat w0waCDM"),
        M("QRC_agegraphic_strict", ["n"], [(0.6, 12.0)],
          lambda n: H.agegraphic_background(n), grid=[lin(0.8, 8.0, 37)],
          description="Wei-Cai new agegraphic DE, eta = conformal time from a=0; Om derived from n"),
        M("QRC_agegraphic_n=1", [], [], lambda: H.agegraphic_background(1.0),
          description="repo's 'parameter-free c_QRC == 1.0' agegraphic coupling, solved self-consistently"),
        M("QRC_agegraphic_relaxed", ["n", "Om"], [(0.6, 30.0), (0.01, 0.9)],
          lambda n, om: H.agegraphic_background(n, om, relaxed=True),
          grid=[lin(1.5, 9.0, 11), lin(0.2, 0.45, 6)],
          description="sensitivity variant: eta offset >= 0 so that n and Om are both free (not Wei-Cai)"),
        M("Rh=ct", [], [], lambda: bg(H.E_rhct()), description="Melia R_h = ct, H = H0(1+z)"),
        M("Rh=ct+backreaction", [], [], lambda: bg(H.E_rhct()), sn_offset=DMU_BACKREACTION,
          description="R_h=ct with constant Delta mu = 5 log10(1.1066) added to SN model"),
        M("QRC_retarded_trace_as_written", ["Om"], [(0.01, 0.99)],
          lambda om: H.retarded_trace_background(om, 1.35, False), grid=[lin(0.05, 0.7, 27)],
          description="repo algorithm: rho_DE shape computed on an LCDM background, tau = 1.35/H0"),
        M("QRC_retarded_trace_selfconsistent", ["Om"], [(0.01, 0.99)],
          lambda om: H.retarded_trace_background(om, 1.35, True), grid=[lin(0.05, 0.7, 27)],
          description="same kernel, background solved self-consistently, tau = 1.35/H0"),
        M("QRC_retarded_trace_sc_tau_free", ["Om", "tau"], [(0.01, 0.99), (0.05, 30.0)],
          lambda om, tau: H.retarded_trace_background(om, tau, True),
          grid=[lin(0.2, 0.45, 6), np.geomspace(0.2, 10, 6)],
          description="sensitivity variant with tau free"),
        M("QRC_holographic_Hubble_cutoff", [], [], lambda: bg(H.E_eds()),
          description="rho_DE = 3 c^2 Mp^2 H^2: c=1 forces Om=0; c<1 gives H ~ a^-3/2 (EdS shape)"),
        M("QRC_CPL_gate7(-0.78,-0.23)", ["Om"], [(0.01, 0.99)],
          lambda om: bg(H.E_cpl(om, -0.78, -0.23)), grid=[lin(0.05, 0.7, 27)],
          description="gate7 claim w0=-0.78, wa=-0.23, Om free"),
        M("QRC_CPL_stress(-0.77,-0.21,Om=.311)", [], [],
          lambda: bg(H.E_cpl(0.311, -0.77, -0.21, orad=9.0e-5)),
          description="multi-survey stress test QRC model, nuisances free"),
        M("QRC_CPL_empirical(-0.77,+0.20,Om=.31)", [], [],
          lambda: bg(H.E_cpl(0.31, -0.77, 0.20, orad=0.0)),
          description="qrc_empirical_data_testing QRC model, nuisances free"),
    ]
    return models


def build_likes(D):
    d, C = H.load_pantheon(os.path.join(D, "PantheonPlusSH0ES.dat"),
                           os.path.join(D, "PantheonPlusSH0ES_STAT+SYS.cov"))
    des, A = H.load_des(os.path.join(D, "DES-Dovekie_HD.csv"), os.path.join(D, "STAT+SYS.npz"))
    zb, vb, kb, Cb = H.load_desi(os.path.join(D, "desi_2024_gaussian_bao_ALL_GCcomb_mean.txt"),
                                 os.path.join(D, "desi_2024_gaussian_bao_ALL_GCcomb_cov.txt"))
    cc = H.CCLike(*H.load_cc(os.path.join(D, "HzTable_MM_BC03.dat")))
    return {"pp": V.build_pp(d, C), "des": V.build_des(des, A), "desi": V.build_desi(zb, vb, kb, Cb),
            "cc": cc, "shoes": V.build_shoes(d, C)}


COMBOS = {
    "PantheonPlus": ["pp"], "DES-Dovekie": ["des"], "DESI-DR1": ["desi"], "CC": ["cc"],
    "PP+DESI": ["pp", "desi"], "DES+DESI": ["des", "desi"],
    "PP+DESI+CC": ["pp", "desi", "cc"], "DES+DESI+CC": ["des", "desi", "cc"],
}


def strip(r):
    return {k: v for k, v in r.items() if not k.startswith("_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "model_results.json"))
    a = ap.parse_args()
    vpath = os.path.join(HERE, "validation_results.json")
    with open(vpath) as f:
        if not json.load(f).get("ALL_PASS"):
            sys.exit("validation has not passed; refusing to run step 2")
    t0 = time.time()
    L = build_likes(a.data)
    models = make_models()
    out = {"constants": {"Omega_r": H.OMEGA_R, "Delta_mu_backreaction": DMU_BACKREACTION,
                         "H0_from_1/t0": H0_INV_T0}, "fits": {}, "profiles": {}, "special": {}}

    # ---------------------------------------------------------------- fits
    for cname, keys in COMBOS.items():
        likes = [L[k] for k in keys]
        res = {}
        for m in models:
            r = H.fit(m, likes)
            res[m.name] = r
        base = res["LCDM"]
        for m in models:
            r = res[m.name]
            r["dAIC"] = r["AIC"] - base["AIC"]
            r["dBIC"] = r["BIC"] - base["BIC"]
            r["p_value"] = float(chi2dist.sf(r["chi2"], max(r["dof"], 1)))
        # DESI DH/rd pulls at best fit
        if "desi" in keys:
            for m in models:
                r = res[m.name]
                if "_bg" in r:
                    pred, pull = L["desi"].residuals(r["_bg"], r["parts"]["DESI-DR1"]["hrd"])
                    r["desi_pulls"] = {f"{k}@{z}": round(float(p), 2)
                                       for z, k, p in zip(L["desi"].z, L["desi"].kind, pull)}
        out["fits"][cname] = {k: strip(v) for k, v in res.items()}
        print(f"[{time.time()-t0:6.1f}s] {cname}: " + ", ".join(
            f"{k.split('(')[0]}={v['chi2']:.1f}" for k, v in res.items()), flush=True)

    # ------------------------------------------------------------ profiles
    lcdm, cpl, age = models[0], models[1], models[2]
    for cname in ["PantheonPlus", "DES-Dovekie", "DESI-DR1", "PP+DESI", "DES+DESI",
                  "PP+DESI+CC", "DES+DESI+CC"]:
        likes = [L[k] for k in COMBOS[cname]]
        prof = {}
        oms = np.linspace(0.15, 0.55, 161)
        c = H.profile_1d(lcdm, likes, 0, oms, [0.3])
        prof["LCDM_Om_1sigma"] = H.interval_from_profile(oms, c)
        ns = np.linspace(1.2, 6.0, 97)
        c = H.profile_1d(age, likes, 0, ns, [2.5])
        lo, hi = H.interval_from_profile(ns, c)
        prof["agegraphic_n_1sigma"] = (lo, hi)
        prof["agegraphic_Om_at_n_1sigma_ends"] = (
            H.agegraphic_om_for_n(hi) if hi else None, H.agegraphic_om_for_n(lo) if lo else None)
        if cname in ("PP+DESI", "DES+DESI", "PP+DESI+CC", "DES+DESI+CC"):
            best = out["fits"][cname]["CPL_w0wa"]["params"]
            st = [best["Om"], best["w0"], best["wa"]]
            w0s = np.linspace(-1.3, -0.4, 46)
            c = H.profile_1d(cpl, likes, 1, w0s, st)
            prof["CPL_w0_1sigma"] = H.interval_from_profile(w0s, c)
            was = np.linspace(-3.5, 1.5, 51)
            c = H.profile_1d(cpl, likes, 2, was, st)
            prof["CPL_wa_1sigma"] = H.interval_from_profile(was, c)
            prof["CPL_wa_profile"] = {f"{w:.1f}": round(float(x - c.min()), 2) for w, x in zip(was[::5], c[::5])}
            prof["LCDM_minus_CPL_chi2"] = out["fits"][cname]["LCDM"]["chi2"] - out["fits"][cname]["CPL_w0wa"]["chi2"]
            # agegraphic predicted (w0,wa) point: chi2 relative to CPL best fit
            ag = out["fits"][cname]["QRC_agegraphic_strict"]["derived"]
            def f_at(w0, wa):
                g = lambda om: H.evaluate(cpl, likes, np.array([om, w0, wa]))[0]
                from scipy.optimize import minimize_scalar
                return minimize_scalar(g, bounds=(0.05, 0.6), method="bounded").fun
            prof["CPL_chi2_at_agegraphic_w0wa_minus_best"] = f_at(ag["w0"], ag["wa"]) - out["fits"][cname]["CPL_w0wa"]["chi2"]
            prof["CPL_chi2_at_agegraphic_w0_with_wa=-wa"] = f_at(ag["w0"], -ag["wa"]) - out["fits"][cname]["CPL_w0wa"]["chi2"]
        out["profiles"][cname] = prof
        print(f"[{time.time()-t0:6.1f}s] profiles {cname}: {prof.get('agegraphic_n_1sigma')} {prof.get('CPL_wa_1sigma')}", flush=True)

    # ------------------------------------------------- specific claim checks
    sp = out["special"]
    om0 = 0.315 - H.OMEGA_R
    n0 = H.agegraphic_n_for_om(om0)
    bg0 = H.agegraphic_background(n0)
    sp["agegraphic_at_OmegaDE0=0.685"] = {"n": n0, **bg0.extra}
    sp["agegraphic_n=1"] = H.agegraphic_background(1.0).extra
    a_vals = [1e-3, 0.1, 0.25, 0.5, 0.75, 1.0]
    w, oq = H.agegraphic_w_of_a(n0, bg0.extra["Om"], a_vals)
    sp["agegraphic_w_of_a(n=2.558)"] = {f"a={x}": round(float(v), 4) for x, v in zip(a_vals, w)}
    sp["repo_agegraphic_algorithm"] = repo_agegraphic_port()

    # R_h = ct absolute tests with SH0ES-calibrated SNe and chronometers
    shoes, cc = L["shoes"], L["cc"]
    bg_r = H.Background(H.E_rhct())
    bg_l = H.Background(H.E_lcdm(out["fits"]["PantheonPlus"]["LCDM"]["params"]["Om"]))
    rows = {}
    c, nu = shoes.chi2(bg_l); rows["LCDM(Om=PP best), H0 free"] = {"chi2": c, **nu}
    c, nu = shoes.chi2(bg_r); rows["Rh=ct, H0 free"] = {"chi2": c, **nu}
    for lab, h, off in [("Rh=ct, H0=70.92 fixed", H0_INV_T0 / 100, 0.0),
                        ("Rh=ct, H0=70.92, +0.219 mag", H0_INV_T0 / 100, DMU_BACKREACTION),
                        ("Rh=ct, H0=70.92*(1+Q/2)=74.70", H0_INV_T0 * (1 + Q_VOID / 2) / 100, 0.0),
                        ("Rh=ct, H0=70.92*(1+Q/2), +0.219 mag", H0_INV_T0 * (1 + Q_VOID / 2) / 100, DMU_BACKREACTION),
                        ("Rh=ct, H0 free, +0.219 mag", None, DMU_BACKREACTION)]:
        c, nu = shoes.chi2(bg_r, offset=off, h_fixed=h)
        rows[lab] = {"chi2": c, **nu}
    base = rows["LCDM(Om=PP best), H0 free"]["chi2"]
    for v in rows.values():
        v["delta_chi2_vs_LCDM"] = v["chi2"] - base
        v["N"] = shoes.N
    sp["PantheonPlusSH0ES_absolute"] = rows

    ccrows = {}
    for lab, bgx in [("LCDM", H.Background(H.E_lcdm(out["fits"]["CC"]["LCDM"]["params"]["Om"]))),
                     ("Rh=ct", bg_r)]:
        c, nu = cc.chi2(bgx); ccrows[lab + ", H0 free"] = {"chi2": c, **nu}
    for lab, h in [("Rh=ct, H0=70.92", H0_INV_T0 / 100), ("Rh=ct, H0=74.70 (1+Q/2)", H0_INV_T0 * (1 + Q_VOID / 2) / 100)]:
        c, nu = cc.chi2(bg_r, h_fixed=h); ccrows[lab] = {"chi2": c, **nu}
    # H0 uncertainty for Rh=ct from CC (linear, exact)
    E = cc.z + 1.0
    sH0 = 1.0 / math.sqrt(np.sum(E * E / cc.err**2))
    ccrows["Rh=ct, H0 free"]["sigma_H0"] = sH0
    ccrows["Rh=ct, H0 free"]["t0_Gyr=1/H0"] = H.GYR_PER_INV_H0 / ccrows["Rh=ct, H0 free"]["H0"]
    for lab, v in ccrows.items():
        v["N"] = cc.N
        v["k"] = (1 if "free" in lab else 0) + (1 if lab.startswith("LCDM") else 0)
        v["p_value"] = float(chi2dist.sf(v["chi2"], cc.N - v["k"]))
    sp["CC_absolute"] = ccrows
    sp["H0_1/t0"] = {"t0_Gyr_input": T0_GYR, "H0": H0_INV_T0,
                     "note": "13.787 Gyr is the Planck 2018 LCDM-derived age (H0=67.4, Om=0.315); in LCDM H0*t0=0.95"}

    # stress-test preregistered absolute model vs Planck-LCDM preregistered, with fixed H0 and r_d
    pre = {}
    for lab, bgx, h, rd in [
        ("stress QRC CPL(-0.77,-0.21) Om=.311 H0=68.3 rd=144.7",
         H.Background(H.E_cpl(0.311, -0.77, -0.21, orad=9.0e-5)), 0.683, 144.7),
        ("Planck LCDM Om=.315 H0=67.4 rd=147.09", H.Background(H.E_lcdm(0.315)), 0.674, 147.09)]:
        r = {}
        c, _ = L["desi"].chi2(bgx, hrd_fixed=h * rd); r["DESI chi2 (N=12, 0 free)"] = c
        _, pull = L["desi"].residuals(bgx, h * rd)
        r["DESI pulls"] = {f"{k}@{z}": round(float(p), 2) for z, k, p in zip(L["desi"].z, L["desi"].kind, pull)}
        c, _ = cc.chi2(bgx, h_fixed=h); r["CC chi2 (N=15, 0 free)"] = c
        c, _ = L["pp"].chi2(bgx); r["PantheonPlus chi2 (N=1580, M free)"] = c
        c, _ = L["des"].chi2(bgx); r["DES chi2 (N=1820, M free)"] = c
        pre[lab] = r
    sp["preregistered_absolute"] = pre
    out["runtime_s"] = round(time.time() - t0, 1)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=lambda o: None if o is None else float(o))
    print("done", out["runtime_s"], "s")


def repo_agegraphic_port():
    """Logic copied from both repo copies of qrc_agegraphic_causal_dark_energy_validation.py
    (LCDM background, eta integrated from a=0.01). Returns their w0 and their 'wa'."""
    out = {}
    om, orad = 0.315, 9.2e-5
    ode = 1 - om - orad
    a = np.linspace(0.01, 1.0, 500)
    Hh = np.sqrt(orad * a**-4 + om * a**-3 + ode)
    d = 1.0 / (a**2 * Hh)
    eta = np.zeros_like(a)
    for i in range(1, len(a)):
        eta[i] = np.trapezoid(d[:i + 1], a[:i + 1])
    # eta[0] only affects w at a=0.01 in either copy, so one variant suffices.
    for variant, e0 in (("both_copies", eta[1] * 0.1),):
        et = eta.copy(); et[0] = e0
        w = -1.0 + 2.0 / (3.0 * a * Hh * et)
        dw = np.gradient(w, a)
        out[variant] = {"w0": float(w[-1]), "dw/da(a=1)": float(dw[-1]), "their_wa=-dw/da": float(-dw[-1]),
                        "eta0_used": float(et[-1]),
                        "eta0_true_from_a=0": float(et[-1] + H.eta_matter_radiation(0.01, om, orad)),
                        "implied_Omega_q0_if_n=1": float(1.0 / et[-1] ** 2)}
    return out


if __name__ == "__main__":
    main()
