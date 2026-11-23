"""Real-data cosmology harness for the QRC audit of 2026-11-22.

Data files are opened read-only and parsed as numbers; nothing in the data
folder is executed. All nuisance parameters that enter linearly (SN absolute
magnitude M, BAO amplitude 1/(h r_d), chronometer h, SH0ES log h) are profiled
analytically, so numerical minimisation only runs over cosmological parameters.

Units: distances are dimensionless in units of c/H0 unless stated; E(z)=H/H0.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.optimize import brentq, minimize, minimize_scalar

C_KMS = 299792.458
DH100 = C_KMS / 100.0            # c / (100 km/s/Mpc) in Mpc
OMEGA_R_H2 = 4.18e-5             # photons + 3.046 massless neutrinos
H_RAD = 0.68                     # h used only to set Omega_r (negligible below z=2.4)
OMEGA_R = OMEGA_R_H2 / H_RAD**2
ZMAX = 2.40
ZGRID = np.linspace(0.0, ZMAX, 24001)
GYR_PER_INV_H0 = 977.792221      # 1/(1 km/s/Mpc) in Gyr

# --------------------------------------------------------------------------
# Backgrounds
# --------------------------------------------------------------------------


class Background:
    """E(z) tabulated on ZGRID plus the comoving distance integral."""

    def __init__(self, E, label="", extra=None):
        E = np.asarray(E, dtype=float)
        if E.shape != ZGRID.shape or not np.all(np.isfinite(E)) or np.any(E <= 0):
            raise ValueError("invalid E(z)")
        self.E = E
        self.chi = cumulative_trapezoid(1.0 / E, ZGRID, initial=0.0)
        self.label = label
        self.extra = extra or {}

    def e(self, z):
        return np.interp(z, ZGRID, self.E)

    def dm(self, z):
        return np.interp(z, ZGRID, self.chi)


def E_lcdm(om, z=ZGRID, orad=OMEGA_R):
    zp = 1.0 + z
    return np.sqrt(om * zp**3 + orad * zp**4 + (1.0 - om - orad))


def E_cpl(om, w0, wa, z=ZGRID, orad=OMEGA_R):
    a = 1.0 / (1.0 + z)
    f = a ** (-3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * (1.0 - a))
    return np.sqrt(om * a**-3 + orad * a**-4 + (1.0 - om - orad) * f)


def E_eds(z=ZGRID):
    return (1.0 + z) ** 1.5


def E_rhct(z=ZGRID):
    return 1.0 + z


# ----------------------------- agegraphic ---------------------------------
# rho_q = 3 n^2 Mp^2 / eta^2, eta = conformal time from the big bang.
# In H0 units: E^2 = Om a^-3 + Or a^-4 + n^2 / eta~^2, d eta~/d ln a = 1/(a E).
A_INI = 1.0e-8
X_INI = math.log(A_INI)
X_END = 0.02  # integrate slightly past a=1 so derivatives at a=1 are central


def eta_matter_radiation(a, om, orad=OMEGA_R):
    """Exact conformal time (H0 units) of a flat matter+radiation universe."""
    return (2.0 / om) * (np.sqrt(om * a + orad) - np.sqrt(orad))


def _age_solve(n, om, orad=OMEGA_R, delta=0.0):
    def rhs(x, y):
        a = math.exp(x)
        eta = math.exp(y[0])
        E = math.sqrt(om * a**-3 + orad * a**-4 + n * n / (eta * eta))
        return [1.0 / (a * E * eta)]

    eta0 = eta_matter_radiation(A_INI, om, orad) + delta
    sol = solve_ivp(rhs, (X_INI, X_END), [math.log(eta0)], method="DOP853",
                    rtol=1e-10, atol=1e-12, dense_output=True)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol


def _age_E_from_sol(sol, n, om, orad, x):
    a = np.exp(x)
    eta = np.exp(sol.sol(x)[0])
    E = np.sqrt(om * a**-3 + orad * a**-4 + n * n / eta**2)
    return a, eta, E


def _age_E1sq(n, om, orad=OMEGA_R, delta=0.0):
    sol = _age_solve(n, om, orad, delta)
    _, _, E = _age_E_from_sol(sol, n, om, orad, np.array([0.0]))
    return E[0] ** 2


def agegraphic_om_for_n(n, orad=OMEGA_R):
    """Strict model: Om0 is fixed by n through E(a=1)=1."""
    f = lambda om: _age_E1sq(n, om, orad) - 1.0
    lo, hi = 1e-3, 0.999
    if f(lo) * f(hi) > 0:
        return None
    return brentq(f, lo, hi, xtol=1e-12)


def agegraphic_n_for_om(om, orad=OMEGA_R):
    f = lambda n: _age_E1sq(n, om, orad) - 1.0
    return brentq(f, 0.2, 20.0, xtol=1e-12)


def agegraphic_delta_for(n, om, orad=OMEGA_R):
    """Relaxed variant: eta offset delta>=0 chosen so that E(1)=1."""
    f = lambda d: _age_E1sq(n, om, orad, d) - 1.0
    f0 = f(0.0)
    if f0 < 0:
        return None
    if f0 == 0:
        return 0.0
    hi = 1.0
    while f(hi) > 0:
        hi *= 4.0
        if hi > 1e4:
            return None
    return brentq(f, 0.0, hi, xtol=1e-12)


def agegraphic_background(n, om=None, orad=OMEGA_R, relaxed=False):
    if relaxed:
        delta = agegraphic_delta_for(n, om, orad)
        if delta is None:
            return None
    else:
        om = agegraphic_om_for_n(n, orad)
        if om is None:
            return None
        delta = 0.0
    sol = _age_solve(n, om, orad, delta)
    x = -np.log1p(ZGRID)
    _, _, E = _age_E_from_sol(sol, n, om, orad, x)
    w0, wa, wa_analytic, oq0 = agegraphic_w0wa(sol, n, om, orad)
    return Background(E, "agegraphic", {"n": n, "Om": om, "delta_eta": delta,
                                        "Omega_q0": oq0, "w0": w0, "wa": wa,
                                        "wa_analytic": wa_analytic})


def agegraphic_w0wa(sol, n, om, orad=OMEGA_R, h=1e-4):
    x = np.array([-h, 0.0, h])
    a, eta, E = _age_E_from_sol(sol, n, om, orad, x)
    w = -1.0 + 2.0 / (3.0 * a * E * eta)
    dwdx = (w[2] - w[0]) / (2 * h)          # at a=1, dw/da = dw/dlna
    oq = n * n / (E[1] ** 2 * eta[1] ** 2)
    orr = orad / E[1] ** 2
    doq = oq * ((1 - oq) * (3 - 2 * math.sqrt(oq) / n) + orr)
    dw_an = (2.0 / (3.0 * n)) * (doq / (2 * math.sqrt(oq)) - math.sqrt(oq))
    return float(w[1]), float(-dwdx), float(-dw_an), float(oq)


def agegraphic_w_of_a(n, om, a_values, orad=OMEGA_R, delta=0.0):
    sol = _age_solve(n, om, orad, delta)
    a, eta, E = _age_E_from_sol(sol, n, om, orad, np.log(np.asarray(a_values)))
    return -1.0 + 2.0 / (3.0 * a * E * eta), n * n / (E**2 * eta**2)


# -------------------------- retarded trace DE ------------------------------
# Repo: rho_DE(t) = int_0^t exp(-(t-t')/tau) S(t') dt',
#       S = rho_m/(rho_m + 2 rho_r) * sqrt(rho_m), normalised to Omega_DE0.
RT_A_INI = 1.0e-7


def _rt_source(x, om, orad):
    a = math.exp(x)
    rm = om * a**-3
    rr = orad * a**-4
    return rm / (rm + 2.0 * rr) * math.sqrt(rm)


def _rt_solve(om, tau, orad, K=None):
    """K=None: background is LCDM (as written in repo). Else self-consistent with
    rho_DE = K * rho in the expansion rate."""
    ode = 1.0 - om - orad

    def rhs(x, y):
        a = math.exp(x)
        if K is None:
            E = math.sqrt(om * a**-3 + orad * a**-4 + ode)
        else:
            E = math.sqrt(om * a**-3 + orad * a**-4 + K * max(y[0], 0.0))
        return [(_rt_source(x, om, orad) - y[0] / tau) / E]

    sol = solve_ivp(rhs, (math.log(RT_A_INI), X_END), [0.0], method="DOP853",
                    rtol=1e-10, atol=1e-14, dense_output=True)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol


def retarded_trace_background(om, tau=1.35, selfconsistent=False, orad=OMEGA_R):
    ode = 1.0 - om - orad
    if ode <= 0:
        return None
    if not selfconsistent:
        sol = _rt_solve(om, tau, orad, None)
        K = ode / sol.sol(0.0)[0]
    else:
        g = lambda K: K * _rt_solve(om, tau, orad, K).sol(0.0)[0] - ode
        hi = 1.0
        while g(hi) < 0:
            hi *= 3.0
            if hi > 1e6:
                return None
        K = brentq(g, 1e-9, hi, xtol=1e-13, rtol=1e-12)
        sol = _rt_solve(om, tau, orad, K)
    x = -np.log1p(ZGRID)
    a = np.exp(x)
    rho = K * sol.sol(x)[0]
    E = np.sqrt(om * a**-3 + orad * a**-4 + rho)
    # equation of state of the DE component from d ln rho/d ln a
    hx = 1e-4
    xs = np.array([-hx, 0.0, hx])
    r = sol.sol(xs)[0]
    as_ = np.exp(xs)
    if selfconsistent:
        Eb = np.sqrt(om * as_**-3 + orad * as_**-4 + K * r)
    else:
        Eb = np.sqrt(om * as_**-3 + orad * as_**-4 + ode)
    S = np.array([_rt_source(xx, om, orad) for xx in xs])
    dlnr = (S - r / tau) / (Eb * r)
    w = -1.0 - dlnr / 3.0
    w0 = float(w[1])
    wa = float(-(w[2] - w[0]) / (2 * hx))
    # Omega_DE at recombination as a sanity number
    xr = math.log(1.0 / 1101.0)
    rr = K * sol.sol(xr)[0]
    ar = 1.0 / 1101.0
    Er2 = om * ar**-3 + orad * ar**-4 + rr
    return Background(E, "retarded_trace", {"Om": om, "tau": tau, "K": float(K),
                                            "w0": w0, "wa": wa,
                                            "Omega_DE_z1100": float(rr / Er2),
                                            "selfconsistent": selfconsistent})


# --------------------------------------------------------------------------
# Data loaders (numbers only)
# --------------------------------------------------------------------------


def load_pantheon(dat_path, cov_path):
    d = np.genfromtxt(dat_path, names=True, dtype=None, encoding="ascii")
    with open(cov_path, "r", encoding="ascii") as f:
        n = int(f.readline().strip())
        vals = np.array(f.read().split(), dtype=float)
    if vals.size != n * n or n != len(d):
        raise ValueError("Pantheon+ covariance size mismatch")
    C = vals.reshape(n, n)
    # The text file is rounded; asymmetry is ~3e-8 mag^2 (checked), so symmetrise.
    if np.max(np.abs(C - C.T)) > 1e-6:
        raise ValueError("Pantheon+ covariance not symmetric")
    return d, 0.5 * (C + C.T)


def load_des(csv_path, npz_path):
    rows = []
    with open(csv_path, "r", encoding="ascii") as f:
        for line in f:
            if line.startswith("VARNAMES:"):
                names = line.split()[1:]
            elif line.startswith("SN:"):
                rows.append(line.split()[1:])
    idx = {k: i for i, k in enumerate(names)}
    get = lambda k: np.array([float(r[idx[k]]) for r in rows])
    data = {k: get(k) for k in ("zHD", "zHEL", "MU", "MUERR", "MUERR_SYS")}
    f = np.load(npz_path, allow_pickle=False)
    n = int(f["nsn"][0])
    if n != len(rows):
        raise ValueError("DES covariance size mismatch")
    A = np.zeros((n, n))
    A[np.triu_indices(n)] = f["cov"].astype(float)
    A = A + A.T - np.diag(np.diag(A))
    return data, A  # A is the INVERSE covariance (README)


def load_desi(mean_path, cov_path):
    z, v, kind = [], [], []
    with open(mean_path, "r", encoding="ascii") as f:
        for line in f:
            if line.strip().startswith("#") or not line.strip():
                continue
            p = line.split()
            z.append(float(p[0])); v.append(float(p[1])); kind.append(p[2])
    C = np.loadtxt(cov_path)
    return np.array(z), np.array(v), kind, C


def load_cc(path):
    z, H, e = [], [], []
    with open(path, "r", encoding="ascii") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split(",")
            z.append(float(p[0])); H.append(float(p[1])); e.append(float(p[2]))
    return np.array(z), np.array(H), np.array(e)


# --------------------------------------------------------------------------
# Likelihoods
# --------------------------------------------------------------------------


class SNLike:
    """mu_obs - mu_th with full inverse covariance; M (or offset) profiled.

    If cal_mu is given (Pantheon+SH0ES mode), rows with is_cal use Cepheid
    distances and the Hubble-flow rows get an extra linear term u=-5log10(h).
    """

    def __init__(self, name, zhd, zhel, mobs, cinv, is_cal=None, cal_mu=None, one_plus_z=True):
        self.name, self.zhd, self.zhel, self.m = name, zhd, zhel, mobs
        self.cinv = cinv
        self.N = len(mobs)
        self.is_cal = np.zeros(self.N, bool) if is_cal is None else is_cal.astype(bool)
        self.cal_mu = cal_mu
        self.one_plus_z = one_plus_z
        self.ones = np.ones(self.N)
        self.c1 = cinv @ self.ones
        self.s11 = self.ones @ self.c1

    def mu_th(self, bg, h=1.0):
        dm = bg.dm(self.zhd)
        fac = (1.0 + self.zhel) if self.one_plus_z else 1.0
        with np.errstate(divide="ignore"):
            return 5.0 * np.log10(fac * dm * DH100 / h) + 25.0

    def chi2(self, bg, offset=0.0, h_fixed=None):
        if self.cal_mu is None:
            r = self.m - self.mu_th(bg, 0.7) - offset
            cr = self.cinv @ r
            b = self.ones @ cr
            chi2 = r @ cr - b * b / self.s11
            return float(chi2), {"M_h70": float(b / self.s11)}
        s = (~self.is_cal).astype(float)
        mu = np.where(self.is_cal, self.cal_mu, self.mu_th(bg, 1.0) + offset)
        r0 = self.m - mu
        if h_fixed is not None:
            r0 = r0 + 5.0 * math.log10(h_fixed) * s
            cr = self.cinv @ r0
            b = self.ones @ cr
            return float(r0 @ cr - b * b / self.s11), {"M": float(b / self.s11), "H0": 100 * h_fixed}
        X = np.vstack([self.ones, s]).T
        CX = self.cinv @ X
        F = X.T @ CX
        th = np.linalg.solve(F, CX.T @ r0)
        res = r0 - X @ th
        chi2 = float(res @ (self.cinv @ res))
        cov = np.linalg.inv(F)
        H0 = 100.0 * 10 ** (-th[1] / 5.0)
        sH0 = H0 * math.log(10) / 5.0 * math.sqrt(cov[1, 1])
        return chi2, {"M": float(th[0]), "H0": float(H0), "sigma_H0_cond": float(sH0)}


class BAOLike:
    def __init__(self, z, val, kind, cov):
        self.name = "DESI-DR1"
        self.z, self.d, self.kind = z, val, list(kind)
        self.cinv = np.linalg.inv(cov)
        self.N = len(val)
        self.dcd = val @ self.cinv @ val

    def template(self, bg):
        t = np.empty(self.N)
        for i, (z, k) in enumerate(zip(self.z, self.kind)):
            dm = bg.dm(z); E = bg.e(z)
            if k == "DM_over_rs":
                t[i] = dm
            elif k == "DH_over_rs":
                t[i] = 1.0 / E
            elif k == "DV_over_rs":
                t[i] = (z * dm * dm / E) ** (1.0 / 3.0)
            else:
                raise ValueError(k)
        return t  # observable = (c/(H0 r_d)) * t = (DH100/(h r_d)) * t

    def chi2(self, bg, hrd_fixed=None):
        t = self.template(bg)
        ct = self.cinv @ t
        tct = t @ ct
        tcd = self.d @ ct
        if hrd_fixed is None:
            alpha = tcd / tct
            chi2 = self.dcd - tcd * tcd / tct
            return float(chi2), {"hrd": float(DH100 / alpha)}
        alpha = DH100 / hrd_fixed
        r = self.d - alpha * t
        return float(r @ self.cinv @ r), {"hrd": float(hrd_fixed)}

    def residuals(self, bg, hrd):
        t = self.template(bg)
        pred = DH100 / hrd * t
        err = np.sqrt(np.diag(np.linalg.inv(self.cinv)))
        return pred, (self.d - pred) / err


class CCLike:
    def __init__(self, z, H, err):
        self.name = "CC"
        self.z, self.H, self.err = z, H, err
        self.N = len(z)

    def chi2(self, bg, h_fixed=None):
        E = bg.e(self.z)
        w = 1.0 / self.err**2
        if h_fixed is None:
            H0 = np.sum(w * E * self.H) / np.sum(w * E * E)
        else:
            H0 = 100.0 * h_fixed
        return float(np.sum(w * (self.H - H0 * E) ** 2)), {"H0": float(H0)}


# --------------------------------------------------------------------------
# Models and fitting
# --------------------------------------------------------------------------


class Model:
    def __init__(self, name, pnames, bounds, build, sn_offset=0.0, cc_h=None,
                 bao_hrd=None, grid=None, description=""):
        self.name, self.pnames, self.bounds, self.build = name, pnames, bounds, build
        self.sn_offset, self.cc_h, self.bao_hrd = sn_offset, cc_h, bao_hrd
        self.grid = grid  # list of 1D arrays for a coarse scan
        self.description = description
        self._cache = {}

    def background(self, p):
        key = tuple(np.round(p, 12))
        if key in self._cache:
            return self._cache[key]
        for v, (lo, hi) in zip(p, self.bounds):
            if not (lo <= v <= hi):
                return None
        try:
            bg = self.build(*p)
        except (ValueError, RuntimeError, ZeroDivisionError, FloatingPointError):
            bg = None
        if len(self._cache) > 4000:
            self._cache.clear()
        self._cache[key] = bg
        return bg


BIG = 1e12


def evaluate(model, likes, p):
    bg = model.background(p)
    if bg is None:
        return BIG, None
    total, parts = 0.0, {}
    for L in likes:
        if isinstance(L, SNLike):
            c, nu = L.chi2(bg, offset=model.sn_offset)
        elif isinstance(L, BAOLike):
            c, nu = L.chi2(bg, hrd_fixed=model.bao_hrd)
        else:
            c, nu = L.chi2(bg, h_fixed=model.cc_h)
        if not np.isfinite(c):
            return BIG, None
        parts[L.name] = {"chi2": c, "N": L.N, **nu}
        total += c
    return total, {"bg": bg, "parts": parts}


def n_nuisance(model, likes):
    k = 0
    for L in likes:
        if isinstance(L, SNLike):
            k += 1 if L.cal_mu is None else 2
        elif isinstance(L, BAOLike):
            k += 0 if model.bao_hrd is not None else 1
        else:
            k += 0 if model.cc_h is not None else 1
    return k


def fit(model, likes, extra_starts=()):
    f = lambda p: evaluate(model, likes, np.atleast_1d(p))[0]
    npar = len(model.pnames)
    if npar == 0:
        best = np.array([])
    elif npar == 1:
        g = model.grid[0]
        vals = np.array([f([x]) for x in g])
        i = int(np.argmin(vals))
        lo = g[max(i - 1, 0)]; hi = g[min(i + 1, len(g) - 1)]
        if hi > lo:
            r = minimize_scalar(lambda x: f([x]), bounds=(lo, hi), method="bounded",
                                options={"xatol": 1e-7})
            best = np.array([r.x]) if r.fun <= vals[i] else np.array([g[i]])
        else:
            best = np.array([g[i]])
    else:
        mesh = np.array(np.meshgrid(*model.grid, indexing="ij")).reshape(npar, -1).T
        vals = np.array([f(p) for p in mesh])
        order = np.argsort(vals)[:3]
        starts = [mesh[j] for j in order] + [np.array(s) for s in extra_starts]
        best, bestv = None, np.inf
        for s in starts:
            r = minimize(f, s, method="Nelder-Mead",
                         options={"xatol": 1e-6, "fatol": 1e-7, "maxiter": 4000, "maxfev": 6000})
            if r.fun < bestv:
                best, bestv = r.x, r.fun
    chi2, info = evaluate(model, likes, np.atleast_1d(best))
    N = sum(L.N for L in likes)
    k = npar + n_nuisance(model, likes)
    out = {"model": model.name, "params": dict(zip(model.pnames, map(float, np.atleast_1d(best)))),
           "chi2": chi2, "N": N, "k": k, "dof": N - k,
           "AIC": chi2 + 2 * k, "BIC": chi2 + k * math.log(N)}
    if info is not None:
        out["parts"] = info["parts"]
        out["derived"] = {kk: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                          for kk, v in info["bg"].extra.items()}
        out["_bg"] = info["bg"]
    return out


def profile_1d(model, likes, idx, values, start):
    """Profile chi2 over parameter idx (others minimised). Returns chi2 array."""
    npar = len(model.pnames)
    res = []
    cur = np.array(start, dtype=float)
    others = [j for j in range(npar) if j != idx]
    for v in values:
        def f(q):
            p = np.empty(npar); p[idx] = v; p[others] = q
            return evaluate(model, likes, p)[0]
        if others:
            r = minimize(f, cur[others], method="Nelder-Mead",
                         options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 3000})
            cur[others] = r.x
            res.append(r.fun)
        else:
            res.append(f(np.array([])))
    return np.array(res)


def interval_from_profile(values, chi2, level=1.0):
    i = int(np.argmin(chi2)); m = chi2[i]
    d = chi2 - m - level
    lo = hi = None
    for j in range(i, 0, -1):
        if d[j - 1] > 0 >= d[j]:
            lo = values[j - 1] + (values[j] - values[j - 1]) * d[j - 1] / (d[j - 1] - d[j]); break
    for j in range(i, len(values) - 1):
        if d[j] <= 0 < d[j + 1]:
            hi = values[j] + (values[j + 1] - values[j]) * (-d[j]) / (d[j + 1] - d[j]); break
    return lo, hi


def grid_posterior_1d(x, chi2):
    w = np.exp(-0.5 * (chi2 - chi2.min()))
    w /= np.trapezoid(w, x)
    mean = np.trapezoid(x * w, x)
    sd = math.sqrt(np.trapezoid((x - mean) ** 2 * w, x))
    return float(mean), float(sd)
