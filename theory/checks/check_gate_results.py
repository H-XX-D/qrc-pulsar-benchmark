"""Independent checks of the Gate 1 and Gate 5 formulas quoted from QRC_PROJECT_STATUS.md.

Every check is run twice: once on the claimed formula (expected PASS) and once on a
deliberately wrong input (expected FAIL), so each check is shown to be able to fail.
Exit code 0 means every observed outcome matched its expectation.

Provenance: copied unchanged (apart from this paragraph) from the independent audit
audits/2026-11-22_noncosmo/check_gate_results.py in the QRC repository. It needs only
numpy and scipy, reads no files and writes nothing. It does NOT test the Gate 1
rank selection; see check_gate1_selection.py for that.

Run: python3 -I -B check_gate_results.py
"""

from __future__ import annotations

import itertools
import math
import sys
from collections import deque

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from scipy.special import gammaln, ive

RESULTS: list[tuple[str, bool, bool]] = []


def record(name: str, observed: bool, expected: bool) -> None:
    RESULTS.append((name, observed, expected))
    tag = "PASS" if observed else "FAIL"
    exp = "PASS" if expected else "FAIL"
    ok = "ok" if observed == expected else "MISMATCH"
    print(f"  [{ok}] {name}: observed {tag}, expected {exp}")


# ---------------------------------------------------------------------------
# 1. V_3(R) by breadth-first search on the Cayley graph of Gamma = Z[D], |D| = 3.
#    The doc defines Gamma as the free abelian group on D and the word metric with
#    generators +d and -d. Group elements are integer coefficient vectors; BFS uses
#    only the generator moves, never the closed form or an L1 norm.
# ---------------------------------------------------------------------------

def word_ball_counts(generators: list[tuple[int, ...]], r_max: int, modulus: int | None = None) -> list[int]:
    rank = len(generators[0])
    origin = (0,) * rank
    dist = {origin: 0}
    queue = deque([origin])
    while queue:
        x = queue.popleft()
        if dist[x] == r_max:
            continue
        for g in generators:
            y = tuple(a + b for a, b in zip(x, g))
            if modulus is not None:
                y = tuple(v % modulus for v in y)
            if y not in dist:
                dist[y] = dist[x] + 1
                queue.append(y)
    counts = [0] * (r_max + 1)
    for d in dist.values():
        counts[d] += 1
    return list(itertools.accumulate(counts))


def v3_claim(r: int) -> float:
    return (4 * r**3 + 6 * r**2 + 8 * r + 3) / 3


print("1. Word-ball volume V_3(R) = (4R^3+6R^2+8R+3)/3 (BFS on Z[D], generators +-d)")
basis = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
gens = basis + [tuple(-v for v in g) for g in basis]
R_MAX = 30
bfs = word_ball_counts(gens, R_MAX)
record("BFS count equals claimed V_3(R) for R=0..30",
       all(abs(bfs[r] - v3_claim(r)) < 1e-9 for r in range(R_MAX + 1)), True)
print(f"     sample: R=0..5 BFS {bfs[:6]}; claim {[int(v3_claim(r)) for r in range(6)]}")
# Wrong inputs: (a) monoid without inverses, (b) rank-2 group, (c) perturbed formula.
bfs_monoid = word_ball_counts(basis, R_MAX)
record("[wrong input] generators D only (no inverses) match V_3",
       all(abs(bfs_monoid[r] - v3_claim(r)) < 1e-9 for r in range(R_MAX + 1)), False)
bfs_rank2 = word_ball_counts([(1, 0), (0, 1), (-1, 0), (0, -1)], R_MAX)
record("[wrong input] rank-2 lattice matches V_3",
       all(abs(bfs_rank2[r] - v3_claim(r)) < 1e-9 for r in range(R_MAX + 1)), False)
record("[wrong input] formula (4R^3+6R^2+8R)/3 matches BFS",
       all(abs(bfs[r] - (4 * r**3 + 6 * r**2 + 8 * r) / 3) < 1e-9 for r in range(R_MAX + 1)), False)
# Torus Q_m = Gamma/m Gamma: the doc says the formula holds "before wrap".
m = 15
bfs_torus = word_ball_counts(gens, 12, modulus=m)
before = all(abs(bfs_torus[r] - v3_claim(r)) < 1e-9 for r in range(0, m // 2 + 1))
after = all(abs(bfs_torus[r] - v3_claim(r)) < 1e-9 for r in range(0, 13))
record(f"torus m={m}: V_3 holds for R <= {m // 2} (before wrap)", before, True)
record(f"[wrong input] torus m={m}: V_3 holds up to R=12 (after wrap)", after, False)
print(f"     cubic coefficient check: V_3(R)/R^3 at R=30 = {bfs[30] / 30**3:.4f} (4/3 = {4 / 3:.4f})")


# ---------------------------------------------------------------------------
# 2. Return probability p_D(t;0,0) = e^{-t} I_0(t/3)^3 for the total-rate-one walk
#    (rate 1/6 for each of +-d). Two computations that do not use the Bessel form:
#    (a) Poisson mixture of the exact multinomial return probability of the
#        6-direction discrete walk; (b) matrix exponential of L_D on a torus large
#        enough that wrap is negligible.
# ---------------------------------------------------------------------------

def discrete_return_prob(steps_even: int) -> float:
    """P(return after 2n steps) of the 6-neighbour walk on Z^3, exact multinomial sum."""
    n = steps_even // 2
    log_terms = []
    for i in range(n + 1):
        for j in range(n - i + 1):
            k = n - i - j
            log_terms.append(gammaln(2 * n + 1) - 2 * (gammaln(i + 1) + gammaln(j + 1) + gammaln(k + 1)))
    lt = np.array(log_terms)
    mx = lt.max()
    return float(math.exp(mx + math.log(np.exp(lt - mx).sum()) - 2 * n * math.log(6)))


def return_prob_series(t: float, n_max: int) -> float:
    total = 0.0
    for steps in range(0, n_max + 1, 2):
        log_poisson = -t + steps * math.log(t) - gammaln(steps + 1) if t > 0 else (0.0 if steps == 0 else -np.inf)
        total += math.exp(log_poisson) * discrete_return_prob(steps)
    return total


def claim_p(t: float, scale: float = 3.0, power: int = 3) -> float:
    # e^{-t} I_0(t/3)^3 = [e^{-t/3} I_0(t/3)]^3 = ive(0, t/3)^3, numerically stable.
    return float(ive(0, t / scale) ** power * math.exp(-t + power * t / scale))


print("\n2. Return probability p_D(t;0,0) = e^{-t} I_0(t/3)^3")
ts = [0.5, 2.0, 5.0, 10.0, 20.0]
series = {t: return_prob_series(t, n_max=int(t + 12 * math.sqrt(t + 1) + 30)) for t in ts}
rel = max(abs(series[t] - claim_p(t)) / claim_p(t) for t in ts)
record(f"multinomial Poisson series matches claim (max rel err {rel:.2e})", rel < 1e-9, True)
rel_bad = max(abs(series[t] - claim_p(t, scale=2.0)) / series[t] for t in ts)
record("[wrong input] e^{-t} I_0(t/2)^3 matches series", rel_bad < 1e-6, False)

# (b) generator on torus (Z/m)^3 with m = 41.
m = 41
N = m**3
idx = np.arange(N).reshape(m, m, m)
rows, cols, vals = [], [], []
for axis in range(3):
    for shift in (1, -1):
        nbr = np.roll(idx, -shift, axis=axis).ravel()
        rows.append(idx.ravel()); cols.append(nbr); vals.append(np.full(N, 1.0 / 6.0))
rows.append(idx.ravel()); cols.append(idx.ravel()); vals.append(np.full(N, -1.0))
L = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
row_sums = np.abs(np.asarray(L.sum(axis=1)).ravel()).max()
e0 = np.zeros(N); e0[idx[0, 0, 0]] = 1.0
gen_vals = {}
for t in [2.0, 5.0, 10.0]:
    gen_vals[t] = expm_multiply(L * t, e0)[idx[0, 0, 0]]
rel_gen = max(abs(gen_vals[t] - claim_p(t)) / claim_p(t) for t in gen_vals)
record(f"expm(t L_D) on 41^3 torus matches claim (max rel err {rel_gen:.2e}, generator row-sum {row_sums:.1e})",
       rel_gen < 1e-8, True)
# wrong input: total rate two (rate 1/3 per direction) without rescaling time.
rel_gen_bad = abs(expm_multiply(2 * L * 5.0, e0)[idx[0, 0, 0]] - claim_p(5.0)) / claim_p(5.0)
record("[wrong input] rate-two generator matches claim at t=5", rel_gen_bad < 1e-6, False)


# ---------------------------------------------------------------------------
# 3. Spectral dimension d_s(t) = -2 t d/dt log p -> 3. Computed by finite
#    differences of log p (no use of the doc's closed form for the derivative).
# ---------------------------------------------------------------------------
print("\n3. Spectral dimension limit")


def ds_numeric(t: float, scale: float = 3.0, power: int = 3) -> float:
    h = 1e-4 * t
    lp = lambda s: math.log(ive(0, s / scale)) * power - s + power * s / scale
    return -2 * t * (lp(t + h) - lp(t - h)) / (2 * h)


for t in [10.0, 100.0, 1e3, 1e4, 1e5]:
    print(f"     t={t:>8.0f}: d_s = {ds_numeric(t):.5f}")
record("d_s(1e5) within 1e-3 of 3", abs(ds_numeric(1e5) - 3) < 1e-3, True)
record("doc closed form 2t[1 - I1/I0](t/3) equals numeric derivative at t=50",
       abs(2 * 50 * (1 - ive(1, 50 / 3) / ive(0, 50 / 3)) - ds_numeric(50.0)) < 1e-6, True)
record("[wrong input] rank-2 kernel e^{-t} I_0(t/2)^2 gives d_s(1e5) within 1e-3 of 3",
       abs(ds_numeric(1e5, scale=2.0, power=2) - 3) < 1e-3, False)


# ---------------------------------------------------------------------------
# 4. q# = (1/3) sum d (x) d as the co-metric with -L_D symbol = (1/2) q#(theta,theta).
#    Use a random (non-orthonormal) realization of D to check the statement is
#    tensorial, and evaluate the symbol from the operator itself on plane waves.
# ---------------------------------------------------------------------------
print("\n4. Co-metric q# = (1/3) sum_d d(x)d from the generator symbol")
rng = np.random.default_rng(7)
Dvec = rng.normal(size=(3, 3))  # rows are images of the three generators in W


def symbol_from_operator(theta: np.ndarray) -> float:
    # (-L_D e^{i theta.x})/e^{i theta.x} evaluated by applying the difference operator.
    vals = [np.exp(1j * theta @ d) + np.exp(-1j * theta @ d) - 2 for d in Dvec]
    return float(-np.real(sum(vals)) / 6.0)


q_claim = sum(np.outer(d, d) for d in Dvec) / 3.0
q_wrong = sum(np.outer(d, d) for d in Dvec)
worst, worst_wrong = 0.0, 0.0
for _ in range(50):
    u = rng.normal(size=3)
    u /= np.linalg.norm(u)
    eps = 1e-3
    theta = eps * u
    lhs = symbol_from_operator(theta) / eps**2
    worst = max(worst, abs(lhs - 0.5 * u @ q_claim @ u) / (0.5 * u @ q_claim @ u))
    worst_wrong = max(worst_wrong, abs(lhs - 0.5 * u @ q_wrong @ u) / (0.5 * u @ q_wrong @ u))
record(f"-L_D symbol = (1/2) q#(theta,theta) + O(theta^4) (max rel err {worst:.1e})", worst < 1e-5, True)
record("[wrong input] q# = sum d(x)d (no 1/3) matches symbol", worst_wrong < 1e-5, False)


# ---------------------------------------------------------------------------
# 5. S(U(2) x U(3)) = (SU(2) x SU(3) x U(1)) / Z6 with phi(g2,g3,z) = (z^3 g2, z^-2 g3).
# ---------------------------------------------------------------------------
print("\n5. S(U(2)xU(3)) = (SU(2)xSU(3)xU(1))/Z6")


def random_su(n: int) -> np.ndarray:
    z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / math.sqrt(2)
    q, r = np.linalg.qr(z)
    q = q @ np.diag(np.diag(r) / np.abs(np.diag(r)))
    return q / np.linalg.det(q) ** (1.0 / n)


def phi(g2, g3, z, a=3, b=-2):
    return z**a * g2, z**b * g3


def in_s_u2u3(A, B) -> bool:
    unit = np.allclose(A.conj().T @ A, np.eye(2)) and np.allclose(B.conj().T @ B, np.eye(3))
    return unit and abs(np.linalg.det(A) * np.linalg.det(B) - 1) < 1e-10


def check_hom(a: int, b: int) -> bool:
    ok = True
    for _ in range(200):
        g2, h2, g3, h3 = random_su(2), random_su(2), random_su(3), random_su(3)
        z, w = np.exp(1j * rng.uniform(0, 2 * np.pi, 2))
        A1, B1 = phi(g2, g3, z, a, b)
        A2, B2 = phi(h2, h3, w, a, b)
        A12, B12 = phi(g2 @ h2, g3 @ h3, z * w, a, b)
        ok &= in_s_u2u3(A1, B1) and np.allclose(A1 @ A2, A12) and np.allclose(B1 @ B2, B12)
    return bool(ok)


def kernel_size(a: int, b: int, K: int = 720) -> int:
    # Kernel elements need z^a g2 = I, so g2 is central (+-I); likewise g3 = omega^k I.
    count = 0
    for k in range(K):
        z = np.exp(2j * np.pi * k / K)
        for s2 in (1, -1):
            for k3 in range(3):
                g2 = s2 * np.eye(2)
                g3 = np.exp(2j * np.pi * k3 / 3) * np.eye(3)
                A, B = phi(g2, g3, z, a, b)
                if np.allclose(A, np.eye(2)) and np.allclose(B, np.eye(3)):
                    count += 1
    return count


def check_surjective() -> bool:
    ok = True
    for _ in range(200):
        A = random_su(2) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        B = random_su(3)
        B = B * (np.linalg.det(A) * np.linalg.det(B)) ** (-1.0 / 3)  # force det A det B = 1
        z = np.linalg.det(A) ** (1.0 / 6)  # any sixth root of det A
        g2, g3 = A / z**3, B * z**2
        ok &= abs(np.linalg.det(g2) - 1) < 1e-10 and abs(np.linalg.det(g3) - 1) < 1e-10
        A2, B2 = phi(g2, g3, z)
        ok &= np.allclose(A2, A) and np.allclose(B2, B)
    return bool(ok)


ks = kernel_size(3, -2)
record("phi = (z^3 g2, z^-2 g3) is a homomorphism into S(U(2)xU(3))", check_hom(3, -2), True)
record("phi is surjective (explicit preimages)", check_surjective(), True)
record(f"kernel has order 6 (found {ks}; searched z in 720th roots of unity)", ks == 6, True)
gen = (-np.eye(2), np.exp(2j * np.pi / 3) * np.eye(3), np.exp(1j * np.pi / 3))
Ag, Bg = phi(*gen)
order = next(k for k in range(1, 13) if abs(gen[2] ** k - 1) < 1e-12)
record("doc's kernel generator (-I2, omega3 I3, e^{i pi/3}) maps to the identity and has order 6",
       np.allclose(Ag, np.eye(2)) and np.allclose(Bg, np.eye(3)) and order == 6, True)
Ab, Bb = phi(np.eye(2), np.exp(2j * np.pi / 3) * np.eye(3), np.exp(1j * np.pi / 3))
record("[wrong input] (I2, omega3 I3, e^{i pi/3}) is in the kernel", np.allclose(Ab, np.eye(2)) and np.allclose(Bb, np.eye(3)), False)
record("[wrong input] phi = (z g2, z^-1 g3) lands in S(U(2)xU(3))", check_hom(1, -1), False)
record("[wrong input] kernel order 3", ks == 3, False)


# ---------------------------------------------------------------------------
# 6. I_6 = (m+n-5) ch_3(E) for F_N = Lambda^2 E (+) E*, N = m+n, c_1(E)=0.
#    ch_3 is evaluated as tr(F^3) (common normalization dropped) on random traceless
#    block-diagonal Hermitian curvatures; Lambda^2 is built as an explicit matrix.
# ---------------------------------------------------------------------------
print("\n6. Anomaly polynomial I_6(F_N) = (N-5) ch_3(E)")


def lambda2_rep(F: np.ndarray) -> np.ndarray:
    n = F.shape[0]
    pairs = list(itertools.combinations(range(n), 2))
    index = {p: k for k, p in enumerate(pairs)}
    M = np.zeros((len(pairs), len(pairs)), dtype=complex)
    for (i, j), col in index.items():
        # F acts on e_i ^ e_j as (F e_i) ^ e_j + e_i ^ (F e_j)
        for k in range(n):
            for (a, b), coef in (((k, j), F[k, i]), ((i, k), F[k, j])):
                if a == b or coef == 0:
                    continue
                sign = 1.0
                if a > b:
                    a, b, sign = b, a, -1.0
                M[index[(a, b)], col] += sign * coef
    return M


def ch3(F: np.ndarray) -> complex:
    return np.trace(F @ F @ F)


def i6_ratio(mm: int, nn: int) -> float:
    X = rng.normal(size=(mm, mm)) + 1j * rng.normal(size=(mm, mm)); X = X + X.conj().T
    Y = rng.normal(size=(nn, nn)) + 1j * rng.normal(size=(nn, nn)); Y = Y + Y.conj().T
    F = np.zeros((mm + nn, mm + nn), dtype=complex)
    F[:mm, :mm] = X; F[mm:, mm:] = Y
    F -= np.trace(F) / (mm + nn) * np.eye(mm + nn)  # c_1(E) = 0 (determinant record)
    total = ch3(lambda2_rep(F)) + ch3(-F.T)  # E* has curvature -F^T
    return float(np.real(total / ch3(F)))


ok_all = True
for mm, nn in [(2, 2), (2, 3), (3, 3), (2, 4), (3, 4), (4, 4)]:
    r = i6_ratio(mm, nn)
    print(f"     (m,n)=({mm},{nn}): I_6/ch_3(E) = {r:+.6f}, claim m+n-5 = {mm + nn - 5}")
    ok_all &= abs(r - (mm + nn - 5)) < 1e-8
record("I_6/ch_3(E) = m+n-5 for six block sizes", ok_all, True)
record("[wrong input] I_6/ch_3(E) = m+n-4 (Lambda^2 term alone) at (2,3)", abs(i6_ratio(2, 3) - 1) < 1e-8, False)
F = np.diag(rng.normal(size=5)).astype(complex); F -= np.trace(F) / 5 * np.eye(5)
record("ch_1(F_N) = tr(Lambda^2 F) + tr(-F^T) = 0 for traceless F",
       abs(np.trace(lambda2_rep(F)) + np.trace(-F.T)) < 1e-10, True)
F_bad = F + 0.3 * np.eye(5)
record("[wrong input] ch_1(F_N) = 0 when c_1(E) != 0",
       abs(np.trace(lambda2_rep(F_bad)) + np.trace(-F_bad.T)) < 1e-10, False)

mismatches = [r for r in RESULTS if r[1] != r[2]]
print(f"\nSummary: {len(RESULTS)} checks, {len(mismatches)} mismatches with expectation")
sys.exit(1 if mismatches else 0)
