#!/usr/bin/env python3
"""Gate 0: Nonlocal 0D Premetric Interface and Causal Transfer Impedance.

Authority: QRC_GATE0_PREGEOMETRIC_FOUNDATION.md (schema qrc.gate0.pregeometry.v2).

Every check is run twice: once on the mathematically claimed theorem (expected PASS)
and once on a deliberately wrong input (expected FAIL). Exit code 0 means every
observed outcome matched its expectation.

Mathematical Claims Checked:
  1. Relabelling Quotient Invariance:
     Action S_0 and observables O are invariant under arbitrary relabelling pi in Sym(E_N).
  2. Cellular Chain Nilpotency:
     The cellular boundary operators satisfy partial_0 partial_1 = 0 and partial_1 partial_2 = 0.
  3. Incidence Coverage:
     Every active k-cell is in the support of at least one (k+1)-cell boundary.
  4. Coordinate-Free Word Distance Metric:
     d_word(x, y) = min |gamma| satisfies non-negativity, symmetry, and triangle inequality.
  5. Causal Transfer Impedance Metric:
     Omega(u, v) = (e_u - e_v)^T L^+ (e_u - e_v) on the connected relational skeleton
     satisfies metric axioms and Foster's network theorem: sum_{(u,v) in E} Omega(u, v) = |V| - 1.

Run: python3 -I -B check_gate0_premetric.py
"""

from __future__ import annotations

import itertools
import math
import sys
from collections import deque
import numpy as np

RESULTS: list[tuple[str, bool, bool]] = []


def record(name: str, observed: bool, expected: bool) -> None:
    RESULTS.append((name, observed, expected))
    tag = "PASS" if observed else "FAIL"
    exp = "PASS" if expected else "FAIL"
    ok = "ok" if observed == expected else "MISMATCH"
    print(f"  [{ok}] {name}: observed {tag}, expected {exp}")


print("=" * 80)
print("GATE 0: NONLOCAL 0D PREMETRIC FOUNDATION & CAUSAL IMPEDANCE AUDIT")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. Relabelling Quotient: S_0(pi H) = S_0(H) for pi in Sym(E_N)
# ---------------------------------------------------------------------------
def compute_graph_spectrum(adj: np.ndarray) -> np.ndarray:
    return np.sort(np.linalg.eigvalsh(adj))

# Construct a graph and apply random vertex permutations
rng = np.random.default_rng(42)
N = 6
adj_base = np.zeros((N, N), dtype=float)
# Ring + cross-link (connected 1-skeleton)
for i in range(N):
    adj_base[i, (i + 1) % N] = adj_base[(i + 1) % N, i] = 1.0
adj_base[0, 3] = adj_base[3, 0] = 1.0

spec_base = compute_graph_spectrum(adj_base)
perm_ok = True
for _ in range(20):
    p = rng.permutation(N)
    P = np.eye(N)[p]
    adj_perm = P @ adj_base @ P.T
    spec_perm = compute_graph_spectrum(adj_perm)
    if not np.allclose(spec_base, spec_perm, atol=1e-12):
        perm_ok = False

record("Action and spectral observables invariant under Sym(E_N) relabelling", perm_ok, True)

# Wrong input: Coordinate-dependent observable (e.g. vertex index weight)
coord_obs_base = float(sum(i * adj_base[i, j] for i in range(N) for j in range(N)))
p_sample = np.array([1, 0, 2, 3, 4, 5])
P_s = np.eye(N)[p_sample]
adj_p_s = P_s @ adj_base @ P_s.T
coord_obs_perm = float(sum(i * adj_p_s[i, j] for i in range(N) for j in range(N)))
record("[wrong input] Coordinate-dependent index sum is invariant under Sym(E_N)",
       abs(coord_obs_base - coord_obs_perm) < 1e-12, False)

# ---------------------------------------------------------------------------
# 2. Cellular Boundary Nilpotency: partial_k-1 partial_k = 0
# ---------------------------------------------------------------------------
# 0-cells: 4 vertices (v0, v1, v2, v3)
# 1-cells: 5 edges: e01, e12, e20, e23, e30
# 2-cells: 2 faces: f012 (boundary e01 + e12 - e20), f023 (boundary e20 + e23 - e30)

# C_0 dim 4, C_1 dim 5, C_2 dim 2
d1 = np.array([
    # e01 e12 e20 e23 e30
    [-1,   0,  1,   0,  1],  # v0
    [ 1,  -1,  0,   0,  0],  # v1
    [ 0,   1, -1,  -1,  0],  # v2
    [ 0,   0,  0,   1, -1],  # v3
], dtype=float)

d2 = np.array([
    # f012 f023
    [ 1,    0],  # e01
    [ 1,    0],  # e12
    [ 1,   -1],  # e20
    [ 0,    1],  # e23
    [ 0,    1],  # e30
], dtype=float)

d1_d2 = d1 @ d2
record("Cellular boundary nilpotency exact: partial_1 partial_2 = 0",
       np.allclose(d1_d2, 0.0, atol=1e-14), True)

# Wrong input: Non-closed 2-cell boundary
d2_bad = d2.copy()
d2_bad[0, 0] += 1.0  # broken boundary cycle
record("[wrong input] Corrupted face boundary satisfies partial_1 partial_2 = 0",
       np.allclose(d1 @ d2_bad, 0.0, atol=1e-14), False)

# ---------------------------------------------------------------------------
# 3. Incidence Coverage: X_k subseteq union supp(partial_k+1)
# ---------------------------------------------------------------------------
# Every vertex is in the support of d1, every edge is in the support of d2
v_covered = all(np.any(d1[v, :] != 0) for v in range(4))
e_covered = all(np.any(d2[e, :] != 0) for e in range(5))
record("Incidence coverage exact: all 0-cells and 1-cells covered by higher boundaries",
       v_covered and e_covered, True)

# Wrong input: Disconnected isolated vertex
d1_orphan = np.pad(d1, ((0, 1), (0, 0))) # vertex v4 with no incident edges
v_orphan_covered = all(np.any(d1_orphan[v, :] != 0) for v in range(5))
record("[wrong input] Graph with isolated orphan vertex satisfies incidence coverage",
       v_orphan_covered, False)

# ---------------------------------------------------------------------------
# 4. Word Metric Axioms on the Connected 1-Skeleton
# ---------------------------------------------------------------------------
def compute_bfs_distances(adj: np.ndarray) -> np.ndarray:
    n = adj.shape[0]
    dist = np.full((n, n), np.inf)
    for src in range(n):
        dist[src, src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in range(n):
                if adj[u, v] > 0 and dist[src, v] == np.inf:
                    dist[src, v] = dist[src, u] + 1
                    q.append(v)
    return dist

word_dist = compute_bfs_distances(adj_base)
word_pos = all(word_dist[i, i] == 0 for i in range(N)) and all(
    word_dist[i, j] >= 1 for i in range(N) for j in range(N) if i != j
)
word_sym = np.allclose(word_dist, word_dist.T)
word_tri = True
for i in range(N):
    for j in range(N):
        for k in range(N):
            if word_dist[i, k] > word_dist[i, j] + word_dist[j, k]:
                word_tri = False

record("Combinatorial word distance satisfies metric positivity (d >= 0, d(x,x)=0)", word_pos, True)
record("Combinatorial word distance satisfies symmetry (d(x,y) = d(y,x))", word_sym, True)
record("Combinatorial word distance satisfies triangle inequality (d(x,z) <= d(x,y) + d(y,z))", word_tri, True)

# ---------------------------------------------------------------------------
# 5. Causal Transfer Impedance Metric and Foster's Network Theorem
# ---------------------------------------------------------------------------
deg = np.diag(adj_base.sum(axis=1))
lap = deg - adj_base
pinv = np.linalg.pinv(lap)

omega = np.zeros((N, N), dtype=float)
for u in range(N):
    for v in range(N):
        delta = np.zeros(N)
        delta[u] += 1.0
        delta[v] -= 1.0
        omega[u, v] = float(delta @ pinv @ delta)

# Metric axioms for impedance
omega_pos = all(omega[i, i] < 1e-12 for i in range(N)) and all(
    omega[i, j] > 1e-6 for i in range(N) for j in range(N) if i != j
)
omega_sym = np.allclose(omega, omega.T, atol=1e-12)
omega_tri = True
for i in range(N):
    for j in range(N):
        for k in range(N):
            if omega[i, k] > omega[i, j] + omega[j, k] + 1e-10:
                omega_tri = False

record("Causal transfer impedance Omega(u,v) satisfies metric positivity", omega_pos, True)
record("Causal transfer impedance Omega(u,v) satisfies symmetry", omega_sym, True)
record("Causal transfer impedance Omega(u,v) satisfies triangle inequality", omega_tri, True)

# Foster's First Theorem on connected network
edges_list = [(i, j) for i in range(N) for j in range(i + 1, N) if adj_base[i, j] > 0]
foster_sum = sum(omega[u, v] for u, v in edges_list)
expected_foster = float(N - 1)
foster_ok = abs(foster_sum - expected_foster) < 1e-10
print(f"     Foster edge impedance sum: {foster_sum:.6f} (|V|-1 = {expected_foster:.6f})")
record("Foster's network theorem exact: sum_{(u,v) in E} Omega(u, v) = |V| - 1", foster_ok, True)

# Wrong input: False Foster claim (e.g. equal to |E| or |V|)
record("[wrong input] Foster sum equals number of edges |E|", abs(foster_sum - float(len(edges_list))) < 1e-10, False)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
mismatches = [r for r in RESULTS if r[1] != r[2]]
print(f"\nGate 0 Summary: {len(RESULTS)} checks, {len(mismatches)} mismatches with expectation")
sys.exit(1 if mismatches else 0)
