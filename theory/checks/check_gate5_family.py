#!/usr/bin/env python3
"""Exact-arithmetic checks of the conditional Gate 5 representation statements.

Source: QRC repository docs/QRC_GATE5_FINAL_INTERNAL_GROUP_ANOMALY_AND_JOINT_PHASE_FRONTIER.md
(Sections 3 to 5) and docs/QRC_GATE5_UNIFIED_GAUGE_PHASE_AND_PAIR_DUAL_ANOMALY_SELECTOR_THEOREM.md
(Sections 2.3, 2.4 and 7). Everything here is conditional on the supplied seed: a Hermitian
five-dimensional internal space E = W (+) C with dim W = 2, dim C = 3, one determinant vector,
and the left-Weyl module Lambda^2 E (+) Lambda^4 E. Nothing here derives that seed.

Checked, in exact rational arithmetic:
  1. The hypercharge Y = p/2 - q/3 on Lambda^p W (x) Lambda^q C reproduces the claimed
     15-state family (3,2)_{1/6} + (3b,1)_{-2/3} + (3b,1)_{1/3} + (1,2)_{-1/2} + (1,1)_1.
  2. Local anomaly sums SU(3)^3, SU(3)^2 Y, SU(2)^2 Y, grav^2 Y, Y^3 vanish, and the number
     of SU(2) doublets is even (Witten parity).
  3. Every multiplet descends to the Z_6 quotient: 3s + 2t + 6Y = 0 mod 6, with s the weak
     doublet parity and t the colour triality.
  4. The three charged Yukawa channels force the Higgs doublet hypercharge Y_H = 1/2.
  5. One-loop coefficients B_3, B_2, B_Y computed from the field content (N_g families plus
     one complex Higgs doublet) equal the closed forms in the pair-dual selector document,
     and sin^2(theta_W) = 3/8 when g_1 = g_2 with g_1 = sqrt(5/3) g_Y.
The pair-dual rank selector I_6 = (m+n-5) ch_3(E) and the Z_6 kernel are checked numerically
in check_gate_results.py and are not repeated here. The spin-bordism global anomaly
statement is cited in the documents from the literature and is not checked here.

Each check is also run on a deliberately wrong input and must FAIL there.
Exit status 0 if and only if every observed outcome equals its expectation.

Run: python3 -I -B check_gate5_family.py
"""

from __future__ import annotations

import sys
from fractions import Fraction as Fr
from math import comb

RESULTS: list[tuple[str, bool, bool]] = []


def record(name: str, observed: bool, expected: bool) -> None:
    RESULTS.append((name, observed, expected))
    tag = "PASS" if observed else "FAIL"
    exp = "PASS" if expected else "FAIL"
    ok = "ok" if observed == expected else "MISMATCH"
    print(f"  [{ok}] {name}: observed {tag}, expected {exp}")


# A multiplet is (colour label, weak dimension, Y). Colour label: "1", "3" or "3b".
COLOUR_DIM = {"1": 1, "3": 3, "3b": 3}
CUBIC_INDEX = {"1": 0, "3": 1, "3b": -1}        # A(r) for SU(3)
DYNKIN = {"1": Fr(0), "3": Fr(1, 2), "3b": Fr(1, 2)}
TRIALITY = {"1": 0, "3": 1, "3b": 2}
SU3_FROM_Q = {0: "1", 1: "3", 2: "3b", 3: "1"}   # Lambda^q of the fundamental of SU(3)


def exterior_family(yp: Fr, yq: Fr, degrees=(2, 4)) -> list[tuple[str, int, Fr]]:
    """Decompose the given exterior degrees of W (+) C with Y = yp*p + yq*q."""
    fam = []
    for deg in degrees:
        for p in range(0, 3):
            q = deg - p
            if 0 <= q <= 3:
                fam.append((SU3_FROM_Q[q], comb(2, p), yp * p + yq * q))
    return fam


CLAIMED = sorted([("3", 2, Fr(1, 6)), ("3b", 1, Fr(-2, 3)), ("3b", 1, Fr(1, 3)),
                  ("1", 2, Fr(-1, 2)), ("1", 1, Fr(1))])


def anomalies(fam) -> dict[str, Fr]:
    a = {"SU3^3": Fr(0), "SU3^2 Y": Fr(0), "SU2^2 Y": Fr(0), "grav^2 Y": Fr(0), "Y^3": Fr(0)}
    for col, w, y in fam:
        dim = COLOUR_DIM[col] * w
        a["SU3^3"] += CUBIC_INDEX[col] * w
        a["SU3^2 Y"] += DYNKIN[col] * w * y
        a["SU2^2 Y"] += (Fr(1, 2) if w == 2 else Fr(0)) * COLOUR_DIM[col] * y
        a["grav^2 Y"] += dim * y
        a["Y^3"] += dim * y ** 3
    return a


def doublets(fam) -> int:
    return sum(COLOUR_DIM[col] for col, w, _ in fam if w == 2)


def descends(fam) -> bool:
    return all((3 * (1 if w == 2 else 0) + 2 * TRIALITY[col] + 6 * y) % 6 == 0 for col, w, y in fam)


def betas(n_g: int, family, higgs_y: Fr = Fr(1, 2)) -> tuple[Fr, Fr, Fr]:
    """b = -11/3 C2(G) + 2/3 sum_Weyl T(R) + 1/3 sum_complex scalars T(R)."""
    b3 = Fr(-11) + Fr(2, 3) * n_g * sum(DYNKIN[c] * w for c, w, _ in family)
    b2 = Fr(-22, 3) + Fr(2, 3) * n_g * sum(Fr(1, 2) * COLOUR_DIM[c] for c, w, _ in family if w == 2) \
        + Fr(1, 3) * Fr(1, 2)
    by = Fr(2, 3) * n_g * sum(COLOUR_DIM[c] * w * y * y for c, w, y in family) + Fr(1, 3) * 2 * higgs_y ** 2
    return b3, b2, by


def main() -> int:
    print("1. Exterior family Lambda^2 E + Lambda^4 E with Y = p/2 - q/3")
    fam = exterior_family(Fr(1, 2), Fr(-1, 3))
    for col, w, y in sorted(fam):
        print(f"     ({col}, {w})_{y}")
    record("decomposition equals the claimed 15-state family", sorted(fam) == CLAIMED, True)
    record("state count is 15", sum(COLOUR_DIM[c] * w for c, w, _ in fam) == 15, True)
    with_sterile = exterior_family(Fr(1, 2), Fr(-1, 3), degrees=(0, 2, 4))
    record("Lambda^even adds one neutral singlet (sterile nu^c), 16 states",
           sum(COLOUR_DIM[c] * w for c, w, _ in with_sterile) == 16 and ("1", 1, Fr(0)) in with_sterile, True)
    bad = exterior_family(Fr(1, 2), Fr(-1, 2))
    record("[wrong input] Y = p/2 - q/2 reproduces the family", sorted(bad) == CLAIMED, False)

    print("\n2. Local anomalies and Witten parity")
    a = anomalies(fam)
    print("     " + ", ".join(f"{k} = {v}" for k, v in a.items()) + f"; doublets = {doublets(fam)}")
    record("all five local anomaly sums vanish", all(v == 0 for v in a.values()), True)
    record("number of SU(2) doublets is even", doublets(fam) % 2 == 0, True)
    no_ec = [x for x in fam if x != ("1", 1, Fr(1))]
    record("[wrong input] family without e^c is anomaly free", all(v == 0 for v in anomalies(no_ec).values()), False)
    record("[wrong input] Y = p/2 - q/2 family is anomaly free", all(v == 0 for v in anomalies(bad).values()), False)
    no_l = [x for x in fam if x != ("1", 2, Fr(-1, 2))]
    record("[wrong input] family without L has even doublet count", doublets(no_l) % 2 == 0, False)
    record("sterile singlet leaves anomalies unchanged",
           anomalies(with_sterile) == a, True)

    print("\n3. Descent to the Z_6 quotient")
    record("every multiplet satisfies 3s + 2t + 6Y = 0 mod 6", descends(fam), True)
    record("[wrong input] (3,1)_0 descends", descends([("3", 1, Fr(0))]), False)

    print("\n4. Higgs hypercharge from the charged Yukawa channels")
    yq, yu, yd, yl, ye = Fr(1, 6), Fr(-2, 3), Fr(1, 3), Fr(-1, 2), Fr(1)
    sols = {"Q u^c H": -(yq + yu), "Q d^c H*": yq + yd, "L e^c H*": yl + ye}
    print("     required Y_H per channel: " + ", ".join(f"{k}: {v}" for k, v in sols.items()))
    record("all three channels require Y_H = 1/2", set(sols.values()) == {Fr(1, 2)}, True)
    record("[wrong input] Y_H = 1/6 neutralises Q u^c H", yq + yu + Fr(1, 6) == 0, False)

    print("\n5. One-loop coefficients and the unification-normalised angle")
    ok = True
    for n_g, claim in [(1, (Fr(-29, 3), Fr(-35, 6), Fr(43, 18))), (3, (Fr(-7), Fr(-19, 6), Fr(41, 6)))]:
        got = betas(n_g, fam)
        print(f"     N_g={n_g}: (B3, B2, BY) = ({got[0]}, {got[1]}, {got[2]}); doc {claim[0]}, {claim[1]}, {claim[2]}")
        ok &= got == claim
        ok &= got == (Fr(-11) + Fr(4, 3) * n_g, Fr(-43 + 8 * n_g, 6), Fr(40 * n_g + 3, 18))
    record("B3, B2, BY from field content equal the document's closed forms", ok, True)
    record("[wrong input] B3 with the sterile line counted as coloured",
           betas(1, fam + [("3", 1, Fr(0))])[0] == Fr(-29, 3), False)
    gy2 = Fr(3, 5)   # g_Y^2 when g_1 = g_2 = 1 and g_1^2 = (5/3) g_Y^2
    record("sin^2 theta_W = 3/8 at g_1 = g_2", gy2 / (1 + gy2) == Fr(3, 8), True)
    record("[wrong input] sin^2 theta_W = 3/8 with g_1 = g_Y (no 5/3)", Fr(1) / (1 + 1) == Fr(3, 8), False)

    mismatches = [x for x in RESULTS if x[1] != x[2]]
    print(f"\nSummary: {len(RESULTS)} checks, {len(mismatches)} mismatches with expectation")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
