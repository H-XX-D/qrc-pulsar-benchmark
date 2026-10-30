"""
=====================================================================================
QRC-PULSAR-BENCH CLI (Command-Line Interface)
=====================================================================================
Command-line interface for evaluating relativistic post-Keplerian binary formulas,
hierarchical 3-body scaling multipoles, and inspecting pulsar timing residuals.
=====================================================================================
"""

import os
import sys
import argparse
import time
from .constants import G, c, M_sun, AU
from .systems import (
    SYSTEMS,
    BinaryPulsarSystem,
    HierarchicalTripleSystem,
    CircumbinaryExoplanetSystem,
    PSR_J0337_1715,
    PSR_J0737_3039,
    PSR_B1913_16,
    PSR_J1738_0333,
    KEPLER_16B
)
from .engine import (
    PostKeplerianSolver,
    RelationalTripleEngine,
    CircumbinarySolver,
    ZenodoWalkForwardEvaluator,
    GenericPulsarDataEvaluator,
    RelationalTimingEngine
)
from .shape_sphere import verify_laplace_beltrami_orthogonality

def print_header(title: str):
    print("\n" + "=" * 84)
    print(title)
    print("=" * 84)

def print_row(label: str, value: str):
    print(f"  {label:<40} : {value}")

def cmd_list(args):
    print_header("SUPPORTED ASTROPHYSICAL BENCHMARK SYSTEMS")
    for key, sys_obj in SYSTEMS.items():
        print(f"\n[{key}]")
        print(f"  Description    : {sys_obj.description}")
        print(f"  Literature Ref : {sys_obj.literature_ref}")

def cmd_run(args):
    sys_name = args.system
    if sys_name not in SYSTEMS:
        print(f"Error: Unknown system '{sys_name}'. Available: {list(SYSTEMS.keys())}")
        sys.exit(1)

    system = SYSTEMS[sys_name]
    t0 = time.perf_counter()

    if isinstance(system, HierarchicalTripleSystem):
        print_header(f"BENCHMARK: {system.name} (Hierarchical Stellar Triple)")
        print(f"Ref: {system.literature_ref}\n")
        res = RelationalTripleEngine.solve(system)
        print_row("Hierarchy Parameter epsilon (a1/a2)", f"{res['epsilon_hierarchy']:.6f}")
        print_row("Inner Mass Asymmetry (delta_m)", f"{res['mass_asymmetry']:.6f}")
        print_row("Quadrupole Tidal Mode (l=2)", f"{res['A_quad_us']:.4f} microseconds")
        print_row("Octupole Tidal Mode (l=3)", f"{res['A_oct_ns']:.2f} nanoseconds")
        print_row("Hexadecapole Tidal Mode (l=4)", f"{res['A_hex_ns']:.2f} nanoseconds")
        print_row("2PN Relativistic Cross-Term", f"{res['delta_t_2pn_ps']:.2f} picoseconds")
        print_row("Inner GW Decay Rate (Pdot_b1)", f"{res['Pdot_b1']:.4e} s/s")
        print_row("Published SEP Bound (Archibald 2018)", f"({res['delta_sep_published']:.1e} +- {res['delta_sep_uncertainty']:.1e})")

    elif isinstance(system, CircumbinaryExoplanetSystem):
        print_header(f"BENCHMARK: {system.name} (Circumbinary Exoplanet)")
        print(f"Ref: {system.literature_ref}\n")
        res = CircumbinarySolver.solve(system)
        print_row("Predicted Nodal Precession Period", f"{res['P_nodal_prec_yr_pred']:.2f} years")
        print_row("Empirical Target (Doyle et al. 2011)", f"{res['P_nodal_prec_yr_obs']:.2f} years")
        print_row("Agreement with Discovery Paper", f"{res['agreement_pct']:.2f}%")
        print_row("Transit Timing Variation (TTV) Amplitude", f"~ {res['ttv_semi_amplitude_hours']:.2f} hours")

    else:
        print_header(f"BENCHMARK: {system.name} (Relativistic Binary Pulsar)")
        print(f"Ref: {system.literature_ref}\n")
        res = PostKeplerianSolver.solve(system)
        agreements = res["agreements_pct"]

        print_row("Periastron Advance (om_dot)", f"{res['om_dot_pred_deg_yr']:.5f} deg/yr (Target: {system.om_dot_obs_deg_yr} deg/yr -> {agreements.get('om_dot', 0.0):.4f}% agreement)")
        if system.Pdot_b_int_obs is not None:
            print_row("GW Damping (Pdot_b intrinsic)", f"{res['Pdot_b_pred']:.4e} s/s (Target: {system.Pdot_b_int_obs:.4e} s/s -> {agreements.get('Pdot_b_int', 0.0):.4f}% agreement)")
        print_row("GW Damping (Pdot_b observed)", f"{res['Pdot_b_pred']:.4e} s/s (Target: {system.Pdot_b_obs:.4e} s/s -> {agreements.get('Pdot_b', 0.0):.4f}% agreement)")
        if system.gamma_obs_ms is not None:
            print_row("Einstein Delay (gamma)", f"{res['gamma_pred_ms']:.4f} ms (Target: {system.gamma_obs_ms} ms -> {agreements.get('gamma', 0.0):.4f}% agreement)")
        if system.r_obs_us is not None:
            print_row("Shapiro Delay Range (r)", f"{res['r_pred_us']:.4f} us (Target: {system.r_obs_us} us -> {agreements.get('r', 0.0):.4f}% agreement)")
        print_row("Shapiro Delay Shape (s = sin i)", f"{res['s_pred']:.5f} (Target: {system.s_obs})")
        print_row("Geodetic Spin Precession", f"{res['omega_geod_deg_yr']:.5f} deg/yr")

    t1 = time.perf_counter()
    elapsed_us = (t1 - t0) * 1e6
    print(f"\n--> Calculation executed in {elapsed_us:.2f} microseconds")

def cmd_all(args):
    print_header("MASTER BENCHMARK ACROSS ALL 5 REFERENCE SYSTEMS")
    for key in SYSTEMS:
        args.system = key
        cmd_run(args)

    print_header("LEGENDRE POLYNOMIAL ORTHOGONALITY CHECK")
    p2_p3 = verify_laplace_beltrami_orthogonality(2, 3)
    p2_p4 = verify_laplace_beltrami_orthogonality(2, 4)
    p3_p4 = verify_laplace_beltrami_orthogonality(3, 4)
    print_row("<P_2, P_3> Orthogonality Integral", f"{p2_p3:.2e}")
    print_row("<P_2, P_4> Orthogonality Integral", f"{p2_p4:.2e}")
    print_row("<P_3, P_4> Orthogonality Integral", f"{p3_p4:.2e}")
    print("\n[ALL REFERENCE SYSTEMS AND LEGENDRE INTEGRALS EVALUATED SUCCESSFULLY]")

def cmd_zenodo(args):
    csv_path = args.path
    zenodo_url = "https://zenodo.org/api/records/1241122/files/tuned2_aligned_pulses_1400_nosun2_primary_residuals.csv/content"
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) < 20 * 1024 * 1024:
        print(f"Zenodo data file '{csv_path}' not found locally. Downloading from Zenodo DOI: 10.5281/zenodo.1241121...")
        import urllib.request
        urllib.request.urlretrieve(zenodo_url, csv_path)
        print(f"Download complete: {os.path.getsize(csv_path) / (1024*1024):.2f} MB saved to {csv_path}")

    print_header("EVALUATION OF PUBLISHED ARCHIBALD ET AL. (2018) RESIDUALS")
    print(f"Target File: {csv_path}")

    res = ZenodoWalkForwardEvaluator.evaluate(csv_path)
    print_row("Total Radio Observations", f"{res['N_total']:,}")
    print_row("Observational Baseline", f"{res['span_days']:.1f} days ({res['span_years']:.2f} years)")
    print_row("Overall Residual RMS", f"{res['overall_rms_us']:.3f} microseconds")
    print_row("Mean Residual Offset", f"{res['overall_mean_us']:+.4f} microseconds")
    print_row("Linear Drift Slope", f"{res['drift_slope_us_yr']:+.4f} microseconds/year")
    print_row("Archibald et al. (2018) Published Delta", f"({res['delta_sep_published']:.1e} +- {res['delta_sep_uncertainty']:.1e})")

    print("\nNote:")
    print(f"  {res['note']}")

    print("\nSequential Forecast Windows:")
    print(f"{'Win':<4} | {'MJD Range':<22} | {'N_Obs':<8} | {'RMS (us)':<10} | {'chi2/dof':<10} | {'Mean Offset (us)'}")
    print("-" * 75)
    for w in res["windows"]:
        print(f"{w['window']:<4} | {f'{w['mjd_start']:.1f} - {w['mjd_end']:.1f}':<22} | {w['n_obs']:<8} | {w['rms_us']:<10.3f} | {w['chi2_dof']:<10.3f} | {w['mean_drift_us']:<+16.4f}")

def cmd_custom(args):
    file_path = args.path
    unit = args.unit
    num_windows = args.windows

    print_header("EVALUATING CUSTOM PULSAR TIMING DATASET")
    print(f"Target File: {file_path}")
    print(f"Input Unit:  {unit}")
    print(f"Windows:     {num_windows}\n")

    res = GenericPulsarDataEvaluator.evaluate(file_path, num_windows=num_windows, unit=unit)
    print_row("Total Observations Loaded", f"{res['N_total']:,}")
    print_row("Temporal Span", f"{res['span_days']:.1f} days ({res['span_years']:.2f} years)")
    print_row("Overall Timing Residual RMS", f"{res['overall_rms_us']:.3f} microseconds ({res['overall_rms_us']*1000:.1f} ns)")
    print_row("Mean Residual Offset", f"{res['overall_mean_us']:+.4f} microseconds")
    print_row("Linear Drift Slope", f"{res['drift_slope_us_yr']:+.4f} microseconds/year")
    print_row("Global Reduced chi2/dof", f"{res['global_chi2_dof']:.3f}")

    print("\nInterpretation Note:")
    print("  In multi-year pulsar datasets, linear drift typically reflects unmodeled red noise or interstellar")
    print("  dispersion measure (DM) fluctuations. Straight-line fits do not detect quadratic orbital decay signals.")

    print("\nSequential Forecast Windows:")
    print(f"{'Win':<4} | {'MJD Range':<22} | {'N_Obs':<8} | {'RMS (us)':<10} | {'chi2/dof':<10} | {'Mean Offset (us)'}")
    print("-" * 75)
    for w in res["windows"]:
        print(f"{w['window']:<4} | {f'{w['mjd_start']:.1f} - {w['mjd_end']:.1f}':<22} | {w['n_obs']:<8} | {w['rms_us']:<10.3f} | {w['chi2_dof']:<10.3f} | {w['mean_drift_us']:<+16.4f}")

def cmd_compare(args):
    print_header("METHODOLOGICAL COMPARISON: ANALYTIC SCALING VS NUMERICAL INTEGRATION")
    print(f"{'Method / Approach':<30} | {'Regime of Validity':<28} | {'Characteristics'}")
    print("-" * 84)
    print(f"{'Full Numerical N-Body ODE':<30} | {'General, non-hierarchical, chaotic':<28} | {'Integrates full coupled DEs'}")
    print(f"{'Barycentric Fitters (TEMPO/PINT)':<30} | {'Multi-parameter telescope fits':<28} | {'Astrometry, DM, red noise'}")
    print(f"{'Textbook Post-Keplerian Formulas':<30} | {'Relativistic isolated binaries':<28} | {'Closed-form algebraic evaluation'}")
    print(f"{'Hierarchical Tidal Multipoles':<30} | {'Tightly hierarchical triples':<28} | {'Order-of-magnitude semi-amplitudes'}")
    print("-" * 84)
    print("\nNote on 3-Body Dynamics:")
    print("  General non-hierarchical 3-body systems are non-integrable and exhibit deterministic chaos (Poincaré 1890).")
    print("  Analytical multipole scaling relations only apply to hierarchical regimes (a1 << a2) and do not replace")
    print("  direct numerical integration of the equations of motion for general orbits.")

def main():
    parser = argparse.ArgumentParser(description="qrc-pulsar-bench: Relativistic Binary and Hierarchical Scaling Calculator")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list", help="List all supported reference systems")

    run_parser = subparsers.add_parser("run", help="Run benchmark on a specific system")
    run_parser.add_argument("--system", choices=list(SYSTEMS.keys()), default="PSR_J0737-3039", help="System to benchmark")

    subparsers.add_parser("all", help="Run master benchmark across all 5 reference systems")

    zenodo_parser = subparsers.add_parser("zenodo", help="Inspect Archibald et al. (2018) post-fit residuals")
    zenodo_parser.add_argument("--path", default="psr_j0337_zenodo_residuals.csv", help="Path to Zenodo residuals CSV file")

    subparsers.add_parser("compare", help="Methodological comparison overview")

    custom_parser = subparsers.add_parser("custom", help="Inspect custom or NANOGrav pulsar residuals")
    custom_parser.add_argument("--path", required=True, help="Path to timing dataset")
    custom_parser.add_argument("--unit", choices=["us", "ns", "s"], default="us", help="Units of input residuals (default: us)")
    custom_parser.add_argument("--windows", type=int, default=10, help="Number of forecast windows (default: 10)")

    args = parser.parse_args()
    if args.command == "list":
        cmd_list(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "all":
        cmd_all(args)
    elif args.command == "zenodo":
        cmd_zenodo(args)
    elif args.command == "custom":
        cmd_custom(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
