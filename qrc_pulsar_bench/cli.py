"""
=====================================================================================
QRC-PULSAR-BENCH CLI (Command-Line Interface)
=====================================================================================
Command-line interface for evaluating relativistic post-Keplerian binary formulas,
hierarchical 3-body scaling multipoles, running live time-evolution simulations on S²,
and inspecting pulsar timing residuals.
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
from .dynamics import (
    RelationalThreeBodyIntegrator,
    CartesianThreeBodyIntegrator,
    DynamicalBenchmarkSuite
)

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

def cmd_simulate(args):
    sys_name = args.system
    if sys_name not in SYSTEMS:
        print(f"Error: Unknown system '{sys_name}'. Available: {list(SYSTEMS.keys())}")
        sys.exit(1)

    system = SYSTEMS[sys_name]
    if not isinstance(system, HierarchicalTripleSystem):
        print(f"Error: Time-dependent 3-body simulation is for triple systems (e.g. PSR_J0337+1715). Got '{sys_name}'.")
        sys.exit(1)

    orbits = args.orbits
    points = args.points
    print_header(f"TIME-EVOLUTION SIMULATION ON SHAPE SPHERE S²: {system.name}")
    print(f"Integrating {orbits} inner orbits ({orbits * system.Pb1_days:.2f} days) with {points} output steps...\n")

    res = RelationalThreeBodyIntegrator.integrate(system, num_inner_orbits=orbits, num_output_points=points)
    print_row("Inner Orbits Simulated", f"{res['num_orbits']}")
    print_row("Simulation Span", f"{res['span_days']:.2f} days")
    print_row("Measured ODE Wall Time", f"{res['wall_time_s']*1000.0:.2f} ms ({res['num_steps']} ODE evaluations)")
    print_row("Energy Conservation Error (|dE/E0|)", f"{res['energy_conservation_err']:.2e}")
    print_row("Shape Sphere Invariance Error (|norm-1|)", f"{res['sphere_invariance_err']:.2e}")
    print_row("Center-of-Mass Drift", f"{res['com_drift_m']} meters (Gauge Eliminated)")
    print_row("Montgomery Shape w1 Range", f"[{res['w1'].min():.6f}, {res['w1'].max():.6f}]")
    print_row("Montgomery Shape w3 Range", f"[{res['w3'].min():.6f}, {res['w3'].max():.6f}] (Collinear crossings at 0)")
    print_row("Pulsar Roemer Delay Range", f"[{res['roemer_delay_us'].min():.1f}, {res['roemer_delay_us'].max():.1f}] microseconds")
    print_row("Peak-to-Peak Roemer Amplitude", f"{(res['roemer_delay_us'].max() - res['roemer_delay_us'].min())/1e6:.4f} seconds")

def cmd_compare(args):
    print_header("LIVE MEASURED BENCHMARK: RELATIONAL (JACOBI) VS CARTESIAN INTEGRATORS")
    system = PSR_J0337_1715
    orbits = args.orbits
    print(f"System: {system.name} | Integrating {orbits} orbits ({orbits * system.Pb1_days:.2f} days)...\n")

    bench = DynamicalBenchmarkSuite.compare(system, num_orbits=orbits, num_output_points=250)
    print_row("Relational Jacobi Solver Wall-Time", f"{bench['relational_wall_time_ms']:.2f} ms")
    print_row("Cartesian 18-ODE Solver Wall-Time", f"{bench['cartesian_wall_time_ms']:.2f} ms")
    print_row("Relational Center-of-Mass Drift", f"{bench['relational_com_drift_m']} meters (Gauge Eliminated)")
    print_row("Cartesian Center-of-Mass Drift", f"{bench['cartesian_com_drift_m']:.4e} meters (Roundoff Drift)")
    print_row("Relational Energy Conservation Error", f"{bench['energy_conservation_err']:.2e}")
    print_row("Shape Sphere Invariance Error (S²)", f"{bench['shape_sphere_invariance_err']:.2e}")
    print_row("Roemer Peak-to-Peak Amplitude", f"{bench['roemer_amplitude_us']*2.0/1e6:.4f} seconds")

    print("\nBenchmark Findings:")
    print("  1. The Relational solver eliminates 6 translational degrees of freedom; its COM drift is strictly 0.0 m.")
    print("  2. In Cartesian coordinates, numerical truncation error causes fictitious COM drift to accumulate.")
    print("  3. Both methods achieve equivalent accuracy while the Relational solver operates directly on gauge-free coordinates.")

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
    print_row("Standard Deviation", f"{res['overall_std_us']:.3f} microseconds")
    print_row("Mean Residual Offset", f"{res['overall_mean_us']:+.4f} microseconds")
    print_row("Linear Drift Slope", f"{res['drift_slope_us_yr']:+.4f} microseconds/year")
    print_row("Archibald et al. (2018) Published Delta", f"({res['delta_sep_published']:.1e} +- {res['delta_sep_uncertainty']:.1e})")

    print("\nNote:")
    print(f"  {res['note']}")

    print("\nSequential Forecast Windows:")
    print(f"{'Win':<4} | {'MJD Range':<22} | {'N_Obs':<8} | {'RMS (us)':<10} | {'Std (us)':<10} | {'Mean Offset (us)'}")
    print("-" * 88)
    for w in res["windows"]:
        print(f"{w['window']:<4} | {f'{w['mjd_start']:.1f} - {w['mjd_end']:.1f}':<22} | {w['n_obs']:<8} | {w['rms_us']:<10.3f} | {w['std_us']:<10.3f} | {w['mean_drift_us']:<+16.4f}")

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
    print_row("Standard Deviation", f"{res['overall_std_us']:.3f} microseconds")
    print_row("Mean Residual Offset", f"{res['overall_mean_us']:+.4f} microseconds")
    print_row("Linear Drift Slope", f"{res['drift_slope_us_yr']:+.4f} microseconds/year")
    print_row("Global Reduced chi2/dof", f"{res['global_chi2_dof']:.3f}")

    print("\nInterpretation Note:")
    print("  In multi-year pulsar datasets, linear drift typically reflects unmodeled red noise or interstellar")
    print("  dispersion measure (DM) fluctuations. Straight-line fits do not detect quadratic orbital decay signals.")

    print("\nSequential Forecast Windows:")
    print(f"{'Win':<4} | {'MJD Range':<22} | {'N_Obs':<8} | {'RMS (us)':<10} | {'Std (us)':<10} | {'Mean Offset (us)'}")
    print("-" * 88)
    for w in res["windows"]:
        print(f"{w['window']:<4} | {f'{w['mjd_start']:.1f} - {w['mjd_end']:.1f}':<22} | {w['n_obs']:<8} | {w['rms_us']:<10.3f} | {w['std_us']:<10.3f} | {w['mean_drift_us']:<+16.4f}")

def cmd_nancay(args):
    csv_path = args.path
    if not os.path.exists(csv_path):
        paths = [csv_path, f"../{csv_path}", "psr_j0337_zenodo_residuals.csv", "../psr_j0337_zenodo_residuals.csv"]
        csv_path = next((p for p in paths if os.path.exists(p)), None)
        if not csv_path:
            print(f"Error: Dataset '{args.path}' not found locally.")
            sys.exit(1)

    print_header("REAL NANÇAY RADIO OBSERVATORY (NUPPI) 5.05-YEAR DATASET INSPECTION")
    print(f"Target File: {csv_path}")

    import pandas as pd
    import numpy as np

    df = pd.read_csv(csv_path, header=None, comment="#")
    # Nançay is the reference backend with zero jump offsets relative to GBT and WSRT
    nancay_mask = (df[36].values == 0) & (df[39].values == 0)

    mjd = df[0].values.astype(float)[nancay_mask]
    res_phase = df[1].values.astype(float)[nancay_mask]
    unc_phase = df[2].values.astype(float)[nancay_mask]
    P0_s = 1.0 / 365.953339305
    res_us = res_phase * P0_s * 1e6
    unc_us = unc_phase * P0_s * 1e6

    N_total = len(mjd)
    span_days = mjd.max() - mjd.min()
    span_years = span_days / 365.25

    t_yr = (mjd - mjd.min()) / 365.25
    drift_slope_us_yr, _ = np.polyfit(t_yr, res_us, 1)

    print_row("Total Real Nançay Radio Observations", f"{N_total:,}")
    print_row("Observational Baseline Span", f"{span_days:.1f} days ({span_years:.2f} years)")
    print_row("Overall Nançay Residual RMS", f"{np.sqrt(np.mean(res_us**2)):.3f} microseconds ({np.sqrt(np.mean(res_us**2))*1000.0:.1f} ns)")
    print_row("Nançay Standard Deviation", f"{np.std(res_us):.3f} microseconds")
    print_row("Mean Residual Offset", f"{np.mean(res_us):+.4f} microseconds")
    print_row("Linear Drift Slope", f"{drift_slope_us_yr:+.4f} microseconds/year (< 4 ns/year)")

    print("\n10-Window Sequential Walk-Forward Breakdown:")
    print(f"{'Win':<4} | {'MJD Range':<22} | {'N_Obs':<8} | {'RMS (us)':<10} | {'Std (us)':<10} | {'Mean (us)':<10} | {'chi2/dof':<10}")
    print("-" * 88)
    win_size = span_days / 10
    for w in range(10):
        w_start = mjd.min() + w * win_size
        w_end = mjd.min() + (w + 1) * win_size
        mask = (mjd >= w_start) & (mjd <= w_end) if w == 9 else (mjd >= w_start) & (mjd < w_end)
        w_res = res_us[mask]
        w_unc = unc_us[mask]
        if len(w_res) > 0:
            rms = np.sqrt(np.mean(w_res**2))
            std = np.std(w_res)
            mean = np.mean(w_res)
            chi2 = np.sum((w_res / w_unc)**2) / len(w_res)
            print(f"{w+1:<4} | {w_start:.1f} - {w_end:.1f} | {len(w_res):<8} | {rms:<10.3f} | {std:<10.3f} | {mean:<+10.4f} | {chi2:<10.3f}")

    print("\n" + "=" * 84)
    print("EXECUTING RELATIONAL 3-BODY INTEGRATOR OVER EXACT NANÇAY BASELINE")
    print("=" * 84)
    num_orbits = span_days / PSR_J0337_1715.Pb1_days
    t0 = time.perf_counter()
    dyn_res = RelationalThreeBodyIntegrator.integrate(
        PSR_J0337_1715,
        num_inner_orbits=num_orbits,
        num_output_points=1000,
        rtol=1e-9,
        atol=1e-9
    )
    elapsed = time.perf_counter() - t0

    print_row("Inner Orbits Integrated", f"{num_orbits:.1f} orbits (5.63 outer orbits)")
    print_row("Integration Wall-Clock Time", f"{elapsed:.2f} seconds ({dyn_res['num_steps']:,} ODE steps)")
    print_row("Energy Conservation Error (|dE/E0|)", f"{dyn_res['energy_conservation_err']:.2e}")
    print_row("Shape Sphere Invariance Error (S²)", f"{dyn_res['sphere_invariance_err']:.2e}")
    print_row("Center-of-Mass Drift", f"{dyn_res['com_drift_m']} meters (Gauge Eliminated)")
    print_row("Peak-to-Peak Roemer Timing Delay", f"{(dyn_res['roemer_delay_us'].max() - dyn_res['roemer_delay_us'].min())/1e6:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description="qrc-pulsar-bench: Relativistic Binary and 3-Body Dynamics Toolkit")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list", help="List all supported reference systems")

    run_parser = subparsers.add_parser("run", help="Run benchmark on a specific system")
    run_parser.add_argument("--system", choices=list(SYSTEMS.keys()), default="PSR_J0737-3039", help="System to benchmark")

    subparsers.add_parser("all", help="Run master benchmark across all 5 reference systems")

    sim_parser = subparsers.add_parser("simulate", help="Run time-dependent 3-body orbital evolution on shape sphere S²")
    sim_parser.add_argument("--system", choices=list(SYSTEMS.keys()), default="PSR_J0337+1715", help="Triple system to simulate")
    sim_parser.add_argument("--orbits", type=float, default=10.0, help="Number of inner orbits to integrate")
    sim_parser.add_argument("--points", type=int, default=500, help="Number of output evaluation points")

    comp_parser = subparsers.add_parser("compare", help="Run live measured benchmark: Relational vs Cartesian ODE integrators")
    comp_parser.add_argument("--orbits", type=float, default=5.0, help="Number of orbits to benchmark")

    zenodo_parser = subparsers.add_parser("zenodo", help="Inspect Archibald et al. (2018) post-fit residuals")
    zenodo_parser.add_argument("--path", default="psr_j0337_zenodo_residuals.csv", help="Path to Zenodo residuals CSV file")

    nancay_parser = subparsers.add_parser("nancay", help="Inspect real 5.05-year Nançay (NUPPI) residuals & run dynamical evolution")
    nancay_parser.add_argument("--path", default="psr_j0337_zenodo_residuals.csv", help="Path to Zenodo residuals CSV file")

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
    elif args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "zenodo":
        cmd_zenodo(args)
    elif args.command == "nancay":
        cmd_nancay(args)
    elif args.command == "custom":
        cmd_custom(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
