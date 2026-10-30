"""Command-line formula calculations and post-fit residual inspection."""

import argparse
import math
import time

from .datasets import ensure_zenodo_csv
from .systems import SYSTEMS, BinaryPulsarSystem, HierarchicalTripleSystem
from .engine import PostKeplerianSolver, RelationalTripleEngine, CircumbinarySolver
from .residuals import GenericPulsarDataEvaluator, ZenodoResidualEvaluator
from .shape_sphere import verify_laplace_beltrami_orthogonality
from .dynamics import RelationalThreeBodyIntegrator, DynamicalBenchmarkSuite


def print_header(title):
    print(f"\n{title}\n{'=' * len(title)}")


def print_row(label, value):
    print(f"  {label:<40}: {value}")


def _number(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}g}"


def _solver(system):
    if isinstance(system, BinaryPulsarSystem):
        return PostKeplerianSolver.solve
    if isinstance(system, HierarchicalTripleSystem):
        return RelationalTripleEngine.solve
    return CircumbinarySolver.solve


def cmd_list(args):
    for key, system in SYSTEMS.items():
        print(f"{key}: {system.description}\n  {system.literature_ref}")


def cmd_run(args):
    system = SYSTEMS[args.system]
    solver = _solver(system)
    start = time.perf_counter()
    result = solver(system)
    elapsed_us = (time.perf_counter() - start) * 1e6
    print_header(system.name)
    print(system.literature_ref)
    if isinstance(system, HierarchicalTripleSystem):
        for label, key, unit in [
            ("Hierarchy a1/a2", "epsilon_hierarchy", ""),
            ("Quadrupole tidal scale", "A_quad_us", "us"),
            ("Signed octupole tidal scale", "A_oct_ns", "ns"),
            ("Hexadecapole tidal scale", "A_hex_ns", "ns"),
            ("2PN delay scale", "delta_t_2pn_ps", "ps"),
            ("Inner binary GW decay estimate", "Pdot_b1", "s/s"),
            ("GW delay scale over 5.23 years", "gw_delay_scale_ns", "ns"),
        ]:
            print_row(label, f"{result[key]:.7g} {unit}")
        published = result["published_sep"]
        if published["absolute_bound_95"] is not None:
            print_row("Published SEP |Delta| bound (95%)", f"{published['absolute_bound_95']:.2g}")
            print_row("SEP reference", published["source"])
        qrc = result.get("qrc_sep_prediction")
        if qrc is not None:
            verdict = "inside" if qrc["within_published_95_bound"] else "OUTSIDE"
            print_row("QRC prediction for Delta (theory input)", f"{qrc['value']:.2g}")
            print_row("QRC vs published measurement", f"{verdict} the 95% bound; "
                      f"{qrc['pull_vs_published_sigma']:.2f} sigma from the published value")
    elif isinstance(system, BinaryPulsarSystem):
        for label, key, target, comparison, unit in [
            ("Periastron advance", "om_dot_pred_deg_yr", "om_dot_obs_deg_yr", "om_dot", "deg/yr"),
            ("Intrinsic GW decay estimate", "Pdot_b_pred", "Pdot_b_int_obs", "Pdot_b_int", "s/s"),
            ("Einstein delay", "gamma_pred_ms", "gamma_obs_ms", "gamma", "ms"),
            ("Shapiro range", "r_pred_us", "r_obs_us", "r", "us"),
        ]:
            print_row(label, f"{result[key]:.7g} {unit}")
            reference = result["targets"][target]
            if reference is not None:
                diff = result["relative_difference_pct"].get(comparison)
                print_row("  Reference / relative difference", f"{reference:.7g} {unit} / {_number(diff)}%")
        if system.Pdot_b_obs is not None:
            print_row("Observed decay (includes kinematics)", f"{system.Pdot_b_obs:.7g} s/s")
        print_row("Shapiro shape sin(i), supplied input", f"{result['s_input']:.7g}")
    else:
        print_row("Nodal precession period estimate", f"{_number(result['P_nodal_prec_yr_pred'], 7)} years")
        print_row("Supplied precession reference", f"{_number(result['P_nodal_prec_yr_obs'])} years")
        print_row("Relative difference", f"{_number(result['relative_difference_pct'])}%")
        print_row("Supplied TTV reference", f"{_number(result['ttv_reference_hours'])} hours")
    print(f"\n{result['note']}")
    print(f"Formula evaluation: {elapsed_us:.2f} microseconds (single local measurement).")


def cmd_all(args):
    for key in SYSTEMS:
        cmd_run(argparse.Namespace(system=key))
    print_header("Legendre inner products")
    for left, right in [(2, 3), (2, 4), (3, 4)]:
        print_row(f"Integral P_{left} P_{right} over [-1, 1]",
                  f"{verify_laplace_beltrami_orthogonality(left, right):.3e}")
    print("Basis orthogonality alone does not imply decoupled nonlinear dynamics.")


def _print_residuals(result):
    print_row("Observations", f"{result['N_total']:,}")
    print_row("Baseline", f"{result['span_years']:.5f} years")
    print_row("RMS about zero", f"{result['overall_rms_us']:.6f} us")
    print_row("Standard deviation about mean", f"{result['overall_std_us']:.6f} us")
    print_row("Mean residual", f"{result['overall_mean_us']:+.6f} us")
    print_row("Unweighted linear slope", f"{result['drift_slope_us_yr']:+.6f} us/year")
    print_row("Chi-square / observation", f"{result['chi2_per_observation']:.6f}")
    print(f"\n{result['note']}")
    print("A small linear slope does not establish phase stability or forecasting accuracy.")
    print("Chi-square/N is descriptive; the upstream fit's degrees of freedom and noise model are not known.")
    print("\nChronological residual windows:")
    print(f"{'Win':>3} {'MJD start':>12} {'MJD end':>12} {'N':>7} {'RMS(us)':>11} {'Std(us)':>11} {'Mean(us)':>11} {'Chi2/N':>10}")
    for w in result["windows"]:
        print(f"{w['window']:3d} {w['mjd_start']:12.3f} {w['mjd_end']:12.3f} {w['n_obs']:7d} "
              f"{_number(w['rms_us']):>11} {_number(w['std_us']):>11} {_number(w['mean_us']):>11} "
              f"{_number(w['chi2_per_observation']):>10}")


def cmd_zenodo(args):
    path = ensure_zenodo_csv(args.path)
    result = ZenodoResidualEvaluator.evaluate(path, args.windows, include_sep_diagnostic=args.sep_diagnostic)
    print_header("Archibald et al. (2018) post-fit residual inspection")
    _print_residuals(result)
    published = result["published_sep"]
    print_row("Published |Delta| bound (95%)", f"{published['absolute_bound_95']:.2g}")
    print_row("Published source", published["source"])
    if "sep_diagnostic" in result:
        d = result["sep_diagnostic"]
        print_header("Optional linearized SEP sensitivity diagnostic")
        print_row("Nuisance derivative rank", d["nuisance_rank"])
        print_row("Conditional sigma (others fixed)", _number(d["conditional_sigma"], 6))
        print_row("Marginal sigma (nuisance projected)", _number(d["marginal_sigma"], 6))
        print_row("Increment about original fit", _number(d["increment_about_reference_fit"], 6))
        print_row("Identifiable at chosen rank cutoff", d["identifiable"])
        print(d["note"])


def cmd_custom(args):
    result = GenericPulsarDataEvaluator.evaluate(args.path, args.windows, args.unit, format=args.format)
    print_header("Post-fit residual inspection")
    print_row("Input format", result["input_format"])
    _print_residuals(result)


def cmd_arecibo(args):
    result = ZenodoResidualEvaluator.evaluate(args.path, args.windows, backend="arecibo")
    print_header("Arecibo post-fit residual subset (Archibald release)")
    print("Selection: named WSRT and GBT jump indicators are both zero.")
    _print_residuals(result)
    if args.simulate:
        print("\nIndependent illustrative simulation over the same duration; no data-to-model fit or residual comparison.")
        print("The CSV supplies only the duration; simulated initial orbital phases remain arbitrary.")
        system = SYSTEMS['PSR_J0337+1715']
        cmd_simulate(argparse.Namespace(system='PSR_J0337+1715',
                     orbits=result['span_days']/system.Pb1_days, points=args.points,
                     tolerance=args.tolerance))


def cmd_nancay(args):
    raise ValueError("The Archibald release contains Arecibo, GBT and WSRT data, not Nancay. "
                     "Use 'arecibo' for the former mislabeled subset, or 'custom' for an actual Nancay residual table.")


def cmd_formula_cost(args):
    print_header("Measured local formula evaluation cost")
    print("Each row measures repeated evaluation of this package's scalar formulas.")
    print("No external timing package or trajectory integration is benchmarked.")
    for name, system in SYSTEMS.items():
        solver = _solver(system)
        solver(system)  # Warm import/runtime paths before measurement.
        start = time.perf_counter()
        for _ in range(args.repeats):
            solver(system)
        mean_us = (time.perf_counter() - start) * 1e6 / args.repeats
        print_row(name, f"{mean_us:.3f} us/call, {args.repeats} evaluations")
    print("These measurements establish no speedup for solving trajectories or predicting arrival times.")


def cmd_simulate(args):
    system = SYSTEMS[args.system]
    result = RelationalThreeBodyIntegrator.integrate(system, args.orbits, args.points,
                                                    rtol=args.tolerance, atol=args.tolerance)
    print_header(f"Newtonian time evolution: {system.name}")
    print(result['initial_conditions']['note'])
    print("Observer: fixed +z; geometric Roemer delay only. No observed timing data is fitted.")
    print_row("Span", f"{result['span_days']:.6g} days")
    print_row("Measured integration wall time", f"{result['wall_time_s']*1000:.3f} ms")
    print_row("Force evaluations", result['nfev'])
    print_row("Sampled energy relative error", _number(result['energy_conservation_err']))
    print_row("Sampled angular momentum error", _number(result['angular_momentum_conservation_err']))
    print_row("Shape norm error (geometric identity)", _number(result['sphere_invariance_err']))
    print_row("Reconstructed COM drift", f"{result['com_drift_m']:.6g} m")
    print_row("Roemer delay range", f"[{min(result['roemer_delay_us']):.6g}, {max(result['roemer_delay_us']):.6g}] us")
    print("The shape trace is unsigned, unweighted triangle geometry derived from the 12-component trajectory.")


def cmd_compare(args):
    result = DynamicalBenchmarkSuite.compare(SYSTEMS[args.system], args.orbits, args.points,
                                             tolerance=args.tolerance)
    print_header("Measured Newtonian Jacobi / Cartesian comparison")
    print("Identical illustrative initial elements; common DOP853 method; zero default orbital phases.")
    for label, key, unit in [
        ("Jacobi integration time", "relational_wall_time_ms", "ms"),
        ("Cartesian integration time", "cartesian_wall_time_ms", "ms"),
        ("Inner vector difference (maximum)", "max_inner_vector_difference_m", "m"),
        ("Outer vector difference (maximum)", "max_outer_vector_difference_m", "m"),
        ("Barycentric position difference", "max_barycentric_position_difference_m", "m"),
        ("Barycentric velocity difference", "max_barycentric_velocity_difference_m_s", "m/s"),
        ("Roemer delay difference (maximum)", "max_roemer_difference_us", "us"),
        ("Jacobi reconstructed COM drift", "relational_com_drift_m", "m"),
        ("Cartesian COM drift", "cartesian_com_drift_m", "m"),
        ("Jacobi sampled energy error", "energy_conservation_err", ""),
        ("Cartesian sampled energy error", "cartesian_energy_conservation_err", ""),
    ]:
        print_row(label, f"{_number(result[key], 7)} {unit}")
    print(result['note'])


def _positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def _positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description="Newtonian three-body dynamics, pulsar formula estimates and residual inspection")
    commands = parser.add_subparsers(dest="command")
    for name, handler, description in [
        ("list", cmd_list, "List reference systems"),
        ("all", cmd_all, "Evaluate all reference systems"),
        ("run", cmd_run, "Evaluate one reference system"),
        ("simulate", cmd_simulate, "Integrate illustrative Newtonian triple initial conditions"),
        ("compare", cmd_compare, "Compare full Jacobi and Cartesian Newtonian trajectories"),
        ("formula-cost", cmd_formula_cost, "Measure this package's scalar formula evaluation costs"),
        ("zenodo", cmd_zenodo, "Inspect published Archibald post-fit residuals"),
        ("arecibo", cmd_arecibo, "Inspect Arecibo residuals; optionally simulate their duration"),
        ("nancay", cmd_nancay, "Explain the corrected telescope attribution; no Nancay data in this release"),
        ("custom", cmd_custom, "Inspect numeric residual tables (not raw TOAs)"),
    ]:
        sub = commands.add_parser(name, help=description)
        sub.set_defaults(handler=handler)
        if name == "run":
            sub.add_argument("--system", choices=list(SYSTEMS), default="PSR_J0737-3039")
        elif name == "formula-cost":
            sub.add_argument("--repeats", type=_positive_int, default=1000)
        elif name in ("simulate", "compare"):
            triples = [key for key, system in SYSTEMS.items() if isinstance(system, HierarchicalTripleSystem)]
            sub.add_argument("--system", choices=triples, default="PSR_J0337+1715")
            sub.add_argument("--orbits", type=_positive_float, default=5.)
            sub.add_argument("--points", type=_positive_int, default=250)
            sub.add_argument("--tolerance", type=_positive_float, default=1e-10)
        elif name in ("zenodo", "custom", "arecibo", "nancay"):
            sub.add_argument("--path", required=name == "custom", default="psr_j0337_zenodo_residuals.csv")
            sub.add_argument("--windows", type=_positive_int, default=10, help="Chronological windows")
            if name == "zenodo":
                sub.add_argument("--sep-diagnostic", action="store_true", help="Report linearized nuisance-aware sensitivity, not an absolute SEP measurement")
            elif name in ("arecibo", "nancay"):
                sub.add_argument("--simulate", action="store_true", help="Also simulate illustrative initial conditions over the data's duration; this is not a timing fit")
                sub.add_argument("--points", type=_positive_int, default=1000)
                sub.add_argument("--tolerance", type=_positive_float, default=1e-10)
            else:
                sub.add_argument("--unit", choices=["us", "ns", "s"], default="us")
                sub.add_argument("--format", choices=["auto", "standard", "nanograv"], default="auto")
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return
    try:
        args.handler(args)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
